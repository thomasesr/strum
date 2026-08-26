"""
Mel-Band RoFormer karaoke pre-separation.

Runs BEFORE Demucs. A karaoke model splits the mix into:

    lead_vocals   -- the main/lead vocal only
    instrumental  -- everything else, INCLUDING backing vocals

Rationale: the lead vocal is the loudest and most spectrally dominant source
in a typical mix, and Demucs' vocal branch is markedly weaker than a dedicated
Mel-Band RoFormer. Stripping the lead vocal first means Demucs spends all of
its capacity on the instruments, which cuts vocal bleed into the `other` /
`guitar` / `piano` stems. That bleed is the dominant source of phantom notes
when basic-pitch transcribes those stems -- a sung melody reads as a very
plausible guitar line.

Karaoke models are trained to remove *only* the lead vocal, so backing vocals
survive into `instrumental` by design. Set `backing_split=True` to run a second
full-vocal pass that lifts those backing vocals into their own stem, leaving
Demucs a completely vocal-free mix (recommended -- see docs/KARAOKE.md).

Two backends:
  * "audio-separator" (default) -- pip package, auto-downloads UVR weights.
  * "msst" -- subprocess into a ZFTurbo Music-Source-Separation-Training
    checkout, same pattern as the Jarredou drumsep pass in
    scripts/batch_infer_hybrid.py. Use this to run becruily/gabox karaoke
    checkpoints that are not in the audio-separator registry.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# UVR registry name -- present in audio-separator's built-in model list.
DEFAULT_KARAOKE_MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
# Used by the optional second pass that isolates backing vocals.
DEFAULT_VOCALS_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

STEM_LEAD = "lead_vocals"
STEM_INSTRUMENTAL = "instrumental"
STEM_BACKING = "backing_vocals"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


@dataclass
class KaraokeConfig:
    """Resolved karaoke pre-separation settings.

    Every field is overridable by a STRUM_KARAOKE_* environment variable so the
    pass can be tuned per-run without touching configs/preprocessing.yaml.
    """

    enabled: bool = True
    backend: str = "audio-separator"
    model: str = DEFAULT_KARAOKE_MODEL
    # Second pass: pull backing vocals out of `instrumental` into their own stem.
    backing_split: bool = False
    backing_model: str = DEFAULT_VOCALS_MODEL
    backing_ckpt: Path | None = None
    backing_cfg: Path | None = None
    # msst backend only
    msst_dir: Path | None = None
    ckpt: Path | None = None
    cfg: Path | None = None
    model_type: str = "mel_band_roformer"
    model_dir: Path | None = None
    timeout_s: int = 1800

    @classmethod
    def from_env(cls, **overrides) -> "KaraokeConfig":
        def _path(name: str) -> Path | None:
            raw = os.environ.get(name)
            return Path(raw) if raw else None

        cfg = cls(**overrides)
        cfg.enabled = _env_flag("STRUM_KARAOKE", cfg.enabled)
        cfg.backend = os.environ.get("STRUM_KARAOKE_BACKEND", cfg.backend)
        cfg.model = os.environ.get("STRUM_KARAOKE_MODEL", cfg.model)
        cfg.backing_split = _env_flag("STRUM_KARAOKE_BACKING_SPLIT", cfg.backing_split)
        cfg.backing_model = os.environ.get("STRUM_KARAOKE_BACKING_MODEL", cfg.backing_model)
        cfg.msst_dir = _path("STRUM_KARAOKE_MSST") or cfg.msst_dir
        cfg.ckpt = _path("STRUM_KARAOKE_CKPT") or cfg.ckpt
        cfg.cfg = _path("STRUM_KARAOKE_CFG") or cfg.cfg
        cfg.model_type = os.environ.get("STRUM_KARAOKE_MODEL_TYPE", cfg.model_type)
        cfg.model_dir = _path("STRUM_KARAOKE_MODEL_DIR") or cfg.model_dir
        cfg.backing_ckpt = _path("STRUM_KARAOKE_BACKING_CKPT") or cfg.backing_ckpt
        cfg.backing_cfg = _path("STRUM_KARAOKE_BACKING_CFG") or cfg.backing_cfg
        return cfg


def _classify(paths: list[Path]) -> dict[str, Path]:
    """Bucket separator outputs into vocal / instrumental by filename.

    Checked instrumental-first: UVR names files like `foo_(Instrumental)_model.wav`
    and `foo_(Vocals)_model.wav`, and our own `lead_vocals` hint also matches
    "vocal", so the more specific test has to win.
    """
    out: dict[str, Path] = {}
    for p in paths:
        low = p.name.lower()
        if "instrumental" in low or low.startswith("instr"):
            out["instrumental"] = p
        elif "vocal" in low:
            out["vocals"] = p
    return out


def _run_audio_separator(
    audio_path: Path,
    out_dir: Path,
    model: str,
    model_dir: Path | None,
    vocal_name: str,
    inst_name: str,
) -> dict[str, Path]:
    from audio_separator.separator import Separator

    kwargs: dict = {
        "output_dir": str(out_dir),
        "output_format": "WAV",
        "log_level": logging.WARNING,
    }
    if model_dir is not None:
        kwargs["model_file_dir"] = str(model_dir)

    sep = Separator(**kwargs)
    sep.load_model(model_filename=model)

    custom = {"Vocals": vocal_name, "Instrumental": inst_name}
    try:
        outputs = sep.separate(str(audio_path), custom_output_names=custom)
    except TypeError:
        # Older audio-separator has no custom_output_names; fall back to
        # whatever it names them and classify after the fact.
        outputs = sep.separate(str(audio_path))

    paths = [p if (p := Path(o)).is_absolute() else out_dir / p.name for o in outputs]
    return _classify(paths)


def _run_msst(
    audio_path: Path,
    out_dir: Path,
    cfg: KaraokeConfig,
    ckpt: Path,
    conf: Path,
) -> dict[str, Path]:
    """Run a karaoke checkpoint through a ZFTurbo MSST checkout.

    Use this for becruily/gabox karaoke weights, which are not in the
    audio-separator registry.
    """
    from src.preprocessing.msst import run_msst

    produced = run_msst(
        audio_path, out_dir,
        model_type=cfg.model_type, ckpt=ckpt, config=conf,
        msst_dir=cfg.msst_dir, timeout_s=cfg.timeout_s,
    )
    return _classify(produced)


def _dispatch(
    audio_path: Path,
    work_dir: Path,
    cfg: KaraokeConfig,
    model: str,
    ckpt: Path | None,
    conf: Path | None,
) -> dict[str, Path]:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if cfg.backend == "msst":
        if ckpt is None or conf is None:
            raise ValueError("msst backend needs STRUM_KARAOKE_CKPT and STRUM_KARAOKE_CFG")
        return _run_msst(audio_path, work_dir, cfg, ckpt, conf)
    if cfg.backend == "audio-separator":
        return _run_audio_separator(
            audio_path, work_dir, model, cfg.model_dir, STEM_LEAD, STEM_INSTRUMENTAL
        )
    raise ValueError(f"unknown karaoke backend: {cfg.backend!r}")


def separate_karaoke(
    audio_path: Path,
    output_dir: Path,
    config: KaraokeConfig | None = None,
) -> dict[str, Path]:
    """Strip the lead vocal from `audio_path` before Demucs sees it.

    Returns a dict with keys `lead_vocals` and `instrumental` (plus
    `backing_vocals` when `backing_split` is on). Returns an empty dict when the
    pass is disabled or fails -- callers must treat that as "feed Demucs the raw
    mix", so a missing model never breaks the pipeline.
    """
    cfg = config or KaraokeConfig.from_env()
    if not cfg.enabled:
        return {}

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lead = output_dir / f"{STEM_LEAD}.wav"
    inst = output_dir / f"{STEM_INSTRUMENTAL}.wav"
    backing = output_dir / f"{STEM_BACKING}.wav"

    # Cache hit: the karaoke pass is the most expensive stage in the pipeline
    # after Demucs, and reruns on the same song are common during charting.
    wanted = [lead, inst] + ([backing] if cfg.backing_split else [])
    if all(p.exists() for p in wanted):
        logger.info(f"  Karaoke cache hit: {output_dir}")
        return {p.stem: p for p in wanted}

    logger.info(f"  Karaoke pre-separation ({cfg.backend}: {cfg.model})...")
    t0 = time.time()
    try:
        res = _dispatch(audio_path, output_dir / "_karaoke_tmp", cfg, cfg.model, cfg.ckpt, cfg.cfg)
        if "vocals" not in res or "instrumental" not in res:
            raise RuntimeError(f"karaoke model produced unexpected stems: {sorted(res)}")
        shutil.move(str(res["vocals"]), lead)
        shutil.move(str(res["instrumental"]), inst)
    except Exception as e:
        logger.warning(f"  ⚠ Karaoke pre-separation failed, falling back to raw mix: {e}")
        shutil.rmtree(output_dir / "_karaoke_tmp", ignore_errors=True)
        return {}
    shutil.rmtree(output_dir / "_karaoke_tmp", ignore_errors=True)

    stems = {STEM_LEAD: lead, STEM_INSTRUMENTAL: inst}

    # Optional second pass. The karaoke model deliberately leaves backing vocals
    # in `instrumental`; a full-vocal model lifts them into their own stem so
    # Demucs gets a genuinely vocal-free mix and HARM2/HARM3 get a clean source.
    if cfg.backing_split:
        try:
            res2 = _dispatch(
                inst, output_dir / "_backing_tmp", cfg,
                cfg.backing_model, cfg.backing_ckpt, cfg.backing_cfg,
            )
            if "vocals" in res2 and "instrumental" in res2:
                shutil.move(str(res2["vocals"]), backing)
                clean = output_dir / "_instrumental_clean.wav"
                shutil.move(str(res2["instrumental"]), clean)
                clean.replace(inst)
                stems[STEM_BACKING] = backing
            else:
                logger.warning(f"  ⚠ Backing-vocal split produced: {sorted(res2)}; skipping")
        except Exception as e:
            logger.warning(f"  ⚠ Backing-vocal split failed, backing vox stay in mix: {e}")
        shutil.rmtree(output_dir / "_backing_tmp", ignore_errors=True)

    logger.info(f"    Karaoke pass done in {time.time() - t0:.1f}s -> {sorted(stems)}")
    return stems
