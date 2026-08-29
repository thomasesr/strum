"""
BS-RoFormer SW 6-stem separation — a drop-in alternative to htdemucs_6s.

Produces the same stem set STRUM's downstream stages already expect
(vocals, drums, bass, guitar, piano, other) but from a RoFormer rather than a
hybrid-transformer Demucs. In practice it holds up better on dense mixes, which
matters most for the `guitar` and `other` stems that feed basic-pitch.

Runs through `audio-separator`, which carries this checkpoint in its registry as
`BS-Roformer-SW.ckpt` and downloads it on demand — the same path the karaoke
model already uses.

MSST is supported as a fallback for checkpoints the registry does not carry, but
is no longer the default: its inference entry point imports bitsandbytes, peft
and loralib unconditionally, and peft pulls numpy 2.x while TensorFlow 2.15 and
audio-separator both pin numpy<2. Installing it would destabilise the image for
one optional backend.

Select this backend with STRUM_SEPARATOR=bs_roformer_sw.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Canonical order/names STRUM's downstream stages key off. Matches htdemucs_6s.
ROFORMER_STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]

# Some 6-stem configs label the keyboard stem `keys`; normalise it to `piano`
# so callers never have to branch on which checkpoint produced the stems.
STEM_ALIASES = {"piano": ["piano", "keys", "keyboard"]}


@dataclass
class RoformerConfig:
    """Settings for the BS-RoFormer SW 6-stem pass."""

    # ONNX export: the only backend that can serve this model here, since every
    # audio-separator version carrying it requires numpy>=2. See roformer_onnx.
    onnx_model: Path | None = None
    # audio-separator registry name, for versions new enough to carry it.
    model: str = "BS-Roformer-SW.ckpt"
    model_dir: Path | None = None
    ckpt: Path | None = None
    cfg: Path | None = None
    msst_dir: Path | None = None
    model_type: str = "bs_roformer"
    device_ids: str = "0"
    timeout_s: int = 2400

    @classmethod
    def from_env(cls, **overrides) -> "RoformerConfig":
        def _path(name: str) -> Path | None:
            raw = os.environ.get(name)
            return Path(raw) if raw else None

        c = cls(**overrides)
        c.onnx_model = _path("STRUM_ROFORMER_ONNX") or c.onnx_model
        c.model = os.environ.get("STRUM_ROFORMER_MODEL", c.model)
        c.model_dir = _path("STRUM_ROFORMER_MODEL_DIR") or _path("STRUM_KARAOKE_MODEL_DIR") or c.model_dir
        c.ckpt = _path("STRUM_ROFORMER_CKPT") or c.ckpt
        c.cfg = _path("STRUM_ROFORMER_CFG") or c.cfg
        c.msst_dir = _path("STRUM_ROFORMER_MSST") or c.msst_dir
        c.model_type = os.environ.get("STRUM_ROFORMER_MODEL_TYPE", c.model_type)
        c.device_ids = os.environ.get("STRUM_ROFORMER_DEVICE_IDS", c.device_ids)
        return c


def _resolve_aliases(found: dict[str, Path], produced: list[Path]) -> dict[str, Path]:
    """Fill in canonical stems that the checkpoint named differently."""
    claimed = set(found.values())
    for canonical, aliases in STEM_ALIASES.items():
        if canonical in found:
            continue
        for alias in aliases:
            hit = next(
                (p for p in produced if p not in claimed and alias in p.name.lower()), None
            )
            if hit is not None:
                found[canonical] = hit
                claimed.add(hit)
                break
    return found


def _run_audio_separator(audio_path: Path, out_dir: Path, cfg: "RoformerConfig") -> list[Path]:
    """Separate with audio-separator and return the files it wrote.

    Frees the model afterwards: at ~700 MB it would otherwise stay resident for
    the rest of the job, which on a 6 GB card comes straight out of the budget
    for the transcription models that follow.
    """
    from audio_separator.separator import Separator

    from src.preprocessing.gpu import free_gpu

    kwargs: dict = {
        "output_dir": str(out_dir),
        "output_format": "WAV",
        "log_level": logging.WARNING,
    }
    if cfg.model_dir is not None:
        kwargs["model_file_dir"] = str(cfg.model_dir)

    sep = Separator(**kwargs)
    sep.load_model(model_filename=cfg.model)
    try:
        outputs = sep.separate(str(audio_path))
    finally:
        del sep
        free_gpu("BS-RoFormer separation")

    return [p if (p := Path(o)).is_absolute() else out_dir / p.name for o in outputs]


def separate_6stem_roformer(
    audio_path: Path,
    output_dir: Path,
    config: RoformerConfig | None = None,
) -> dict[str, Path]:
    """Separate `audio_path` into the six canonical stems with BS-RoFormer SW.

    Returns a dict of stem name -> path in `output_dir`, or an empty dict if the
    model is unavailable or inference fails. Callers must treat `{}` as "fall
    back to Demucs" so a missing checkout never breaks a run.
    """
    from src.preprocessing.msst import classify_stems, run_msst

    cfg = config or RoformerConfig.from_env()
    audio_path, output_dir = Path(audio_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only the four core stems gate the cache: `guitar` and `piano` are legitimately
    # absent when a song has neither, and requiring them would rerun every time.
    core = ["vocals", "drums", "bass", "other"]
    if all((output_dir / f"{s}.wav").exists() for s in core):
        logger.info(f"  BS-RoFormer cache hit: {output_dir}")
        return {
            s: output_dir / f"{s}.wav"
            for s in ROFORMER_STEMS
            if (output_dir / f"{s}.wav").exists()
        }

    # ONNX first: it is the only backend that works alongside basic-pitch, which
    # pins numpy<2. An explicit MSST checkpoint overrides it for anyone who has
    # a working checkout; audio-separator is the last resort and only succeeds
    # on versions new enough to list the model.
    if cfg.onnx_model and Path(cfg.onnx_model).exists():
        backend = "onnx"
    elif cfg.ckpt and cfg.cfg:
        backend = "msst"
    else:
        backend = "audio-separator"

    work = output_dir / "_roformer_tmp"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Separating stems with BS-RoFormer SW ({backend})...")
    t0 = time.time()
    try:
        if backend == "onnx":
            from src.preprocessing.roformer_onnx import separate as onnx_separate

            def _log_progress(done: int, total: int) -> None:
                if done == 1 or done == total or done % 20 == 0:
                    logger.info(f"    BS-RoFormer chunk {done}/{total}")

            stems = onnx_separate(
                audio_path, output_dir, Path(cfg.onnx_model), on_progress=_log_progress
            )
            logger.info(f"    BS-RoFormer done in {time.time() - t0:.1f}s -> {sorted(stems)}")
            return stems

        if backend == "msst":
            produced = run_msst(
                audio_path, work,
                model_type=cfg.model_type, ckpt=cfg.ckpt, config=cfg.cfg,
                msst_dir=cfg.msst_dir, device_ids=cfg.device_ids, timeout_s=cfg.timeout_s,
            )
        else:
            produced = _run_audio_separator(audio_path, work, cfg)
        found = _resolve_aliases(classify_stems(produced, ROFORMER_STEMS), produced)
        missing = [s for s in core if s not in found]
        if missing:
            raise RuntimeError(f"BS-RoFormer produced no {missing} stem(s)")

        stems: dict[str, Path] = {}
        for name, src in found.items():
            dst = output_dir / f"{name}.wav"
            shutil.move(str(src), dst)
            stems[name] = dst
    except Exception as e:
        logger.warning(f"  ⚠ BS-RoFormer separation failed, falling back to Demucs: {e}")
        return {}
    finally:
        shutil.rmtree(work, ignore_errors=True)

    logger.info(f"    BS-RoFormer done in {time.time() - t0:.1f}s -> {sorted(stems)}")
    return stems
