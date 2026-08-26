"""
BS-RoFormer SW 6-stem separation — a drop-in alternative to htdemucs_6s.

Produces the same stem set STRUM's downstream stages already expect
(vocals, drums, bass, guitar, piano, other) but from a RoFormer rather than a
hybrid-transformer Demucs. In practice it holds up better on dense mixes, which
matters most for the `guitar` and `other` stems that feed basic-pitch.

Weights (BS-Rofo-SW, "fixed" release):
    ckpt   bs_6stem_fixed.ckpt
    config bs_6stem_fixed_config.yaml
    https://huggingface.co/jarredou/BS-ROFO-SW-Fixed

There is no pip package, so inference goes through a ZFTurbo MSST checkout —
see src/preprocessing/msst.py. Select it at run time with STRUM_SEPARATOR=bs_roformer_sw.
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

    if cfg.ckpt is None or cfg.cfg is None:
        logger.warning(
            "  ⚠ BS-RoFormer selected but STRUM_ROFORMER_CKPT / STRUM_ROFORMER_CFG "
            "are unset; falling back to Demucs"
        )
        return {}

    work = output_dir / "_roformer_tmp"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    logger.info("  Separating stems with BS-RoFormer SW (6-stem)...")
    t0 = time.time()
    try:
        produced = run_msst(
            audio_path, work,
            model_type=cfg.model_type, ckpt=cfg.ckpt, config=cfg.cfg,
            msst_dir=cfg.msst_dir, device_ids=cfg.device_ids, timeout_s=cfg.timeout_s,
        )
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
