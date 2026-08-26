"""
Thin wrapper around a ZFTurbo Music-Source-Separation-Training checkout.

MSST (`inference.py`) is how STRUM runs RoFormer-family checkpoints that have no
pip package: the Jarredou 6-stem drumsep model, Mel-Band RoFormer karaoke models,
and BS-RoFormer SW. It is a CLI, not a library, so everything goes through a
subprocess — same pattern as separate_drums_6stem() in
scripts/batch_infer_hybrid.py, generalised here so each caller only has to
supply a model type, a checkpoint and a config.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MSST_DIR = Path(os.environ.get("STRUM_MSST_DIR", "/home/opria123/drumsep/msst"))


def resolve_msst_dir(explicit: Path | None = None) -> Path:
    """Locate the MSST checkout: explicit arg > STRUM_MSST_DIR > legacy default."""
    if explicit is not None:
        return Path(explicit)
    return DEFAULT_MSST_DIR


def run_msst(
    audio_path: Path,
    out_dir: Path,
    model_type: str,
    ckpt: Path,
    config: Path,
    msst_dir: Path | None = None,
    device_ids: str = "0",
    timeout_s: int = 1800,
    extra_args: list[str] | None = None,
) -> list[Path]:
    """Separate `audio_path` with one MSST checkpoint.

    Returns every .wav MSST produced, in sorted order. Callers classify those by
    filename, because MSST's output layout varies by version: some write
    `<store_dir>/<basename>/<stem>.wav`, others `<store_dir>/<basename>_<stem>.wav`.

    Raises FileNotFoundError if the checkout or weights are missing, and
    CalledProcessError / TimeoutExpired if inference itself fails. Callers are
    expected to catch and fall back rather than let a missing model break a run.
    """
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    msst_dir = resolve_msst_dir(msst_dir)
    ckpt, config = Path(ckpt), Path(config)

    if not (msst_dir / "inference.py").exists():
        raise FileNotFoundError(f"MSST checkout has no inference.py: {msst_dir}")
    missing = [str(p) for p in (ckpt, config) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"MSST weights missing: {', '.join(missing)}")

    stage_in = out_dir / "_msst_in"
    stage_out = out_dir / "_msst_out"
    for d in (stage_in, stage_out):
        if d.exists():
            shutil.rmtree(d)
    stage_in.mkdir(parents=True, exist_ok=True)

    # MSST takes a folder of inputs, so stage a single symlink (copy on
    # filesystems that refuse links) rather than pointing it at the song dir.
    staged = stage_in / f"mix{audio_path.suffix or '.wav'}"
    try:
        os.symlink(audio_path.resolve(), staged)
    except OSError:
        shutil.copy2(audio_path, staged)

    cmd = [
        sys.executable, "inference.py",
        "--model_type", model_type,
        "--config_path", str(config),
        "--start_check_point", str(ckpt),
        "--input_folder", str(stage_in),
        "--store_dir", str(stage_out),
        "--device_ids", device_ids,
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"    MSST {model_type}: {ckpt.name}")
    t0 = time.time()
    try:
        subprocess.run(
            cmd, cwd=str(msst_dir),
            check=True, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or "")[-500:]
        logger.warning(f"    MSST failed ({model_type}); stderr tail: {tail}")
        raise
    finally:
        shutil.rmtree(stage_in, ignore_errors=True)

    produced = sorted(stage_out.rglob("*.wav"))
    logger.info(f"    MSST done in {time.time() - t0:.1f}s -> {len(produced)} stems")
    return produced


def classify_stems(paths: list[Path], names: list[str]) -> dict[str, Path]:
    """Map MSST outputs onto expected stem names by filename substring.

    Longest name first so `backing_vocals` cannot be swallowed by `vocals`, and
    each file is claimed once so a stem never gets two candidates.
    """
    result: dict[str, Path] = {}
    remaining = list(paths)
    for name in sorted(names, key=len, reverse=True):
        needle = name.lower()
        for p in remaining:
            if needle in p.name.lower():
                result[name] = p
                remaining.remove(p)
                break
    return result
