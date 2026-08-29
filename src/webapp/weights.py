"""
Charting weights: check the volume at startup, download whatever is missing.

Weights are deliberately not baked into the image. They are 1.8 GB, licensed
separately, and worth keeping across rebuilds, so they live on a mounted volume
that starts empty. Filling it is therefore a first-run job, not a build step.

This runs inside the app rather than in a container entrypoint for three
reasons: it works the same when `strum-web` is run directly, the server binds
immediately so the UI is reachable while the download proceeds, and progress is
visible in the browser instead of only in container logs.

Jobs wait for this to finish. A job queued during the download sits in the queue
rather than failing against weights that are not there yet.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import threading
from pathlib import Path

from src.preprocessing.karaoke import DEFAULT_KARAOKE_MODEL

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

CHECKPOINT_DIR = Path(os.environ.get("STRUM_CHECKPOINT_DIR") or (REPO_ROOT / "checkpoints"))

# Separation weights are a different set from the charting checkpoints and come
# from different places, so they are fetched separately.
#
# `audio-separator` downloads karaoke weights itself, but defaults to a temp
# directory. Left alone in a container that means ~900 MB re-downloaded on every
# restart, so it is pointed at the models volume instead.
SEPARATION_DIR = Path(os.environ.get("STRUM_SEPARATION_DIR") or "/models")
KARAOKE_MODEL_DIR = Path(
    os.environ.get("STRUM_KARAOKE_MODEL_DIR") or (SEPARATION_DIR / "audio-separator")
)

# MSST-format weights, for the backends audio-separator does not carry.
MVSEP_REPO = "noblebarkrr/mvsepless_resources"
# ONNX export of BS-RoFormer SW. fp16 is half the size of fp32 and matches the
# published reference vectors to ~1e-6, which is well below anything audible.
ROFORMER_ONNX_REPO = "elicwhite/bs-roformer-sw-6stem-onnx"
ROFORMER_ONNX_FILE = os.environ.get(
    "STRUM_ROFORMER_ONNX_FILE", "bs_roformer_sw_6stem_fp16.onnx"
)
MSST_REPO_URL = "https://github.com/ZFTurbo/Music-Source-Separation-Training"
MSST_DIR = Path(os.environ.get("STRUM_MSST_DIR") or (SEPARATION_DIR / "msst"))
ROFORMER_CKPT = "bs_roformer/bs_6stem_fixed.ckpt"
ROFORMER_CFG = "bs_roformer/bs_6stem_fixed_config.yaml"


def _enabled() -> bool:
    raw = os.environ.get("STRUM_FETCH_CHECKPOINTS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off", "")


class Weights:
    """Tracks whether the charting weights are on disk, and fetches them once."""

    def __init__(self, dest: Path = CHECKPOINT_DIR):
        self.dest = Path(dest)
        self.state = "unknown"      # unknown | downloading | ready | failed | disabled
        self.done = 0
        self.total = 0
        self.error = ""
        self._ready = asyncio.Event()
        self._lock = threading.Lock()

    # -- reporting ---------------------------------------------------------

    def public(self) -> dict:
        return {
            "state": self.state,
            "done": self.done,
            "total": self.total,
            "error": self.error,
            "path": str(self.dest),
        }

    @property
    def ready(self) -> bool:
        return self.state in ("ready", "disabled")

    async def wait_ready(self) -> bool:
        """Block until the fetch settles. False means weights are unusable."""
        await self._ready.wait()
        return self.state in ("ready", "disabled")

    def _finish(self, state: str, error: str = "") -> None:
        self.state, self.error = state, error
        self._ready.set()

    # -- the fetch ---------------------------------------------------------

    async def ensure(self) -> None:
        """Fill the checkpoint directory if anything is missing.

        Safe to call once at startup. Never raises: a failure is recorded in
        `state` so the UI can explain it, because a server that refuses to start
        is far harder to diagnose than one that says what is wrong.
        """
        try:
            from fetch_checkpoints import fetch_plan, missing, resolve_plan
        except ImportError as e:
            self._finish("failed", f"weight downloader unavailable: {e}")
            return

        plan = resolve_plan(self.dest)
        self.total = len(plan)
        absent = missing(plan)

        if not absent:
            logger.info(f"Weights present in {self.dest} ({self.total} files)")
            self.done = self.total
            self._finish("ready")
            return

        if not _enabled():
            self._finish(
                "disabled",
                f"{len(absent)} weight file(s) missing and STRUM_FETCH_CHECKPOINTS is off",
            )
            logger.warning(self.error)
            return

        self.state = "downloading"
        self.done = self.total - len(absent)
        logger.info(
            f"Fetching {len(absent)} of {self.total} weight file(s) into {self.dest}. "
            "The UI is usable now; charting waits for this to finish."
        )

        def progress(group: str, done: int, total: int) -> None:
            with self._lock:
                self.done = done
            logger.info(f"  weights [{done}/{total}] {group}")

        try:
            fetched, skipped, failed = await asyncio.to_thread(
                fetch_plan, plan, on_progress=progress
            )
        except Exception as e:
            logger.exception("Weight download failed")
            self._finish("failed", str(e))
            return

        self.done = self.total
        if failed:
            self._finish("failed", f"could not download: {', '.join(failed)}")
            logger.error(self.error)
            return

        logger.info(f"Weights ready: {fetched} downloaded, {skipped} already present")
        self._finish("ready")


weights = Weights()


def _hf_download(repo: str, filename: str, dest: Path) -> Path:
    """Fetch one Hub file to `dest`, copying it out of the shared HF cache."""
    from huggingface_hub import hf_hub_download

    if dest.exists():
        return dest
    cached = hf_hub_download(repo_id=repo, filename=filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, dest)
    return dest


def ensure_karaoke_model() -> None:
    """Pre-download the Mel-Band RoFormer karaoke weights.

    audio-separator fetches these on first use anyway; doing it now means the
    first job is not silently stalled behind ~900 MB, and pins them to the
    models volume so a restart does not re-download them.
    """
    model = os.environ.get("STRUM_KARAOKE_MODEL") or DEFAULT_KARAOKE_MODEL
    KARAOKE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if (KARAOKE_MODEL_DIR / model).exists():
        logger.info(f"Karaoke model present: {model}")
        return

    from audio_separator.separator import Separator

    logger.info(f"Fetching karaoke model {model} into {KARAOKE_MODEL_DIR}")
    sep = Separator(
        model_file_dir=str(KARAOKE_MODEL_DIR),
        output_dir=str(KARAOKE_MODEL_DIR / "_scratch"),
        log_level=logging.WARNING,
    )
    sep.download_model_files(model) if hasattr(sep, "download_model_files") \
        else sep.load_model(model_filename=model)
    logger.info("Karaoke model ready")


def ensure_roformer_model() -> None:
    """Pre-download BS-RoFormer SW, when that backend is selected.

    Fetches the ONNX export rather than the original checkpoint. Every
    audio-separator release that carries this model requires numpy>=2, and
    TensorFlow 2.15 -- which basic-pitch needs for guitar and bass -- requires
    numpy<2, so the two cannot coexist. MSST is out for the same reason via
    peft. ONNX Runtime is already installed and has no such constraint.

    Only runs when the backend is selected: the file is ~350 MB and Demucs is
    the default.
    """
    if os.environ.get("STRUM_SEPARATOR", "demucs").strip().lower() not in (
        "bs_roformer_sw", "bs_roformer", "roformer"
    ):
        logger.info("BS-RoFormer SW not selected; skipping its download")
        separation.steps["roformer"] = "skipped (not selected)"
        return

    from src.preprocessing.roformer import RoformerConfig

    cfg = RoformerConfig.from_env()
    if cfg.ckpt and cfg.cfg:
        # An explicit checkpoint means MSST, which needs its checkout present.
        logger.info(f"BS-RoFormer SW using explicit checkpoint {cfg.ckpt}")
        _ensure_msst_checkout()
        return

    dest = SEPARATION_DIR / "roformer" / ROFORMER_ONNX_FILE
    _hf_download(ROFORMER_ONNX_REPO, ROFORMER_ONNX_FILE, dest)
    # The pipeline runs as a child of this process and inherits the environment.
    os.environ.setdefault("STRUM_ROFORMER_ONNX", str(dest))
    logger.info(f"BS-RoFormer SW ONNX ready: {dest}")


def _ensure_msst_checkout() -> None:
    """Clone the MSST inference repo, which the RoFormer backends run through.

    Without it the backend downloads its weights and then silently falls back to
    Demucs, which is a confusing way to spend 700 MB. MSST is a CLI rather than
    a package, so this is a checkout, not a pip install; its dependencies are
    already satisfied by audio-separator's.
    """
    if (MSST_DIR / "inference.py").exists():
        return
    import subprocess

    MSST_DIR.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cloning MSST into {MSST_DIR}")
    subprocess.run(
        ["git", "clone", "--depth", "1", MSST_REPO_URL, str(MSST_DIR)],
        check=True, capture_output=True, text=True, timeout=600,
    )


def ensure_demucs_model() -> None:
    """Warm the Demucs weights so the first job does not stall on them."""
    from demucs.pretrained import get_model

    for name in {
        os.environ.get("STRUM_DEMUCS_MODEL") or "htdemucs_6s",
        os.environ.get("STRUM_DRUMS_DEMUCS") or "htdemucs_ft",
    }:
        logger.info(f"Warming Demucs model {name}")
        get_model(name)


class SeparationWeights:
    """Tracks the separation models, which come from sources of their own.

    Kept apart from the charting checkpoints because a failure here is less
    severe: every step is a warm-up of something the pipeline would otherwise
    download on demand, so a failure costs a slow first job rather than a
    broken one. Charting is never blocked on this.
    """

    STEPS = (
        ("karaoke", ensure_karaoke_model),
        ("roformer", ensure_roformer_model),
        ("demucs", ensure_demucs_model),
    )

    def __init__(self) -> None:
        self.state = "unknown"     # unknown | downloading | ready | partial | disabled
        self.current = ""
        self.steps: dict[str, str] = {}

    def public(self) -> dict:
        return {"state": self.state, "current": self.current, "steps": dict(self.steps)}

    async def ensure(self) -> None:
        if not _enabled():
            self.state = "disabled"
            return

        self.state = "downloading"
        for name, fn in self.STEPS:
            self.current = name
            try:
                await asyncio.to_thread(fn)
                self.steps[name] = self.steps.get(name) or "ready"
            except Exception as e:
                # Warm-up only: the pipeline still downloads on demand.
                logger.warning(f"Separation weights ({name}) not pre-fetched: {e}")
                self.steps[name] = f"failed: {e}"
        self.current = ""
        self.state = "ready" if all(
            v == "ready" for v in self.steps.values()
        ) else "partial"


separation = SeparationWeights()
