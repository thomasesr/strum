"""
BS-RoFormer SW 6-stem separation via ONNX Runtime.

This is the only route to this model that fits the image. audio-separator
carries it, but only from 0.42 onward, and every version from 0.42 requires
numpy>=2 while TensorFlow 2.15 -- pulled in by basic-pitch for guitar and bass
transcription -- requires numpy<2. MSST is ruled out for the same reason, via
peft. The ONNX export needs nothing but onnxruntime, which is already present.

Model: https://huggingface.co/elicwhite/bs-roformer-sw-6stem-onnx

The contract is fixed rather than dynamic: the model's rotary embeddings cache
by sequence length, so it only accepts exactly 345 frames, which is 176400
samples at 44.1 kHz. Audio is therefore processed in 4-second chunks and
recombined by overlap-add.

Verified against the reference vectors published with the model: STFT to 1.8e-07,
the model call to 1.0e-06, and iSTFT to 5.6e-09.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Fixed by the export; none of these are free parameters.
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
CHUNK_SAMPLES = 176400
CHUNK_FRAMES = 345

# The order the model emits stems in, which is not the order anything else uses.
STEM_ORDER = ("bass", "drums", "other", "vocals", "guitar", "piano")

DEFAULT_OVERLAP = 0.25


def _providers() -> list[str]:
    """Execution providers, preferring the GPU when it is actually available.

    CPU inference runs about 8.6 s per 4-second chunk, which is roughly 17
    minutes for a 6-minute song -- usable as a fallback, not as a default.
    """
    import onnxruntime as ort

    override = os.environ.get("STRUM_ONNX_PROVIDERS", "").strip()
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]

    available = ort.get_available_providers()
    preferred = [p for p in ("CUDAExecutionProvider", "ROCMExecutionProvider")
                 if p in available]
    if not preferred:
        logger.warning(
            "  ONNX Runtime has no GPU provider; BS-RoFormer will run on CPU "
            "and will be slow. Install onnxruntime-gpu for the CUDA provider."
        )
    return preferred + ["CPUExecutionProvider"]


def _stft(chunk: np.ndarray):
    """(2, CHUNK_SAMPLES) float32 -> real, imag each (1, 2, 1025, 345) float32."""
    import torch

    spec = torch.stft(
        torch.from_numpy(chunk),
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT,
        window=torch.hann_window(N_FFT), center=True, return_complex=True,
    )
    return (spec.real.numpy()[None].astype(np.float32),
            spec.imag.numpy()[None].astype(np.float32))


def _istft(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    """(6, 2, 1025, 345) -> (6, 2, CHUNK_SAMPLES) float32."""
    import torch

    spec = torch.complex(torch.from_numpy(real), torch.from_numpy(imag))
    flat = spec.reshape(-1, real.shape[-2], real.shape[-1])
    audio = torch.istft(
        flat, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT,
        window=torch.hann_window(N_FFT), center=True, length=CHUNK_SAMPLES,
    )
    return audio.reshape(len(STEM_ORDER), 2, CHUNK_SAMPLES).numpy()


def _load_audio(path: Path) -> np.ndarray:
    """Read as (2, n) float32 at 44.1 kHz, duplicating mono to stereo."""
    import librosa

    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return np.ascontiguousarray(y[:2], dtype=np.float32)


def separate(
    audio_path: Path,
    output_dir: Path,
    model_path: Path,
    overlap: float = DEFAULT_OVERLAP,
    on_progress=None,
) -> dict[str, Path]:
    """Separate `audio_path` into six stems, written as WAVs in `output_dir`.

    Chunks are cross-faded with a Hann window and divided by the summed window
    at the end, so the overlap-add is unity-gain everywhere including the first
    and last chunk, where fewer windows overlap.
    """
    import onnxruntime as ort
    import soundfile as sf

    audio_path, output_dir, model_path = map(Path, (audio_path, output_dir, model_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    audio = _load_audio(audio_path)
    n_samples = audio.shape[1]

    hop = max(1, int(round(CHUNK_SAMPLES * (1.0 - overlap))))
    # Pad so the first and last samples get the same window coverage as the
    # middle, and so the final chunk is never short of the fixed input length.
    pad = CHUNK_SAMPLES
    padded = np.pad(audio, ((0, 0), (pad, pad + CHUNK_SAMPLES)), mode="constant")
    starts = list(range(0, pad + n_samples + 1, hop))

    session = ort.InferenceSession(str(model_path), providers=_providers())
    logger.info(
        f"  BS-RoFormer ONNX: {len(starts)} chunks, providers={session.get_providers()[:1]}"
    )

    total = padded.shape[1]
    acc = np.zeros((len(STEM_ORDER), 2, total), dtype=np.float32)
    weight = np.zeros(total, dtype=np.float32)
    fade = np.hanning(CHUNK_SAMPLES).astype(np.float32)

    for index, start in enumerate(starts, start=1):
        chunk = padded[:, start:start + CHUNK_SAMPLES]
        if chunk.shape[1] < CHUNK_SAMPLES:      # only possible on the last one
            chunk = np.pad(chunk, ((0, 0), (0, CHUNK_SAMPLES - chunk.shape[1])))

        real, imag = _stft(np.ascontiguousarray(chunk))
        out_real, out_imag = session.run(
            ["out_spec_real", "out_spec_imag"],
            {"spec_real": real, "spec_imag": imag},
        )
        stems = _istft(out_real[0], out_imag[0])

        acc[:, :, start:start + CHUNK_SAMPLES] += stems * fade
        weight[start:start + CHUNK_SAMPLES] += fade
        if on_progress is not None:
            on_progress(index, len(starts))

    # Where no window landed the weight is zero; those samples are silent anyway.
    np.maximum(weight, 1e-8, out=weight)
    acc /= weight

    written: dict[str, Path] = {}
    for i, name in enumerate(STEM_ORDER):
        path = output_dir / f"{name}.wav"
        sf.write(str(path), acc[i, :, pad:pad + n_samples].T, SAMPLE_RATE)
        written[name] = path

    del acc, weight, session
    from src.preprocessing.gpu import free_gpu

    free_gpu("BS-RoFormer ONNX")
    logger.info(f"  BS-RoFormer ONNX wrote: {sorted(written)}")
    return written
