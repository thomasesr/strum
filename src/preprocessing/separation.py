"""
Demucs wrapper for audio source separation.

Separates audio into stems: drums, bass, vocals, other.
"""

from pathlib import Path
from typing import Optional
import logging

import torch
import torchaudio

logger = logging.getLogger(__name__)


def separate_stems(
    audio_path: Path,
    output_dir: Path,
    model_name: str = "htdemucs_6s",  # 6-stem model: drums, bass, vocals, guitar, piano, other
    device: Optional[str] = None,
    shifts: int = 1,
    overlap: float = 0.25,
    backend: Optional[str] = None,
    karaoke: Optional[bool] = None,
) -> dict[str, Path]:
    """
    Separate audio file into stems using Demucs.
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save separated stems
        model_name: Demucs model to use:
            - htdemucs_6s: 6-stem (drums, bass, vocals, guitar, piano, other) - RECOMMENDED
            - htdemucs: 4-stem (drums, bass, vocals, other)
            - htdemucs_ft: fine-tuned 4-stem
        device: Device to run on (cuda, cpu, or None for auto)
        shifts: Number of random shifts for better quality
        overlap: Overlap between segments
        backend: "demucs" (default) or "bs_roformer_sw". BS-RoFormer SW emits the
            same six stem names but runs through an MSST checkout; it falls back
            to Demucs when its weights are not configured.
        karaoke: Run Mel-Band RoFormer karaoke pre-separation first, so the
            6-stem model never sees the lead vocal. None reads STRUM_KARAOKE.
        
    Returns:
        Dictionary mapping stem names to output paths. Includes `lead_vocals`
        (and `backing_vocals`, when the backing split ran) if karaoke was used.
    """
    from src.preprocessing.karaoke import KaraokeConfig, separate_karaoke
    from demucs.pretrained import get_model
    from demucs.separate import load_track
    from demucs.apply import apply_model
    
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Stage 0: strip the lead vocal before the 6-stem model sees the mix.
    kcfg = KaraokeConfig.from_env()
    if karaoke is not None:
        kcfg.enabled = karaoke
    karaoke_stems = separate_karaoke(audio_path, output_dir / "karaoke", kcfg)
    separation_input = karaoke_stems.get("instrumental", audio_path)
    # `instrumental` is an intermediate, not a chartable stem — keep it out of
    # the returned mapping so downstream stage loops never pick it up.
    karaoke_stems = {k: v for k, v in karaoke_stems.items() if k != "instrumental"}

    # Stage 1: BS-RoFormer SW if selected and available, else Demucs.
    import os

    backend = (backend or os.environ.get("STRUM_SEPARATOR", "demucs")).strip().lower()
    if backend in ("bs_roformer_sw", "bs_roformer", "roformer"):
        from src.preprocessing.roformer import separate_6stem_roformer

        stems = separate_6stem_roformer(separation_input, output_dir)
        if stems:
            stems.update(karaoke_stems)
            if "lead_vocals" in karaoke_stems:
                stems["vocals"] = karaoke_stems["lead_vocals"]
            return stems
        logger.info("Falling back to Demucs")

    logger.info(f"Separating {separation_input} with {model_name} on {device}")
    
    # Load model
    model = get_model(model_name)
    model.to(device)
    model.eval()
    
    # Load audio
    wav = load_track(separation_input, model.audio_channels, model.samplerate)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    
    # Apply model
    with torch.no_grad():
        sources = apply_model(
            model, 
            wav[None].to(device), 
            shifts=shifts,
            overlap=overlap,
        )[0]
    
    # Denormalize
    sources = sources * ref.std() + ref.mean()
    
    # Save stems
    stem_paths = {}
    source_names = model.sources  # e.g., ['drums', 'bass', 'other', 'vocals']
    
    for i, stem_name in enumerate(source_names):
        stem_audio = sources[i].cpu()
        stem_path = output_dir / f"{stem_name}.wav"
        
        torchaudio.save(
            str(stem_path),
            stem_audio,
            model.samplerate,
        )
        
        stem_paths[stem_name] = stem_path
        logger.debug(f"Saved {stem_name} to {stem_path}")
    
    # The karaoke isolate is a better vocal stem than the Demucs branch, which
    # here only saw an instrumental anyway.
    stem_paths.update(karaoke_stems)
    if "lead_vocals" in karaoke_stems:
        stem_paths["vocals"] = karaoke_stems["lead_vocals"]

    logger.info(f"Separation complete. Stems: {list(stem_paths.keys())}")
    
    # Free GPU memory from Demucs model
    del model, sources, wav
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return stem_paths


def load_audio(
    audio_path: Path,
    sample_rate: int = 44100,
    mono: bool = False,
) -> tuple[torch.Tensor, int]:
    """
    Load audio file and resample if needed.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        mono: Whether to convert to mono
        
    Returns:
        Tuple of (audio tensor, sample rate)
    """
    audio, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(audio)
        sr = sample_rate
    
    # Convert to mono if needed
    if mono and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    return audio, sr


def normalize_loudness(
    audio: torch.Tensor,
    sample_rate: int,
    target_lufs: float = -14.0,
) -> torch.Tensor:
    """
    Normalize audio to target loudness (LUFS).
    
    Args:
        audio: Audio tensor (channels, samples)
        sample_rate: Sample rate
        target_lufs: Target loudness in LUFS
        
    Returns:
        Normalized audio tensor
    """
    # Simple peak normalization for now
    # TODO: Implement proper LUFS measurement
    peak = audio.abs().max()
    if peak > 0:
        # Normalize to -1 dB headroom, then scale to approximate target LUFS
        audio = audio / peak * 0.9
    
    return audio
