"""
Input validation and audio extraction for uploads.

Uploads are untrusted: the filename and the browser-supplied content type are
both attacker-controlled, so neither decides anything here. The extension only
picks a container hint; ffprobe is what actually decides whether a file is
audio we can chart, and video files get their audio stream demuxed out before
the pipeline ever sees them.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
VIDEO_EXTS = {".mp4", ".mkv"}
ALLOWED_EXTS = AUDIO_EXTS | VIDEO_EXTS

# Extracting to FLAC keeps the pipeline's input lossless without re-encoding
# cost mattering; separation quality is sensitive to lossy artefacts.
EXTRACTED_SUFFIX = ".flac"

FFPROBE_TIMEOUT_S = 60
FFMPEG_TIMEOUT_S = 900

# Guards against a single upload occupying the GPU for an hour.
MIN_DURATION_S = 5.0
MAX_DURATION_S = 30 * 60


class MediaError(ValueError):
    """Raised when an upload is not usable audio. Message is user-facing."""


def probe(path: Path) -> dict:
    """Run ffprobe and return the parsed JSON, or raise MediaError."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT_S
        )
    except FileNotFoundError as e:
        raise MediaError("ffprobe is not installed on the server") from e
    except subprocess.TimeoutExpired as e:
        raise MediaError("Timed out inspecting the file") from e

    if result.returncode != 0:
        raise MediaError("File could not be read as audio or video")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise MediaError("File could not be read as audio or video") from e


def inspect(path: Path) -> dict:
    """Validate an upload and return {duration_s, has_video, format, title, artist}.

    Raises MediaError with a message safe to show the user.
    """
    path = Path(path)
    if path.suffix.lower() not in ALLOWED_EXTS:
        allowed = ", ".join(sorted(e.lstrip(".") for e in ALLOWED_EXTS))
        raise MediaError(f"Unsupported file type. Accepted: {allowed}")

    info = probe(path)
    streams = info.get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        raise MediaError("No audio stream found in this file")

    fmt = info.get("format", {})
    try:
        duration = float(fmt.get("duration") or audio[0].get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration and not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
        raise MediaError(
            f"Audio is {duration / 60:.1f} min; must be between "
            f"{MIN_DURATION_S:.0f}s and {MAX_DURATION_S / 60:.0f} min"
        )

    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    return {
        "duration_s": duration,
        # Cover art rides along as a video stream, so a real video needs frames.
        "has_video": any(
            s.get("codec_type") == "video" and s.get("avg_frame_rate") not in ("0/0", None)
            for s in streams
        ),
        "format": fmt.get("format_name", ""),
        "title": tags.get("title", ""),
        "artist": tags.get("artist", ""),
    }


def extract_audio(src: Path, dest_dir: Path) -> Path:
    """Return a path to pipeline-ready audio, demuxing video if needed.

    Audio uploads pass straight through -- the pipeline reads all of them, and
    re-encoding would only add loss.
    """
    src, dest_dir = Path(src), Path(dest_dir)
    if src.suffix.lower() in AUDIO_EXTS:
        return src

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (src.stem + EXTRACTED_SUFFIX)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-sn", "-dn",   # drop video, subtitles and data streams
        "-map", "0:a:0",       # first audio stream only
        "-c:a", "flac",
        str(dest),
    ]
    logger.info(f"Extracting audio from {src.name}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S
        )
    except subprocess.TimeoutExpired as e:
        raise MediaError("Timed out extracting audio from the video") from e
    if result.returncode != 0 or not dest.exists():
        logger.error(f"ffmpeg extract failed: {result.stderr[-500:]}")
        raise MediaError("Could not extract an audio track from this video")
    return dest
