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
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
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

# Clone Hero loads smaller art noticeably faster, and 512 is the size the
# reference .sng encoder defaults to.
ALBUM_ART_SIZE = 512
MAX_COVER_MB = 12

# song.ini fields a user can set. `year` is `date` in most tag schemes.
TAG_FIELDS = ("title", "artist", "album", "genre")


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

    # Container tags live on the format for most codecs, but Ogg/FLAC put them
    # on the stream instead, so merge both with the format winning.
    tags = {k.lower(): v for k, v in (audio[0].get("tags") or {}).items()}
    tags.update({k.lower(): v for k, v in (fmt.get("tags") or {}).items()})

    metadata = {f: str(tags.get(f, "")).strip() for f in TAG_FIELDS}
    # Year is spelled differently per container; take the first four digits of
    # whichever one is present, since values like "2004-11-16" are common.
    raw_year = str(tags.get("date") or tags.get("year") or
                   tags.get("originalyear") or "").strip()
    match = re.search(r"\d{4}", raw_year)
    metadata["year"] = match.group(0) if match else ""

    return {
        "duration_s": duration,
        # Cover art rides along as a video stream, so a real video needs frames.
        "has_video": any(
            s.get("codec_type") == "video" and s.get("avg_frame_rate") not in ("0/0", None)
            for s in streams
        ),
        "has_cover": cover_stream_index(info) is not None,
        "format": fmt.get("format_name", ""),
        "metadata": metadata,
        # Kept for callers that only want these two.
        "title": metadata["title"],
        "artist": metadata["artist"],
    }


def cover_stream_index(info: dict) -> int | None:
    """Index of the embedded cover art stream, if the file carries one.

    Embedded art is a video stream flagged `attached_pic`; that flag is what
    separates a cover from an actual video track.
    """
    for stream in info.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        if (stream.get("disposition") or {}).get("attached_pic") == 1:
            return stream.get("index")
    return None


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


def _run_ffmpeg(cmd: list[str], what: str) -> bool:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"{what}: timed out")
        return False
    if result.returncode != 0:
        logger.warning(f"{what}: ffmpeg failed: {result.stderr[-300:]}")
        return False
    return True


def extract_cover(src: Path, dest: Path) -> Path | None:
    """Pull embedded cover art out of `src` into `dest` as PNG.

    Returns None when the file has no attached picture, or when the picture
    cannot be decoded -- neither is an error, it just means the user has to
    supply art themselves.
    """
    src, dest = Path(src), Path(dest)
    try:
        index = cover_stream_index(probe(src))
    except MediaError:
        return None
    if index is None:
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = _run_ffmpeg(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-map", f"0:{index}",
            "-frames:v", "1",
            "-vf", _art_filter(),
            str(dest),
        ],
        "cover extraction",
    )
    return dest if ok and dest.exists() else None


def _art_filter() -> str:
    """Fit art inside a square without stretching, padding to the exact size."""
    n = ALBUM_ART_SIZE
    return (
        f"scale={n}:{n}:force_original_aspect_ratio=decrease,"
        f"pad={n}:{n}:(ow-iw)/2:(oh-ih)/2:color=black"
    )


def normalize_cover(src: Path, dest: Path) -> Path:
    """Convert user-supplied art to a square PNG the games will load.

    Raises MediaError if the upload is not a decodable image.
    """
    src, dest = Path(src), Path(dest)
    if src.suffix.lower() not in IMAGE_EXTS:
        allowed = ", ".join(sorted(e.lstrip(".") for e in IMAGE_EXTS))
        raise MediaError(f"Album art must be one of: {allowed}")

    info = probe(src)
    if not any(s.get("codec_type") == "video" for s in info.get("streams", [])):
        raise MediaError("That file is not a readable image")

    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = _run_ffmpeg(
        ["ffmpeg", "-y", "-i", str(src), "-frames:v", "1",
         "-vf", _art_filter(), str(dest)],
        "cover conversion",
    )
    if not ok or not dest.exists():
        raise MediaError("Album art could not be converted")
    return dest
