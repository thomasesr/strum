"""
Turn a finished song folder into downloadable packages.

Clone Hero and YARG both read an extracted song folder, so the .zip serves
both. YARG additionally reads the single-file `.sng` container, which is nicer
to move around and is what YARG users tend to expect.

Intermediate separation output (`stems/`, `demucs_temp/`) lives inside the song
folder while the pipeline runs, so both writers exclude it -- otherwise every
download would carry a few hundred MB of working WAVs.
"""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

from src.packaging.sng import write_sng

logger = logging.getLogger(__name__)

# Working directories the pipeline leaves inside the song folder.
EXCLUDE_DIRS = {"stems", "work", "demucs_temp", "_karaoke_tmp", "_roformer_tmp"}

# Windows forbids these outright, and Clone Hero song folders end up on Windows
# far more often than not.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_name(name: str, fallback: str = "song") -> str:
    """Make `name` usable as a filename on Windows, macOS and Linux."""
    cleaned = _UNSAFE.sub("_", name).strip(" .")
    return cleaned[:120] or fallback


def _packable(folder: Path) -> list[Path]:
    return [
        p for p in sorted(folder.rglob("*"))
        if p.is_file()
        and not any(part in EXCLUDE_DIRS for part in p.relative_to(folder).parts[:-1])
    ]


def write_zip(song_folder: Path, output_path: Path, root_name: str | None = None) -> Path:
    """Zip a song folder, keeping the folder itself as the archive's top level.

    Clone Hero expects to find `song.ini` one level down from the extraction
    point, so the archive must not be flat. The stored root is sanitised because
    these archives are usually extracted on Windows, which rejects `:` and `?`
    in paths outright.
    """
    song_folder, output_path = Path(song_folder), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = safe_name(root_name or song_folder.name)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in _packable(song_folder):
            zf.write(path, f"{root}/{path.relative_to(song_folder).as_posix()}")

    logger.info(f"Wrote {output_path.name}: {output_path.stat().st_size / 1e6:.1f} MB")
    return output_path


def package_song(
    song_folder: Path,
    output_dir: Path,
    formats: tuple[str, ...] = ("zip", "sng"),
    base_name: str | None = None,
) -> dict[str, Path]:
    """Write every requested package format for one song folder.

    Args:
        song_folder: Finished song folder from the pipeline.
        output_dir: Where to write packages.
        formats: Any of "zip" (Clone Hero + YARG) and "sng" (YARG single-file).
        base_name: Filename stem. Defaults to the song folder's name.

    Returns:
        Mapping of format -> written path. A format that fails is logged and
        omitted rather than raised, so one bad writer cannot lose the other's
        output.
    """
    song_folder, output_dir = Path(song_folder), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(base_name or song_folder.name)

    written: dict[str, Path] = {}
    for fmt in formats:
        try:
            if fmt == "zip":
                written["zip"] = write_zip(song_folder, output_dir / f"{stem}.zip", stem)
            elif fmt == "sng":
                written["sng"] = write_sng(
                    song_folder, output_dir / f"{stem}.sng", exclude_dirs=EXCLUDE_DIRS
                )
            else:
                logger.warning(f"Unknown package format: {fmt}")
        except Exception as e:
            logger.error(f"Packaging {fmt} failed: {e}")

    if not written:
        raise RuntimeError(f"No packages could be written for {song_folder}")
    return written
