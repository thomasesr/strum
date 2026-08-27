"""
Apply user-supplied song metadata to a finished song folder.

The pipeline derives artist and title from the input filename and looks album
art up online. Both are guesses, and both are what the game shows in its song
list, so the web UI lets the user correct them.

Corrections are applied here, after charting, rather than by feeding values
into the pipeline: its `parse_filename` runs a MusicBrainz check that can swap
artist and title around, so a filename is not a reliable channel. Rewriting the
finished song.ini is deterministic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.packaging.package import safe_name

logger = logging.getLogger(__name__)

# song.ini keys this module owns. Everything else in the file -- the diff_*
# ratings the chart enhancer computes, song_length, preview_start_time -- is
# left exactly as the pipeline wrote it.
INI_KEYS = {
    "title": "name",
    "artist": "artist",
    "album": "album",
    "year": "year",
    "genre": "genre",
    "charter": "charter",
}


@dataclass
class SongMeta:
    """Metadata the user can override in the web UI."""

    title: str = ""
    artist: str = ""
    album: str = ""
    year: str = ""
    genre: str = ""
    charter: str = ""

    def filled(self) -> dict[str, str]:
        """Only the fields the user actually set. Blanks must not erase
        whatever the pipeline managed to work out on its own."""
        return {k: v.strip() for k, v in vars(self).items() if v and v.strip()}

    @property
    def folder_name(self) -> str:
        """`Artist - Title`, or empty if either half is missing."""
        artist, title = self.artist.strip(), self.title.strip()
        if not (artist and title):
            return ""
        return safe_name(f"{artist} - {title}", fallback="")


def _sanitize(value: str) -> str:
    """Strip what would break an INI line: comments, newlines, nulls."""
    return value.replace(";", "").replace("\x00", "").replace("\r", " ").replace("\n", " ").strip()


def update_song_ini(ini_path: Path, values: dict[str, str]) -> None:
    """Rewrite the given `[song]` keys in place, preserving everything else.

    Done line by line rather than with configparser so that key order, spacing
    and any keys we do not know about survive untouched.
    """
    ini_path = Path(ini_path)
    wanted = {INI_KEYS[k]: _sanitize(v) for k, v in values.items() if k in INI_KEYS}
    if not wanted:
        return

    lines = ini_path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    out: list[str] = []
    last_kv = -1

    for line in lines:
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith((";", "#", "[")):
            key = stripped.split("=", 1)[0].strip()
            if key.lower() in wanted:
                out.append(f"{key} = {wanted[key.lower()]}")
                seen.add(key.lower())
                last_kv = len(out) - 1
                continue
            last_kv = len(out)
        out.append(line)

    # Keys the pipeline never wrote get appended next to the others rather than
    # after any trailing blank lines.
    missing = [f"{k} = {v}" for k, v in wanted.items() if k not in seen]
    if missing:
        at = last_kv + 1 if last_kv >= 0 else len(out)
        out[at:at] = missing

    ini_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    logger.info(f"song.ini updated: {', '.join(sorted(wanted))}")


def apply_metadata(
    song_folder: Path,
    meta: SongMeta,
    cover: Path | None = None,
) -> Path:
    """Apply overrides to a finished song folder, returning its (possibly new) path.

    Args:
        song_folder: Folder the pipeline produced.
        meta: User overrides. Blank fields are ignored.
        cover: Album art to install, already normalised to PNG. When None, the
            art the pipeline found is kept.

    The folder is renamed to `Artist - Title` when both are given, since that
    name is what ends up as the package filename and the extracted directory.
    """
    song_folder = Path(song_folder)

    values = meta.filled()
    if values:
        ini = song_folder / "song.ini"
        if ini.exists():
            update_song_ini(ini, values)
        else:
            logger.warning(f"No song.ini in {song_folder}; metadata not applied")

    if cover is not None and Path(cover).exists():
        # The pipeline may have written album.jpg instead; drop it so the game
        # cannot pick the stale one.
        for stale in song_folder.glob("album.*"):
            stale.unlink()
        target = song_folder / "album.png"
        target.write_bytes(Path(cover).read_bytes())
        logger.info("Album art replaced with the uploaded image")

    desired = meta.folder_name
    if desired and desired != song_folder.name:
        renamed = song_folder.parent / desired
        if renamed.exists():
            logger.warning(f"{renamed} already exists; keeping {song_folder.name}")
            return song_folder
        song_folder.rename(renamed)
        logger.info(f"Song folder renamed to {desired}")
        return renamed

    return song_folder
