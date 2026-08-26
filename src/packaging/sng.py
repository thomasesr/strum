"""
Writer for the `.sng` container — YARG's single-file song package.

Spec: https://github.com/mdsitton/SngFileFormat

Layout, all integers little-endian:

    "SNGPKG"                 6 bytes
    version                  uint32   (1)
    xorMask                  16 bytes

    metadataLen              uint64   bytes after this field
    metadataCount            uint64
      keyLen int32, key, valueLen int32, value   x metadataCount

    fileMetaLen              uint64   bytes after this field
    fileCount                uint64
      filenameLen byte, filename, contentsLen uint64, contentsIndex uint64

    fileDataLen              uint64
      masked file bytes, concatenated in fileMeta order

`contentsIndex` is an absolute offset from the start of the file, so the whole
index has to be sized before any of it is written.

song.ini does not survive into the container as a file: its `[Song]` keys become
the metadata section, and the game rebuilds the ini from them.
"""

from __future__ import annotations

import configparser
import logging
import os
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

MAGIC = b"SNGPKG"
VERSION = 1
XOR_MASK_LEN = 16

# Filenames are length-prefixed with a single byte.
MAX_FILENAME_LEN = 255

# Characters the spec bans in metadata, because the game round-trips these
# pairs through an INI file.
_BAD_IN_KEY = ";=\r\n"
_BAD_IN_VALUE = ";\r\n"


def mask(data: bytes, xor_mask: bytes) -> bytes:
    """Apply the .sng masking transform.

    Each byte is XORed with `xorMask[i % 16] ^ (i & 0xFF)`. The transform is its
    own inverse, so this both masks and unmasks.
    """
    out = bytearray(len(data))
    for i, b in enumerate(data):
        out[i] = b ^ xor_mask[i % XOR_MASK_LEN] ^ (i & 0xFF)
    return bytes(out)


def _clean(text: str, banned: str) -> str:
    for ch in banned:
        text = text.replace(ch, "")
    return text.replace("\x00", "").strip()


def read_song_ini(ini_path: Path) -> dict[str, str]:
    """Read a song.ini `[Song]` section into a metadata mapping.

    Keys and values are stripped of the characters the spec disallows rather
    than rejected, since the values come from tag lookups we do not control.
    """
    parser = configparser.ConfigParser(strict=False)
    # Preserve key case: the game's key registry is lowercase, but a value like
    # `<color=#00FF00>` in a name must survive untouched.
    parser.optionxform = str
    parser.read(ini_path, encoding="utf-8")

    section = next((s for s in parser.sections() if s.lower() == "song"), None)
    if section is None:
        return {}

    metadata: dict[str, str] = {}
    for key, value in parser.items(section):
        k, v = _clean(key, _BAD_IN_KEY), _clean(value, _BAD_IN_VALUE)
        if k and v:
            metadata[k] = v
    return metadata


def _collect_files(folder: Path, exclude_dirs: set[str]) -> list[tuple[str, Path]]:
    """List packable files as (relative name, path), sorted for reproducibility.

    song.ini is dropped -- it becomes the metadata section instead. Directories
    in `exclude_dirs` are skipped wholesale, which is how intermediate `stems/`
    working data stays out of the package.
    """
    files: list[tuple[str, Path]] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(folder)
        if any(part in exclude_dirs for part in rel.parts[:-1]):
            continue
        if rel.name.lower() == "song.ini":
            continue
        name = rel.as_posix()
        if len(name.encode("utf-8")) > MAX_FILENAME_LEN:
            logger.warning(f"Skipping {name}: filename exceeds {MAX_FILENAME_LEN} bytes")
            continue
        files.append((name, path))
    return files


def write_sng(
    folder: Path,
    output_path: Path,
    metadata: dict[str, str] | None = None,
    exclude_dirs: set[str] | None = None,
    xor_mask: bytes | None = None,
) -> Path:
    """Pack a song folder into a `.sng` container.

    Args:
        folder: Song folder (notes.mid, the .ogg tracks, album art, song.ini).
        output_path: Destination .sng path.
        metadata: Metadata pairs. Defaults to the `[Song]` section of song.ini.
        exclude_dirs: Directory names to skip anywhere in the tree.
        xor_mask: 16-byte mask. Defaults to fresh random bytes, as the reference
            encoder does; pass a fixed mask only to make output byte-reproducible.

    Returns:
        `output_path`.
    """
    folder, output_path = Path(folder), Path(output_path)
    exclude_dirs = exclude_dirs or {"stems", "work", "demucs_temp"}

    if metadata is None:
        ini = folder / "song.ini"
        metadata = read_song_ini(ini) if ini.exists() else {}
    metadata = {k: v for k, v in metadata.items() if k and v}

    files = _collect_files(folder, exclude_dirs)
    if not files:
        raise ValueError(f"No packable files in {folder}")

    if xor_mask is None:
        xor_mask = os.urandom(XOR_MASK_LEN)
    if len(xor_mask) != XOR_MASK_LEN:
        raise ValueError(f"xor_mask must be {XOR_MASK_LEN} bytes, got {len(xor_mask)}")

    # --- metadata section ---
    meta_body = bytearray()
    for key, value in metadata.items():
        kb, vb = key.encode("utf-8"), value.encode("utf-8")
        meta_body += struct.pack("<i", len(kb)) + kb
        meta_body += struct.pack("<i", len(vb)) + vb
    # The declared length covers the count field plus the pairs.
    metadata_len = 8 + len(meta_body)

    # --- file index section ---
    sizes = [p.stat().st_size for _, p in files]
    index_len = 8 + sum(1 + len(name.encode("utf-8")) + 16 for name, _ in files)

    # contentsIndex is absolute, so file data starts after every fixed-size part:
    # header + both length prefixes + both section bodies + the fileDataLen field.
    header_len = len(MAGIC) + 4 + XOR_MASK_LEN
    data_start = header_len + 8 + metadata_len + 8 + index_len + 8

    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", VERSION)
    out += xor_mask

    out += struct.pack("<Q", metadata_len)
    out += struct.pack("<Q", len(metadata))
    out += meta_body

    out += struct.pack("<Q", index_len)
    out += struct.pack("<Q", len(files))
    offset = data_start
    for (name, _), size in zip(files, sizes):
        nb = name.encode("utf-8")
        out += struct.pack("<B", len(nb)) + nb
        out += struct.pack("<Q", size)
        out += struct.pack("<Q", offset)
        offset += size

    out += struct.pack("<Q", sum(sizes))
    assert len(out) == data_start, f"index sizing off: {len(out)} != {data_start}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(out)
        for _, path in files:
            fh.write(mask(path.read_bytes(), xor_mask))

    logger.info(
        f"Wrote {output_path.name}: {len(files)} files, "
        f"{len(metadata)} metadata keys, {output_path.stat().st_size / 1e6:.1f} MB"
    )
    return output_path


def read_sng(path: Path) -> tuple[dict[str, str], dict[str, bytes]]:
    """Unpack a `.sng` back into (metadata, {filename: contents}).

    Exists so the writer can be verified by round-trip; the pipeline itself
    never needs to read these back.
    """
    raw = Path(path).read_bytes()
    if raw[:6] != MAGIC:
        raise ValueError(f"Not a .sng file: {raw[:6]!r}")
    version = struct.unpack_from("<I", raw, 6)[0]
    if version != VERSION:
        raise ValueError(f"Unsupported .sng version {version}")
    xor_mask = raw[10:26]

    pos = 26
    (metadata_len,) = struct.unpack_from("<Q", raw, pos); pos += 8
    meta_end = pos + metadata_len
    (count,) = struct.unpack_from("<Q", raw, pos); pos += 8
    metadata: dict[str, str] = {}
    for _ in range(count):
        (klen,) = struct.unpack_from("<i", raw, pos); pos += 4
        key = raw[pos:pos + klen].decode("utf-8"); pos += klen
        (vlen,) = struct.unpack_from("<i", raw, pos); pos += 4
        metadata[key] = raw[pos:pos + vlen].decode("utf-8"); pos += vlen
    pos = meta_end

    (index_len,) = struct.unpack_from("<Q", raw, pos); pos += 8
    index_end = pos + index_len
    (file_count,) = struct.unpack_from("<Q", raw, pos); pos += 8
    files: dict[str, bytes] = {}
    entries = []
    for _ in range(file_count):
        nlen = raw[pos]; pos += 1
        name = raw[pos:pos + nlen].decode("utf-8"); pos += nlen
        size, index = struct.unpack_from("<QQ", raw, pos); pos += 16
        entries.append((name, size, index))
    pos = index_end

    for name, size, index in entries:
        files[name] = mask(raw[index:index + size], xor_mask)
    return metadata, files
