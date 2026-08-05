"""Takeout input handling: auto-detect archives vs directories, extract, discover."""

from __future__ import annotations

import tarfile
from pathlib import Path

from .media import is_media_file

EXTRACTED_DIRNAME = ".pixel-purge-extracted"


def _is_within(base: Path, target: Path) -> bool:
    """Guard against path traversal (tar slip) during extraction."""
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def extract_tgz(archive: Path, extract_dir: Path) -> None:
    """Safely extract a .tgz archive, skipping any member that escapes extract_dir."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        safe = [m for m in tar.getmembers() if _is_within(extract_dir, extract_dir / m.name)]
        tar.extractall(extract_dir, members=safe)


def resolve_media_root(source_path: Path) -> Path:
    """Given a CLI input path, return the directory to scan for media.

    Handles three cases (auto-detected):
      - a single .tgz file            -> extract it, scan the extraction dir
      - a directory containing .tgz   -> extract all, scan the extraction dir
      - a plain (already-extracted)   -> scan it directly
    """
    if source_path.is_file() and source_path.suffix.lower() == ".tgz":
        extract_dir = source_path.parent / EXTRACTED_DIRNAME
        extract_tgz(source_path, extract_dir)
        return extract_dir

    if source_path.is_dir():
        archives = sorted(source_path.glob("*.tgz"))
        if archives:
            extract_dir = source_path / EXTRACTED_DIRNAME
            for archive in archives:
                extract_tgz(archive, extract_dir)
            return extract_dir
        return source_path

    raise FileNotFoundError(f"Input path not found or unsupported: {source_path}")


def discover_media_files(media_root: Path) -> list[Path]:
    """Recursively find all media files under media_root, sorted for determinism."""
    return sorted(f for f in media_root.rglob("*") if f.is_file() and is_media_file(f))
