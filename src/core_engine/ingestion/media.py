"""Media type classification and the extension universe."""

from __future__ import annotations

from pathlib import Path

PHOTO_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".heic", ".heif",
    ".webp", ".bmp", ".tiff", ".tif", ".raw", ".cr2", ".nef", ".arw", ".dng",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".3gp", ".webm", ".m4v"}
MEDIA_EXTENSIONS = PHOTO_EXTENSIONS | VIDEO_EXTENSIONS

# Formats needing an optional decoder plugin (see decode.py) [M3].
HEIC_EXTENSIONS = {".heic", ".heif"}
RAW_EXTENSIONS = {".raw", ".cr2", ".nef", ".arw", ".dng"}


def is_media_file(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_EXTENSIONS


def classify_media_type(path: Path) -> str:
    """Return 'PHOTO' or 'VIDEO' for a media path."""
    return "VIDEO" if path.suffix.lower() in VIDEO_EXTENSIONS else "PHOTO"
