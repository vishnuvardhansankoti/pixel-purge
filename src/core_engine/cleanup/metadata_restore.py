"""Format-aware restoration of GPS + capture time back into media files [M2].

v1.0 used piexif unconditionally, which is JPEG/TIFF-only, so PNG screenshots,
HEIC (iPhone default), and videos silently got no metadata and uploaded with the
wrong date. Here we route by format:

  * JPEG / TIFF        -> piexif (pure Python, always available)
  * HEIC / PNG / video -> exiftool subprocess when installed, else skip w/ reason

Returns a RestoreResult describing what happened so the caller can report a
summary instead of failing the whole run.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..models import MediaRecord

_PIEXIF_EXTS = {".jpg", ".jpeg", ".tif", ".tiff"}
_EXIFTOOL_EXTS = {
    ".heic", ".heif", ".png", ".webp",
    ".mp4", ".mov", ".avi", ".mkv", ".3gp", ".webm", ".m4v",
}


@dataclass
class RestoreResult:
    ok: bool
    reason: str = ""


def exiftool_available() -> bool:
    return shutil.which("exiftool") is not None


def _deg_to_dms_rational(deg: float):
    import piexif  # local import; part of core deps but keep module import-light

    deg_abs = abs(deg)
    d = int(deg_abs)
    m = int((deg_abs - d) * 60)
    s = round((deg_abs - d - m / 60) * 3600 * 100)
    return ((d, 1), (m, 1), (s, 100))


def _restore_piexif(path: Path, record: MediaRecord) -> RestoreResult:
    import piexif

    try:
        exif = piexif.load(str(path))
    except Exception:  # noqa: BLE001 - corrupt/absent EXIF, start fresh
        exif = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    if record.taken_timestamp is not None:
        dt = datetime.fromtimestamp(record.taken_timestamp, tz=timezone.utc)
        stamp = dt.strftime("%Y:%m:%d %H:%M:%S").encode()
        exif["Exif"][piexif.ExifIFD.DateTimeOriginal] = stamp
        exif["Exif"][piexif.ExifIFD.DateTimeDigitized] = stamp
        exif["0th"][piexif.ImageIFD.DateTime] = stamp

    if record.latitude is not None and record.longitude is not None:
        gps = exif.setdefault("GPS", {})
        gps[piexif.GPSIFD.GPSLatitudeRef] = "N" if record.latitude >= 0 else "S"
        gps[piexif.GPSIFD.GPSLatitude] = _deg_to_dms_rational(record.latitude)
        gps[piexif.GPSIFD.GPSLongitudeRef] = "E" if record.longitude >= 0 else "W"
        gps[piexif.GPSIFD.GPSLongitude] = _deg_to_dms_rational(record.longitude)

    try:
        piexif.insert(piexif.dump(exif), str(path))
        return RestoreResult(True)
    except Exception as e:  # noqa: BLE001
        return RestoreResult(False, f"piexif insert failed: {e}")


def _restore_exiftool(path: Path, record: MediaRecord) -> RestoreResult:
    if not exiftool_available():
        return RestoreResult(False, "exiftool not installed (brew install exiftool)")

    args = ["exiftool", "-overwrite_original", "-q"]
    if record.taken_timestamp is not None:
        dt = datetime.fromtimestamp(record.taken_timestamp, tz=timezone.utc)
        stamp = dt.strftime("%Y:%m:%d %H:%M:%S")
        args += [f"-DateTimeOriginal={stamp}", f"-CreateDate={stamp}",
                 f"-QuickTime:CreateDate={stamp}"]
    if record.latitude is not None and record.longitude is not None:
        args += [
            f"-GPSLatitude={abs(record.latitude)}",
            f"-GPSLatitudeRef={'N' if record.latitude >= 0 else 'S'}",
            f"-GPSLongitude={abs(record.longitude)}",
            f"-GPSLongitudeRef={'E' if record.longitude >= 0 else 'W'}",
        ]
    args.append(str(path))
    try:
        r = subprocess.run(args, capture_output=True, timeout=60)
        if r.returncode == 0:
            return RestoreResult(True)
        return RestoreResult(False, r.stderr.decode()[:200])
    except subprocess.SubprocessError as e:
        return RestoreResult(False, str(e))


def restore_metadata(path: Path, record: MediaRecord) -> RestoreResult:
    """Write GPS + capture time back into `path` based on its format."""
    if record.taken_timestamp is None and record.latitude is None:
        return RestoreResult(True, "nothing to restore")

    ext = path.suffix.lower()
    if ext in _PIEXIF_EXTS:
        return _restore_piexif(path, record)
    if ext in _EXIFTOOL_EXTS:
        return _restore_exiftool(path, record)
    return RestoreResult(False, f"unsupported format for metadata restore: {ext}")
