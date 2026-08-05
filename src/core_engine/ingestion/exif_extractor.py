"""Embedded-EXIF fallback for when a sidecar JSON is missing or incomplete.

Uses Pillow's EXIF reader (works for JPEG/TIFF/HEIC via pillow-heif). Best-effort:
any parse failure is swallowed so ingestion continues.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PIL import ExifTags, Image

from ..models import MediaRecord
from .decode import UnsupportedImageError

_DATETIME_TAGS = ("DateTimeOriginal", "DateTimeDigitized", "DateTime")
_TAG_IDS = {name: tid for tid, name in ExifTags.TAGS.items()}
_GPS_IDS = {name: tid for tid, name in ExifTags.GPSTAGS.items()}


def _parse_exif_datetime(value: str) -> int | None:
    # EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(value.strip(), fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except (ValueError, AttributeError):
            continue
    return None


def _dms_to_deg(dms, ref) -> float | None:
    try:
        d, m, s = (float(x) for x in dms)
        deg = d + m / 60.0 + s / 3600.0
        if ref in ("S", "W"):
            deg = -deg
        return deg
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def extract_exif_metadata(record: MediaRecord, media_file: Path) -> bool:
    """Fill missing taken_timestamp / lat / lon from embedded EXIF. Returns True if
    anything was filled."""
    try:
        img = Image.open(media_file)
        exif = img.getexif()
    except (UnsupportedImageError, OSError, ValueError):
        return False
    if not exif:
        return False

    filled = False

    if record.taken_timestamp is None:
        for tag in _DATETIME_TAGS:
            tid = _TAG_IDS.get(tag)
            if tid and exif.get(tid):
                ts = _parse_exif_datetime(str(exif.get(tid)))
                if ts:
                    record.taken_timestamp = ts
                    filled = True
                    break

    if record.latitude is None:
        gps_ifd = None
        try:
            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        except (AttributeError, KeyError, ValueError):
            gps_ifd = None
        if gps_ifd:
            lat = _dms_to_deg(
                gps_ifd.get(_GPS_IDS["GPSLatitude"]),
                gps_ifd.get(_GPS_IDS["GPSLatitudeRef"]),
            )
            lon = _dms_to_deg(
                gps_ifd.get(_GPS_IDS["GPSLongitude"]),
                gps_ifd.get(_GPS_IDS["GPSLongitudeRef"]),
            )
            if lat is not None and lon is not None:
                record.latitude = lat
                record.longitude = lon
                filled = True

    return filled
