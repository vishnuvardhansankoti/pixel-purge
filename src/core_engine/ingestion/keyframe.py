"""Video keyframe extraction via ffmpeg (graceful when ffmpeg is absent).

A single keyframe (t=1s, or t=0 for very short clips) is extracted next to the
video and used for all downstream visual analysis. If ffmpeg is not installed,
extraction is skipped and the video simply has no keyframe (hash-based dedup
still applies) — per the PRD error-handling table.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

KEYFRAME_SUFFIX = "_keyframe.jpg"


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _probe_duration(video_path: Path) -> float | None:
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return None


def extract_keyframe(video_path: Path, output_dir: Path | None = None) -> Path | None:
    """Extract one keyframe. Returns the keyframe path, or None if unavailable."""
    if not ffmpeg_available():
        return None

    output_dir = output_dir or video_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframe_path = output_dir / f"{video_path.stem}{KEYFRAME_SUFFIX}"

    duration = _probe_duration(video_path)
    seek = "1" if (duration is None or duration >= 1.0) else "0"

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-ss", seek, "-i", str(video_path),
                "-frames:v", "1", "-q:v", "2", str(keyframe_path),
            ],
            capture_output=True, timeout=60,
        )
    except subprocess.SubprocessError:
        return None

    return keyframe_path if (result.returncode == 0 and keyframe_path.exists()) else None
