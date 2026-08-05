"""Video keyframe extraction via ffmpeg (graceful when ffmpeg is absent).

The middle keyframe is used for thumbnails + vision; several evenly-spaced frames
are extracted for robust multi-frame video dedup [M9]. If ffmpeg is not installed,
extraction is skipped and the video simply has no keyframe (hash-based dedup still
applies) — per the PRD error-handling table.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

KEYFRAME_SUFFIX = "_keyframe.jpg"


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe_duration(video_path: Path) -> float | None:
    """Public wrapper for the video duration probe."""
    return _probe_duration(video_path)


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


@dataclass
class KeyframeSet:
    paths: list[Path]          # all extracted frames, in time order
    primary: Path | None       # middle frame (thumbnail + vision)
    duration: float | None


def _extract_frame(video_path: Path, seek: float, out_path: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{seek:.3f}", "-i", str(video_path),
                "-frames:v", "1", "-q:v", "2", str(out_path),
            ],
            capture_output=True, timeout=60,
        )
    except subprocess.SubprocessError:
        return False
    return result.returncode == 0 and out_path.exists()


def extract_keyframes(
    video_path: Path, output_dir: Path | None = None, n: int = 3
) -> KeyframeSet:
    """Extract ``n`` evenly-spaced keyframes for multi-frame dedup [M9].

    Frames are taken at (i+1)/(n+1) of the duration (e.g. 25/50/75% for n=3), so
    the opening/closing seconds don't dominate. Returns whatever was extracted
    (possibly empty when ffmpeg is missing or the clip is unreadable).
    """
    if not ffmpeg_available():
        return KeyframeSet([], None, None)

    output_dir = output_dir or video_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = _probe_duration(video_path)

    if duration and duration > 0:
        seeks = [duration * (i + 1) / (n + 1) for i in range(n)]
    else:
        seeks = [1.0]  # unknown duration: one frame at t=1s

    paths: list[Path] = []
    for idx, seek in enumerate(seeks):
        out = output_dir / f"{video_path.stem}_kf{idx}.jpg"
        if _extract_frame(video_path, seek, out):
            paths.append(out)

    primary = paths[len(paths) // 2] if paths else None
    return KeyframeSet(paths, primary, duration)
