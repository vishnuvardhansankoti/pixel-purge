"""Multi-frame video comparison helpers [M9].

A single keyframe is fragile for video near-dup: a re-encode or trim shifts the
frame, and two unrelated clips with similar opening frames false-match. We hash
several frames per video and compare frame *sets* (best-matching frame wins), and
gate video↔video comparisons by duration so clips of very different length aren't
merged.

Pure functions — unit-tested with hand-provided hashes, no ffmpeg required.
"""

from __future__ import annotations

from typing import Sequence


def frame_set_distance(a: Sequence, b: Sequence) -> int:
    """Minimum pairwise Hamming distance between two sets of frame pHashes.

    Using the minimum means a trimmed/re-encoded clip still matches on the frames
    it shares with the original. Inputs are imagehash.ImageHash (support ``-``).
    """
    if not a or not b:
        return 1 << 30
    return min(int(ha - hb) for ha in a for hb in b)


def duration_gate(dur_a: float | None, dur_b: float | None, tol: float = 0.25) -> bool:
    """True if two durations are close enough to be the same video.

    Unknown durations (None/0) pass — we don't want a missing probe to block a
    legitimate exact/near match. `tol` is the max fractional difference.
    """
    if not dur_a or not dur_b:
        return True
    longer = max(dur_a, dur_b)
    return abs(dur_a - dur_b) / longer <= tol
