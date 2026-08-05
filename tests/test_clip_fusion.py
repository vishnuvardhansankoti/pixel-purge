"""CLIP + quality rule-fusion logic (pure function, mocked scores)."""

from core_engine.vision.clip_tagger import fuse_classification
from core_engine.vision.taxonomy import ADHOC_PURGE, FAMILY_KEEP, TRIP


def _probs(**overrides) -> dict[str, float]:
    """Build a prompt->prob map, dominant on one chosen prompt."""
    from core_engine.vision.taxonomy import PROMPTS

    base = {p: 0.01 for p in PROMPTS}
    for prompt, val in overrides.items():
        base[prompt] = val
    return base


PERSON_PROMPT = "a portrait photo of a person or a selfie"
LANDMARK_PROMPT = "a travel photo of a landmark or tourist attraction"


def test_people_photo_is_family_keep():
    r = fuse_classification(_probs(**{PERSON_PROMPT: 0.9}), blur=500.0, density=0.05)
    assert r.bucket == FAMILY_KEEP


def test_landmark_is_trip():
    r = fuse_classification(_probs(**{LANDMARK_PROMPT: 0.8}), blur=500.0, density=0.05)
    assert r.bucket == TRIP


def test_text_density_overrides_to_purge():
    # Even if CLIP leans "person", high text density forces ADHOC_PURGE.
    r = fuse_classification(_probs(**{PERSON_PROMPT: 0.7}), blur=500.0, density=0.6)
    assert r.bucket == ADHOC_PURGE
    assert r.label == "text_heavy"


def test_blurry_non_family_is_purged():
    r = fuse_classification(_probs(**{LANDMARK_PROMPT: 0.6}), blur=10.0, density=0.05)
    assert r.bucket == ADHOC_PURGE
    assert r.label == "blurry"


def test_blurry_confident_family_is_kept():
    # A slightly-soft but confident family shot is NOT purged.
    r = fuse_classification(_probs(**{PERSON_PROMPT: 0.8}), blur=10.0, density=0.05)
    assert r.bucket == FAMILY_KEEP
