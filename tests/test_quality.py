"""Blur + text-density heuristics (pure numpy — tested for real)."""

import numpy as np
from PIL import Image

from core_engine.vision import quality


def _img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype("uint8"), "RGB")


def test_smooth_image_is_blurry():
    # A smooth gradient has almost no high-frequency detail -> low Laplacian variance.
    grad = np.tile(np.linspace(0, 255, 128), (128, 1))
    img = _img(np.stack([grad] * 3, axis=-1))
    score = quality.blur_score(img)
    assert score < quality.BLUR_THRESHOLD
    assert quality.is_blurry(score)


def test_high_detail_image_is_sharp():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(128, 128, 3))
    score = quality.blur_score(_img(noise))
    assert score > quality.BLUR_THRESHOLD
    assert not quality.is_blurry(score)


def test_document_has_high_text_density():
    # White canvas with small dark marks in a dense grid across the whole image,
    # like text on a page: edges present in (nearly) every block.
    arr = np.full((160, 160, 3), 255)
    for y in range(4, 160, 8):
        for x in range(4, 160, 8):
            arr[y:y + 3, x:x + 5] = 0
    density = quality.text_density(_img(arr))
    assert density > quality.TEXT_DENSITY_THRESHOLD
    assert quality.is_text_heavy(density)


def test_photo_has_low_text_density():
    # A single subject on a plain background: edges concentrated, not spread everywhere.
    arr = np.full((160, 160, 3), 200)
    arr[60:100, 60:100] = 20  # one object
    density = quality.text_density(_img(arr))
    assert density < quality.TEXT_DENSITY_THRESHOLD
    assert not quality.is_text_heavy(density)
