from pathlib import Path

from core_engine.models import MediaRecord
from core_engine.ingestion.sidecar_merger import merge_sidecar_metadata, resolve_sidecar
from tests.conftest import make_image, make_sidecar


def _rec(path: Path) -> MediaRecord:
    return MediaRecord(path.name, str(path), path.stat().st_size, "PHOTO")


def test_resolve_direct_json(tmp_path):
    img = make_image(tmp_path / "IMG_1.jpg")
    sc = make_sidecar(img, taken_ts=100)
    assert resolve_sidecar(img) == sc


def test_resolve_supplemental_metadata(tmp_path):
    img = make_image(tmp_path / "IMG_2.jpg")
    sc = img.with_name("IMG_2.jpg.supplemental-metadata.json")
    sc.write_text("{}")
    assert resolve_sidecar(img) == sc


def test_resolve_duplicate_named_export(tmp_path):
    img = make_image(tmp_path / "IMG_3(1).jpg")
    sc = img.with_name("IMG_3.jpg(1).json")
    sc.write_text("{}")
    assert resolve_sidecar(img) == sc


def test_resolve_edited_points_to_base(tmp_path):
    make_image(tmp_path / "IMG_4-edited.jpg")
    base_json = tmp_path / "IMG_4.json"
    base_json.write_text("{}")
    assert resolve_sidecar(tmp_path / "IMG_4-edited.jpg") == base_json


def test_merge_fields(tmp_path):
    img = make_image(tmp_path / "IMG_5.jpg")
    make_sidecar(img, taken_ts=1700, lat=37.5, lon=-122.1,
                 description="hi", title="IMG_5.jpg", views=9)
    rec = _rec(img)
    assert merge_sidecar_metadata(rec, resolve_sidecar(img))
    assert rec.taken_timestamp == 1700
    assert rec.latitude == 37.5 and rec.longitude == -122.1
    assert rec.user_description == "hi"
    assert rec.view_count == 9


def test_zero_gps_treated_as_unknown(tmp_path):
    img = make_image(tmp_path / "IMG_6.jpg")
    make_sidecar(img, taken_ts=1700, lat=0.0, lon=0.0)
    rec = _rec(img)
    merge_sidecar_metadata(rec, resolve_sidecar(img))
    assert rec.latitude is None and rec.longitude is None
