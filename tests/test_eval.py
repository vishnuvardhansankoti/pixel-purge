"""H2 — vision accuracy eval harness (pure metrics + labeled-set loading)."""

import pytest

from core_engine.vision.eval import (
    EvalCase,
    compute_metrics,
    evaluate,
    load_eval_manifest,
)
from core_engine.vision.taxonomy import ADHOC_PURGE, FAMILY_KEEP, OTHER, TRIP


def test_perfect_accuracy():
    pairs = [(ADHOC_PURGE, ADHOC_PURGE), (TRIP, TRIP), (FAMILY_KEEP, FAMILY_KEEP)]
    m = compute_metrics(pairs)
    assert m.accuracy == 1.0
    assert m.per_bucket[TRIP]["precision"] == 1.0
    assert m.per_bucket[TRIP]["recall"] == 1.0


def test_confusion_and_precision_recall():
    # 2x ADHOC correct; 1x FAMILY predicted as ADHOC (a purge false-positive).
    pairs = [
        (ADHOC_PURGE, ADHOC_PURGE),
        (ADHOC_PURGE, ADHOC_PURGE),
        (FAMILY_KEEP, ADHOC_PURGE),
    ]
    m = compute_metrics(pairs)
    assert m.accuracy == pytest.approx(2 / 3)
    # ADHOC precision = 2 correct / 3 predicted-ADHOC
    assert m.per_bucket[ADHOC_PURGE]["precision"] == pytest.approx(2 / 3)
    assert m.per_bucket[ADHOC_PURGE]["recall"] == 1.0
    # FAMILY recall = 0 (its one instance was mislabeled)
    assert m.per_bucket[FAMILY_KEEP]["recall"] == 0.0
    assert m.confusion[FAMILY_KEEP][ADHOC_PURGE] == 1


def test_evaluate_with_fake_classifier():
    cases = [EvalCase("a", TRIP), EvalCase("b", FAMILY_KEEP), EvalCase("c", OTHER)]
    # classifier that always predicts by looking up a dict
    preds = {"a": TRIP, "b": FAMILY_KEEP, "c": ADHOC_PURGE}
    m = evaluate(cases, lambda p: preds[p])
    assert m.total == 3
    assert m.correct == 2
    assert m.accuracy == pytest.approx(2 / 3)


def test_load_manifest(tmp_path):
    p = tmp_path / "eval.csv"
    p.write_text("path,bucket\n/x/a.jpg,TRIP\n/x/b.jpg,ADHOC_PURGE\n")
    cases = load_eval_manifest(p)
    assert [c.expected for c in cases] == [TRIP, ADHOC_PURGE]
    assert cases[0].path == "/x/a.jpg"


def test_load_manifest_rejects_bad_bucket(tmp_path):
    p = tmp_path / "eval.csv"
    p.write_text("path,bucket\n/x/a.jpg,NONSENSE\n")
    with pytest.raises(ValueError):
        load_eval_manifest(p)
