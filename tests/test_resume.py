"""Resume primitives: atomic writes, job paths, idempotency."""
import json

from src.utils.resume import (sanitize_id, atomic_write_json, read_json,
                              clean_job_path, adv_job_path, is_done)


def test_sanitize_id():
    assert sanitize_id("n01440764/img_0.JPEG") == "n01440764__img_0.JPEG"


def test_atomic_write_and_read_roundtrip(tmp_path):
    p = tmp_path / "sub" / "a.json"
    atomic_write_json(p, {"x": 1, "y": [1, 2]})
    assert read_json(p) == {"x": 1, "y": [1, 2]}
    assert is_done(p)


def test_atomic_write_leaves_no_tmp(tmp_path):
    p = tmp_path / "a.json"
    atomic_write_json(p, {"ok": True})
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_read_json_missing_returns_default(tmp_path):
    assert read_json(tmp_path / "nope.json", default={"d": 1}) == {"d": 1}


def test_read_json_corrupt_returns_default(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    assert read_json(p, default=None) is None


def test_job_paths_are_distinct_and_stable(tmp_path):
    c = clean_job_path(tmp_path, "n1/x.JPEG")
    a1 = adv_job_path(tmp_path, "n1/x.JPEG", 8 / 255)
    a2 = adv_job_path(tmp_path, "n1/x.JPEG", 16 / 255)
    assert c.name == "n1__x.JPEG.json"
    assert a1 != a2
    assert "eps0.0314" in a1.name and "eps0.0627" in a2.name


def test_idempotency_skip_pattern(tmp_path):
    """Simulate the runner's skip: a done job is not recomputed."""
    jp = adv_job_path(tmp_path, "n1/x.JPEG", 8 / 255)
    assert not is_done(jp)
    atomic_write_json(jp, {"done": True})
    assert is_done(jp)                      # a rerun would skip this job
