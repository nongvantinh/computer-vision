"""Analysis helpers must not rewrite or empty stored result tables (regression for the
truncation seen when check_gates.py aggregated a partial local copy of a run)."""
import json

from src.experiment.aggregate import ensure_derived
from src.experiment.layout import RunLayout


def _lay(tmp_path):
    root = tmp_path / "run"
    (root / "jobs" / "defense").mkdir(parents=True)
    return RunLayout(root)


def test_existing_tables_are_never_overwritten_when_jobs_are_missing(tmp_path):
    lay = _lay(tmp_path)
    lay.defense_results.write_text(json.dumps({"image_id": "a", "defense": "D0"}) + "\n")
    before = lay.defense_results.read_text()
    assert ensure_derived(lay) is False
    assert lay.defense_results.read_text() == before


def test_existing_tables_are_not_overwritten_even_if_jobs_exist(tmp_path):
    lay = _lay(tmp_path)
    lay.defense_results.write_text("stored\n")
    (lay.defense_jobs / "j.json").write_text(json.dumps({"image_id": "b", "defense": "jpeg50"}))
    assert ensure_derived(lay) is False
    assert lay.defense_results.read_text() == "stored\n"


def test_no_write_when_there_are_no_jobs(tmp_path):
    lay = _lay(tmp_path)
    assert ensure_derived(lay) is False
    assert not lay.defense_results.exists()
