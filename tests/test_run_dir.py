"""Run directories must never overwrite a previous run's results."""
from src.utils.config import new_run_dir


def test_new_run_dir_never_collides(tmp_path):
    a = new_run_dir("mvp", base=tmp_path)
    b = new_run_dir("mvp", base=tmp_path)
    assert a != b
    for sub in ("clean", "attacks", "metrics", "statistics", "plots", "logs"):
        assert (a / sub).is_dir()
        assert (b / sub).is_dir()
