"""Figures are generated from a statistics dict and written as PNG + PDF."""
import pytest

pytest.importorskip("matplotlib")

from src.analysis import figures


STATS = {
    "asr": {
        "D0@0.0314": {"asr": 0.8, "asr_ci95": [0.6, 0.95], "n": 12},
        "jpeg75@0.0314": {"asr": 0.3, "asr_ci95": [0.1, 0.5], "n": 12},
        "dangerzone@0.0314": {"asr": 0.1, "asr_ci95": [0.0, 0.25], "n": 12},
    },
    "preservation": {
        "D0": {"preservation": 1.0, "ci95": [1.0, 1.0], "n": 12},
        "jpeg75": {"preservation": 0.9, "ci95": [0.7, 1.0], "n": 12},
        "dangerzone": {"preservation": 0.7, "ci95": [0.5, 0.9], "n": 12},
    },
    "mechanism": {
        "jpeg75": {"energy_removed_frac_mean": 0.6, "residual_highfreq_frac_mean": 0.2},
        "dangerzone": {"energy_removed_frac_mean": 0.9, "residual_highfreq_frac_mean": 0.05},
    },
}


def test_all_figures_written(tmp_path):
    made = figures.all_figures(STATS, tmp_path)
    assert set(made) == {"asr", "preservation", "mechanism"}
    for paths in made.values():
        for p in paths:
            from pathlib import Path
            assert Path(p).exists() and Path(p).stat().st_size > 0
    assert (tmp_path / "fig_asr_by_defense.pdf").exists()


def test_defense_controlled_mechanism_is_plotted(tmp_path):
    stats = {"mechanism": {
        "D0": {"raw_highfreq_frac_mean": 0.82, "post_highfreq_frac_mean": 0.82,
               "energy_ratio_mean": 1.0},
        "dangerzone": {"raw_highfreq_frac_mean": 0.82, "post_highfreq_frac_mean": 0.61,
                       "energy_ratio_mean": 0.59},
    }}
    made = figures.all_figures(stats, tmp_path)
    assert set(made) == {"mechanism"}
    assert (tmp_path / "fig_mechanism.png").stat().st_size > 5000


def test_missing_sections_skip_cleanly(tmp_path):
    assert figures.all_figures({}, tmp_path) == {}
