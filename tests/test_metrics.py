"""Metrics: ASR, accuracy, label matching, fidelity."""
import numpy as np
import pytest

from src.evaluation import metrics as M


def test_targeted_asr_and_accuracy():
    preds = ["cat", "dog", "cat", "banana"]
    targets = ["cat", "cat", "cat", "cat"]
    labels = ["cat", "dog", "fish", "banana"]
    assert M.targeted_asr(preds, targets) == pytest.approx(2 / 4)
    assert M.accuracy(preds, labels) == pytest.approx(3 / 4)


def test_targeted_asr_length_mismatch():
    with pytest.raises(ValueError):
        M.targeted_asr(["a"], ["a", "b"])


def test_match_label_substring_and_longest():
    names = ["cat", "wildcat", "dog"]
    assert M.match_label("this is a wildcat sitting", names) == "wildcat"
    assert M.match_label("a plain dog", names) == "dog"
    assert M.match_label("nothing here", names) is None


def test_psnr_identical_is_inf_and_linf_zero():
    img = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    assert M.psnr(img, img) == float("inf")
    assert M.linf(img, img) == 0.0


def test_ssim_identical_is_one():
    img = np.random.default_rng(1).random((32, 32, 3)).astype(np.float32)
    assert M.ssim(img, img) == pytest.approx(1.0, abs=1e-4)


def test_linf_bound():
    a = np.zeros((4, 4, 3), np.float32)
    b = np.full((4, 4, 3), 0.1, np.float32)
    assert M.linf(a, b) == pytest.approx(0.1, abs=1e-6)
