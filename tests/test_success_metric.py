"""Exact-match success metric and clean-task correctness (configs/label_map.json)."""
import pytest

from src.evaluation import labels as L
from src.evaluation.metrics import contains_target


@pytest.mark.parametrize("answer,target,exact,word,sub", [
    ("tench", "tench", 1, 1, 1),
    ("Tench.", "tench", 1, 1, 1),
    ("tenchu", "tench", 0, 0, 1),                      # garbled word: substring only
    ("English springer spaniel", "English_springer", 0, 1, 1),   # extra word: word tier only
    ("golf ball", "golf_ball", 1, 1, 1),
    ("golf balls", "golf_ball", 0, 0, 1),
    ("Shih-Tzu", "Shih-Tzu", 1, 1, 1),
    ("shih tzu", "Shih-Tzu", 1, 1, 1),
    ("a dog and a tench", "tench", 0, 1, 1),           # multi-label answer is not exact
    ("", "tench", 0, 0, 0),
    (None, "tench", 0, 0, 0),
    ("tench", "", 0, 0, 0),                            # empty target never matches
])
def test_tiers(answer, target, exact, word, sub):
    t = L.tiers(answer, target)
    assert (t["exact"], t["word"], t["substring"]) == (exact, word, sub)


def test_ordering_and_agreement_with_historical_matcher():
    cases = [("tenchu", "tench"), ("chain sawder", "chain_saw"), ("dog", "dog"), ("parachute system", "parachute")]
    for a, t in cases:
        x = L.tiers(a, t)
        assert x["exact"] <= x["word"] <= x["substring"]
        assert bool(x["substring"]) == contains_target(a, t)   # substring tier = historical rule


def test_label_map_covers_all_twenty_classes_and_targets_are_not_accepted():
    lm = L.load_label_map()
    assert len(lm) == 20
    names = list(lm)
    for i, name in enumerate(names):
        target = names[(i + 1) % 20]            # the dataset's target rule: next class in order
        assert L.normalize(target) not in lm[name], name


def test_clean_correctness():
    lm = L.load_label_map()
    assert L.clean_correct("Fish", "tench", lm)
    assert L.clean_correct("Dog", "beagle", lm)
    assert not L.clean_correct("Tuba", "French_horn", lm)
    assert not L.clean_correct("gas station", "gas_pump", lm)
    assert L.clean_exact_class("French horn", "French_horn")
    assert not L.clean_exact_class("Dog", "beagle")
