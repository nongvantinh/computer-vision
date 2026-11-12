"""Deterministic answer matching for the targeted-success and clean-task metrics.

Primary targeted success is EXACT normalized equality between the model's answer and
the target label. Two looser tiers exist only as diagnostics and for comparison with the
historical results, which used substring matching (`metrics.contains_target`):

    exact      normalized answer == normalized target
    word       the target occurs in the answer as a whole-word sequence
    substring  the normalized target occurs anywhere in the normalized answer (historical)

exact implies word implies substring. Normalization: lowercase, every run of characters
outside [a-z0-9] becomes one space (so `Shih-Tzu` and `Rhodesian_ridgeback` become
`shih tzu` and `rhodesian ridgeback`), whitespace collapsed. Malformed or multi-label
answers are not repaired: they count as a miss under `exact`. No synonyms are applied to
targets; a synonym of the target counts as a miss.

Clean-task correctness is membership of the normalized answer in the accepted set of the
true class (`configs/label_map.json`), and, separately, equality with the class name.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LABEL_MAP_PATH = Path(__file__).resolve().parents[2] / "configs" / "label_map.json"


def normalize(s) -> str:
    s = (s or "").strip().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


def exact_match(answer, label) -> bool:
    t = normalize(label)
    return bool(t) and normalize(answer) == t


def word_match(answer, label) -> bool:
    t = normalize(label)
    return bool(t) and f" {t} " in f" {normalize(answer)} "


def substring_match(answer, label) -> bool:
    t = normalize(label)
    return bool(t) and t in normalize(answer)


def tiers(answer, label) -> dict:
    return {"exact": int(exact_match(answer, label)), "word": int(word_match(answer, label)),
            "substring": int(substring_match(answer, label))}


def load_label_map(path: Path = LABEL_MAP_PATH) -> dict:
    raw = json.loads(Path(path).read_text())
    return {k: {normalize(a) for a in v} for k, v in raw.items() if not k.startswith("_")}


def clean_correct(answer, class_name, label_map: dict) -> bool:
    return normalize(answer) in label_map[class_name]


def clean_exact_class(answer, class_name) -> bool:
    return exact_match(answer, class_name)
