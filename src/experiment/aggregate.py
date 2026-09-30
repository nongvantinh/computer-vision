"""Rebuild derived result files from the job directories (item 22).

Derived files are disposable: they are regenerated from jobs/ every session, so a
partial run still yields readable results. Nothing here recomputes the model; it
only collates finished job JSON into JSONL tables and a coverage summary.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..utils.resume import read_json, atomic_write_json
from .layout import RunLayout


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _collect(job_dir: Path) -> list[dict]:
    out = []
    for jp in sorted(job_dir.glob("*.json")):
        rec = read_json(jp)
        if rec:
            out.append(rec)
    return out


def aggregate(lay: RunLayout) -> dict:
    """Collate clean/attack/defense jobs into the derived JSONL tables + coverage."""
    clean = _collect(lay.clean_jobs)
    attack = _collect(lay.attack_jobs)
    defense = _collect(lay.defense_jobs)

    _write_jsonl(lay.clean_results, clean)
    _write_jsonl(lay.attack_results, attack)
    _write_jsonl(lay.defense_results, defense)

    ok_defense = [r for r in defense if not r.get("error")]
    failed_defense = [r for r in defense if r.get("error")]

    # coverage counts, honest about failures (item 12)
    by_defense = defaultdict(lambda: {"ok": 0, "failed": 0})
    for r in defense:
        key = r.get("defense", "?")
        by_defense[key]["failed" if r.get("error") else "ok"] += 1

    summary = {
        "n_clean_jobs": len(clean),
        "n_attack_jobs": len(attack),
        "n_defense_jobs": len(defense),
        "n_defense_ok": len(ok_defense),
        "n_defense_failed": len(failed_defense),
        "adv_images": len(list(lay.adv.glob("*.npy"))),
        "epsilons": sorted({r["epsilon"] for r in attack if "epsilon" in r}),
        "defenses_seen": sorted(by_defense),
        "coverage_by_defense": dict(by_defense),
        "failures": [{"image_id": r.get("image_id"), "epsilon": r.get("epsilon"),
                      "defense": r.get("defense"), "error": r.get("error")}
                     for r in failed_defense],
    }
    atomic_write_json(lay.summary, summary)
    return summary
