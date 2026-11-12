#!/usr/bin/env python
"""Read-only rescoring of stored predictions with the exact-match success metric.

    python scripts/rescore_predictions.py --runs results/runs --out results/corrected

Never writes into a run directory. For every run's defense_results.jsonl it recomputes the
three matching tiers (exact, word, substring) from the stored `answer` and `target`, checks
that the substring tier reproduces the stored historical `target_success`, and writes
per-(run, defense, epsilon) counts plus the list of rows whose outcome changes under the
primary exact criterion. Also reports clean-task correctness per defense from
clean_results.jsonl and whether any clean answer already contains the target.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import labels as L  # noqa: E402


def jl(p: Path):
    return [json.loads(x) for x in p.open() if x.strip()] if p.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results/corrected")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lm = L.load_label_map()
    counts, changed, mismatch_hist = [], [], 0
    clean_summary = []
    for run in sorted(p for p in Path(a.runs).iterdir() if p.is_dir()):
        rows = [r for r in jl(run / "defense_results.jsonl") if not r.get("error")]
        if not rows:
            continue
        agg = defaultdict(lambda: {"n": 0, "historical": 0, "substring": 0, "word": 0, "exact": 0})
        for r in rows:
            t = L.tiers(r["answer"], r["target"])
            key = (run.name, r["defense"], round(float(r["epsilon"]), 6))
            g = agg[key]
            g["n"] += 1
            g["historical"] += int(r["target_success"])
            for k in ("substring", "word", "exact"):
                g[k] += t[k]
            if int(r["target_success"]) != t["substring"]:
                mismatch_hist += 1
            if int(r["target_success"]) and not t["exact"]:
                changed.append({"run": run.name, "defense": r["defense"], "epsilon": r["epsilon"],
                                "image_id": r["image_id"], "answer": r["answer"], "target": r["target"],
                                "word": t["word"]})
        for (rid, d, eps), g in agg.items():
            counts.append({"run": rid, "defense": d, "epsilon": eps, **g,
                           "changed_by_exact": g["historical"] - g["exact"]})
        cl = [r for r in jl(run / "clean_results.jsonl") if not r.get("error")]
        if cl:
            cls = {r["image_id"]: r["class"] for r in jl(run / "defense_results.jsonl")}
            byd = defaultdict(list)
            for r in cl:
                r = dict(r)
                r["gt"] = r.get("gt") or cls.get(r["image_id"])
                if r["gt"] is None or "defense" not in r:
                    continue
                byd[r["defense"]].append(r)
            for d, rs in byd.items():
                clean_summary.append({
                    "run": run.name, "defense": d, "n": len(rs),
                    "accuracy_accepted_set": sum(L.clean_correct(r["answer"], r["gt"], lm) for r in rs) / len(rs),
                    "accuracy_exact_class_name": sum(L.clean_exact_class(r["answer"], r["gt"]) for r in rs) / len(rs),
                    "preservation_vs_clean_answer": sum(int(r["preserved"]) for r in rs) / len(rs),
                })
    clean_target = {}
    for run in sorted(p for p in Path(a.runs).iterdir() if p.is_dir()):
        seen = {}
        for r in jl(run / "defense_results.jsonl"):
            seen[r["image_id"]] = (r["clean_answer"], r["target"])
        if seen:
            clean_target[run.name] = {"n_images": len(seen),
                                      "clean_answer_contains_target_substring": sum(L.substring_match(a_, t) for a_, t in seen.values()),
                                      "clean_answer_equals_target_exact": sum(L.exact_match(a_, t) for a_, t in seen.values())}
    (out / "exact_match_counts.json").write_text(json.dumps(counts, indent=1))
    (out / "rows_changed_by_exact.json").write_text(json.dumps(changed, indent=1))
    (out / "clean_task.json").write_text(json.dumps(clean_summary, indent=1))
    (out / "clean_answer_vs_target.json").write_text(json.dumps(clean_target, indent=1))
    with (out / "exact_match_counts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(counts[0]))
        w.writeheader()
        w.writerows(counts)
    print(f"{len(counts)} condition rows; {len(changed)} historical successes lost under exact; "
          f"{mismatch_hist} rows where the substring tier disagrees with the stored historical flag")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
