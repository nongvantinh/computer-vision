#!/usr/bin/env python
"""Convergence analysis (docs/phase3_plan.md section 2). Read-only.

    python scripts/analyze_convergence.py --runs results/runs --out results/convergence

Per attack (jpeg50, dangerzone_ll at 16/255, 100-image subset):
  - final-iterate exact-match success by cumulative step through the attack's own real
    defense (500 = the confirmatory run, 600..2000 = stored checkpoints, 1000/2000 also
    from the conv runs' own defense rows as a cross-check);
  - best-so-far success (nondecreasing by construction);
  - successes that are gained and later lost along the checkpoint grid;
  - the pre-specified decision rule: if final-iterate success at 2000 exceeds 500 by
    more than 15 points, the confirmatory budget is not adequate for a ranking claim.

Also reports the probe arms on the 25-image subset: two fresh random starts
(restart variance) and the momentum+step-decay optimizer variant, both against the
confirmatory 500-step result on the same images. Exact match is the primary metric;
substring and whole-word are reported as diagnostics. Nothing outside --out is written.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import labels as L  # noqa: E402

EPS = 0.062745098
ATTACKS = {"jpeg50": "jpeg50", "dangerzone_ll": "dangerzone_ll"}   # attack -> scored-through
CONF = {"jpeg50": "conf_jpeg50_e16", "dangerzone_ll": "conf_dz_ll_e16"}
CONV = {a: [f"conv_{a}_e16_s1000", f"conv_{a}_e16_s2000"] for a in ATTACKS}
PROBES = {"conv_jpeg50_e16_r1": "restart 1 (seed 1235)",
          "conv_jpeg50_e16_r2": "restart 2 (seed 1236)",
          "conv_jpeg50_e16_mom": "momentum 0.9 + linear step decay"}
MEI = 0.15


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


def vec(rows: list[dict], defense: str, metric: str) -> dict[str, int]:
    out = {}
    for r in rows:
        if r.get("error") or r.get("defense") != defense:
            continue
        if abs(float(r["epsilon"]) - EPS) >= 1e-6:
            continue
        out[r["image_id"]] = int(L.tiers(r["answer"], r["target"])[metric])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results/convergence")
    ap.add_argument("--metric", default="exact", choices=["exact", "word", "substring"])
    ap.add_argument("--manifest", default="configs/conv_100.json")
    ap.add_argument("--probe-manifest", default="configs/conv_25.json")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = Path(args.runs)

    ids = [s["image_id"] for s in json.loads(Path(args.manifest).read_text())["samples"]]
    probe_ids = [s["image_id"] for s in json.loads(Path(args.probe_manifest).read_text())["samples"]]

    decision, tables, lost = [], [], []
    for atk, scored_through in ATTACKS.items():
        base = vec(jl(runs / CONF[atk] / "defense_results.jsonl"), scored_through, args.metric)
        base = {i: base[i] for i in ids if i in base}
        # checkpoint curve: step -> {image: success}; jpeg50 comes from ckpt_results,
        # dangerzone_ll from the sanitized coarse grid in ckpt_results as well.
        by_step: dict[int, dict[str, int]] = {500: base}
        for rid in CONV[atk]:
            for r in jl(runs / rid / "ckpt_results.jsonl"):
                if r.get("error") or r.get("defense") != scored_through:
                    continue
                by_step.setdefault(int(r["ckpt_step"]), {})[r["image_id"]] = int(
                    L.tiers(r["answer"], r["target"])[args.metric])
            # final iterate through the conv run's own defense rows (cross-check)
            fin = vec(jl(runs / rid / "defense_results.jsonl"), scored_through, args.metric)
            fin = {i: fin[i] for i in ids if i in fin}
            if fin:
                step = 1000 if rid.endswith("s1000") else 2000
                cross = by_step.get(step, {})
                n_diff = sum(1 for i in fin if i in cross and fin[i] != cross[i])
                if n_diff:
                    decision.append({"attack": atk, "run": rid,
                                     "note": "final-iterate defense row differs from its "
                                             "checkpoint row on the same images",
                                     "n_diff": n_diff})
                by_step[step] = fin
        steps = sorted(by_step)
        # per-step final-iterate and best-so-far success
        best: dict[str, int] = {i: 0 for i in ids}
        for s in steps:
            cur = by_step[s]
            for i in ids:
                best[i] = max(best.get(i, 0), cur.get(i, 0))
            k, n = sum(cur.get(i, 0) for i in ids), len(ids)
            kb = sum(best.get(i, 0) for i in ids)
            tables.append({"attack": atk, "defense": scored_through, "step": s,
                           "k_final": k, "n": n, "rate_final": k / n,
                           "k_best": kb, "rate_best": kb / n,
                           "n_at_step": len(cur)})
        # successes gained then lost on the available grid
        prev_s = None
        for s in steps:
            if prev_s is not None:
                for i in ids:
                    if by_step[prev_s].get(i, 0) and not by_step[s].get(i, 0):
                        lost.append({"attack": atk, "image_id": i,
                                     "lost_between": [prev_s, s]})
            prev_s = s
        # decision rule: 2000 vs 500 final iterate
        if steps and 500 in by_step and max(steps) >= 2000:
            r500 = sum(by_step[500].get(i, 0) for i in ids) / len(ids)
            r2000 = sum(by_step[2000].get(i, 0) for i in ids) / len(ids)
            decision.append({"attack": atk, "rule": "final-iterate 2000 vs 500",
                             "rate_500": r500, "rate_2000": r2000,
                             "delta": r2000 - r500, "mei": MEI,
                             "budget_inadequate_for_ranking": bool(r2000 - r500 > MEI)})
        else:
            decision.append({"attack": atk, "rule": "final-iterate 2000 vs 500",
                             "note": "grid incomplete", "steps_available": steps})

    # probe arms (JPEG 50 attack, 25 images)
    ref = vec(jl(runs / CONF["jpeg50"] / "defense_results.jsonl"), "jpeg50", args.metric)
    ref = {i: ref[i] for i in probe_ids if i in ref}
    probes = []
    for rid, label in PROBES.items():
        v = vec(jl(runs / rid / "defense_results.jsonl"), "jpeg50", args.metric)
        v = {i: v[i] for i in probe_ids if i in v}
        if not v:
            probes.append({"run": rid, "label": label, "note": "not available yet"})
            continue
        probes.append({"run": rid, "label": label, "k": sum(v.values()), "n": len(v),
                       "rate": sum(v.values()) / len(v),
                       "confirmatory_same_images": {"k": sum(ref.values()), "n": len(ref),
                                                    "rate": sum(ref.values()) / len(ref) if ref else None},
                       "discordant_vs_confirmatory": sum(1 for i in v if v[i] != ref.get(i, 0))})

    (out / "success_by_step.json").write_text(json.dumps(tables, indent=1))
    (out / "decision.json").write_text(json.dumps(decision, indent=1))
    (out / "lost_successes.json").write_text(json.dumps(lost, indent=1))
    (out / "probes.json").write_text(json.dumps(probes, indent=1))
    with (out / "success_by_step.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attack", "defense", "step", "k_final", "n", "rate_final",
                    "k_best", "rate_best", "n_at_step"])
        for t in tables:
            w.writerow([t["attack"], t["defense"], t["step"], t["k_final"], t["n"],
                        f"{t['rate_final']:.4f}", t["k_best"], f"{t['rate_best']:.4f}",
                        t["n_at_step"]])

    for t in tables:
        print(f"{t['attack']:14s} {t['defense']:14s} s={t['step']:5d}  final {t['k_final']:3d}/{t['n']}"
              f" ({t['rate_final']:.3f})  best {t['k_best']:3d}/{t['n']} ({t['rate_best']:.3f})"
              f"  rows {t['n_at_step']}")
    for d in decision:
        if "delta" in d:
            print(f"decision {d['attack']}: 2000 {d['rate_2000']:.3f} vs 500 {d['rate_500']:.3f} "
                  f"delta {d['delta']:+.3f} (MEI {MEI}) "
                  f"budget_inadequate_for_ranking={d['budget_inadequate_for_ranking']}")
        else:
            print(f"decision {d['attack']}: {d.get('note', d)}")
    for p in probes:
        if "k" in p:
            print(f"probe {p['label']}: {p['k']}/{p['n']} ({p['rate']:.3f}) vs confirmatory "
                  f"{p['confirmatory_same_images']['k']}/{p['confirmatory_same_images']['n']} "
                  f"on the same images")
        else:
            print(f"probe {p['label']}: {p.get('note')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
