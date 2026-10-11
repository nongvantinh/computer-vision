#!/usr/bin/env python
"""Tables, tests and figures for the 50-image screen. Everything is exploratory.

    python scripts/analyze_screen.py --runs results/runs --controls results/runs/controls_v1 \
        --manifest configs/screen_50.json --out results/screen

Reads the screen run directories (screen_dz_ll_e16, screen_jpeg50_e16, screen_dz_e16 and the
e08 counterparts), takes the non-adaptive condition from `controls_v1` restricted to the
screen images, and writes rates.json/csv, tests.json, utility.json, runtime.json and figures.
Missing runs are skipped and listed, so a partial screen can be analysed and labelled as such.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import screen_analysis as sa  # noqa: E402
from src.experiment.aggregate import aggregate  # noqa: E402
from src.experiment.layout import RunLayout  # noqa: E402

EPS = {"e16": 0.0627450980, "e08": 0.0313725490}
RUN_ATTACK = {"screen_dz_ll_": "dangerzone_ll", "screen_jpeg50_": "jpeg50", "screen_dz_": "dangerzone"}


def load(run: Path, name: str) -> list[dict]:
    lay = RunLayout(run)
    p = {"defense": lay.defense_results, "clean": lay.clean_results}[name]
    if not p.exists() and (lay.defense_jobs.exists() and any(lay.defense_jobs.glob("*.json"))):
        aggregate(lay)           # derived tables are only built when missing, never rewritten
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


def attack_of(run_id: str):
    for pre, atk in RUN_ATTACK.items():       # longest prefix first: dz_ll before dz
        if run_id.startswith(pre):
            return atk, run_id[len(pre):]
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--controls", default="results/runs/controls_v1")
    ap.add_argument("--manifest", default="configs/screen_50.json")
    ap.add_argument("--out", default="results/screen")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = {s["image_id"] for s in json.loads(Path(a.manifest).read_text())["samples"]}
    non = load(Path(a.controls), "defense")
    clean = load(Path(a.controls), "clean")

    adaptive, curves, jobs_all, found = {}, {}, [], []
    for run in sorted(Path(a.runs).glob("screen_*")):
        atk, tag = attack_of(run.name)
        if atk is None or tag not in EPS:
            continue
        rows = load(run, "defense")
        if not rows:
            continue
        found.append(run.name)
        adaptive.setdefault(atk, []).extend(rows)
        for jp in sorted((run / "jobs" / "attack").glob("*.json")):
            j = json.loads(jp.read_text())
            jobs_all.append({**j, "attack": atk})
            if j.get("loss_curve"):
                curves.setdefault((atk, tag), []).append(j["loss_curve"])
    eps_list = sorted({EPS[attack_of(r)[1]] for r in found}, reverse=True)
    missing = sorted(set(f"{k}{t}" for k in RUN_ATTACK for t in EPS) - set(found))

    rates = sa.rate_rows(non, adaptive, eps_list, ids)
    fam = sa.exploratory_family(non, adaptive, eps_list, ids)
    fam["discordant_rate"] = sa.discordant_rate(fam["tests"])
    util = {"preservation_clean": sa.preservation(clean, ids),
            "n_screen_images": len(ids)}
    runtime = {atk: sa.runtime_summary([j for j in jobs_all if j["attack"] == atk])
               for atk in adaptive}
    (out / "rates.json").write_text(json.dumps(rates, indent=1))
    (out / "tests.json").write_text(json.dumps(fam, indent=1))
    (out / "utility.json").write_text(json.dumps(util, indent=1))
    (out / "runtime.json").write_text(json.dumps(runtime, indent=1))
    (out / "status.json").write_text(json.dumps({"runs_found": found, "runs_missing": missing,
                                                  "complete": not missing}, indent=1))
    with (out / "rates.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attack", "epsilon", "scored_through", "k", "n", "rate", "ci_lo", "ci_hi"])
        for r in rates:
            w.writerow([r["attack"], f"{r['epsilon']:.4f}", r["scored_through"], r["k"], r["n"],
                        f"{r['rate']:.4f}", f"{r['ci95'][0]:.4f}", f"{r['ci95'][1]:.4f}"])

    colors = {"non_adaptive": "#4c78a8", "adaptive_jpeg50": "#f58518",
              "adaptive_dangerzone_ll": "#e45756", "adaptive_dangerzone": "#b279a2"}
    for eps in eps_list:
        sel = [r for r in rates if abs(r["epsilon"] - eps) < 1e-6]
        defs = [d for d in ("D0", "jpeg90", "jpeg75", "jpeg50", "dangerzone_ll", "dangerzone")
                if any(r["scored_through"] == d for r in sel)]
        atks = [k for k in colors if any(r["attack"] == k for r in sel)]
        fig, ax = plt.subplots(figsize=(10, 3.8))
        wbar = 0.8 / max(len(atks), 1)
        for ai, atk in enumerate(atks):
            for di, d in enumerate(defs):
                r = next((x for x in sel if x["attack"] == atk and x["scored_through"] == d), None)
                if r is None:
                    continue
                x = di + (ai - (len(atks) - 1) / 2) * wbar
                ax.bar(x, r["rate"], wbar * 0.92, color=colors[atk], label=atk if di == 0 or d == defs[0] else None)
                ax.errorbar(x, r["rate"], yerr=[[r["rate"] - r["ci95"][0]], [r["ci95"][1] - r["rate"]]],
                            color="k", capsize=1.5, lw=0.8)
                ax.text(x, min(r["ci95"][1], 1) + 0.015, f"{r['k']}/{r['n']}", ha="center", fontsize=6)
        ax.set_xticks(range(len(defs)), [sa.DEFENSE_LABEL[d] for d in defs], fontsize=7, rotation=12)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("targeted success (95% exact CI)")
        ax.set_title(f"Screen, 50 images, epsilon {eps * 255:.0f}/255 (exploratory)", fontsize=9)
        h, l = ax.get_legend_handles_labels()
        ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"screen_asr_e{round(eps * 255):02d}.png", dpi=160)
        plt.close(fig)
    if curves:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for (atk, tag), cs in sorted(curves.items()):
            n = min(map(len, cs))
            m = np.median(np.array([c[:n] for c in cs]), axis=0)
            k = 10
            sm = np.convolve(m, np.ones(k) / k, mode="valid")
            ax.plot(range(len(sm)), sm, label=f"{sa.ATTACK_LABEL[atk]} {tag}", lw=1)
        ax.set_xlabel("step")
        ax.set_ylabel("target loss through the defense (median, 10-step mean)")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / "screen_loss_curves.png", dpi=160)
        plt.close(fig)
    print("wrote", out, "| runs found:", found, "| missing:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
