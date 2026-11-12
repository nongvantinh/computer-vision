#!/usr/bin/env python
"""Record or verify SHA-256 checksums of raw experimental data.

    python scripts/checksum_runs.py record --runs results/runs --out docs/provenance/raw_checksums.json
    python scripts/checksum_runs.py verify --manifest docs/provenance/raw_checksums.json --runs results/runs

Covers the raw result files of each run directory (config, dataset manifest, the jsonl result
tables, the job files and the adversarial arrays). Verification is read-only and reports
missing, changed and new files. Intended to detect accidental modification of historical runs.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

RAW = ("config.json", "dataset_manifest.json", "attack_results.jsonl", "defense_results.jsonl",
       "clean_results.jsonl", "defense_info.json")
RAW_DIRS = ("jobs", "adv")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files_of(run: Path):
    for n in RAW:
        if (run / n).is_file():
            yield n, run / n
    for d in RAW_DIRS:
        for p in sorted((run / d).rglob("*")) if (run / d).is_dir() else []:
            if p.is_file():
                yield str(p.relative_to(run)), p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "verify"])
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="docs/provenance/raw_checksums.json")
    ap.add_argument("--manifest", default="docs/provenance/raw_checksums.json")
    ap.add_argument("--only", nargs="*", default=None, help="run ids; default all run dirs")
    a = ap.parse_args()
    runs = sorted(p for p in Path(a.runs).iterdir() if p.is_dir() and (a.only is None or p.name in a.only))
    if a.mode == "record":
        out = {r.name: {n: sha(p) for n, p in files_of(r)} for r in runs}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
        print("recorded", {k: len(v) for k, v in out.items()})
        return 0
    want = json.loads(Path(a.manifest).read_text())
    bad = 0
    for r in runs:
        if r.name not in want:
            print("NOT RECORDED", r.name)
            continue
        have = {n: sha(p) for n, p in files_of(r)}
        miss = sorted(set(want[r.name]) - set(have))
        chg = sorted(n for n in have if n in want[r.name] and have[n] != want[r.name][n])
        new = sorted(set(have) - set(want[r.name]))
        status = "OK" if not (miss or chg) else "CHANGED"
        bad += status != "OK"
        print(f"{status:8s} {r.name}  missing={len(miss)} changed={len(chg)} new={len(new)}")
        for n in chg[:5]:
            print("   changed:", n)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
