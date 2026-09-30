"""Cross-session / cross-account state logic for Colab runs.

Adapted from the knowledge-tracing-comparison Colab notebook. The pure functions
here (merge, filters, lock staleness, state store) are unit-tested; the notebook
handles Drive mount and rsync and calls into these.

Design (see docs/implementation/colab-resume.md):
- Durable store lives on Google Drive. State travels two ways, merged never
  overwritten: (A) a shared Drive folder across accounts, (B) a light state zip.
- Merges only ADD missing files, so restoring from Drive then from a zip is safe
  and order-independent; a finished job's result file is immutable.
- `sessions.jsonl` is append-only and deduped by session id.
- `session.lock` is advisory with a stale-heartbeat timeout.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

# rsync include/exclude filters that pull only the LIGHT state back from Drive
# (per-job JSON, derived tables, plots, manifest, configs) and never the heavy
# example PNGs or model/data caches. First matching rule wins, so "*/" must lead
# for rsync to descend into subdirectories.
LIGHT_FILTERS = [
    "--include=*/",
    "--include=*.json",
    "--include=*.jsonl",
    "--include=*.csv",
    "--include=*.yaml",
    "--include=*.png",       # plots + example sheets are small; keep them
    "--exclude=*",
]

# Drive subfolders (one asset class each, so they can be shared/wiped separately).
DRIVE_TREE = ("results", "state", "hf_cache", "imagenet", "reports")


def merge_missing(source: Path, target: Path, skip: tuple[str, ...] = ()) -> int:
    """Copy files that exist under `source` but not `target`. Never overwrites.

    Returns the number of files added. Safe to run repeatedly and from multiple
    sources; a finished job's file is immutable so sources cannot conflict.
    """
    source, target = Path(source), Path(target)
    added = 0
    if not source.exists():
        return 0
    for path in source.rglob("*"):
        if path.is_dir() or path.name in skip:
            continue
        dest = target / path.relative_to(source)
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        added += 1
    return added


def merge_jsonl(source: Path, target: Path) -> int:
    """Append lines from `source` not already present in `target` (dedup, in order)."""
    source, target = Path(source), Path(target)
    if not source.exists():
        return 0
    seen = set(target.read_text().splitlines()) if target.exists() else set()
    fresh = [ln for ln in source.read_text().splitlines() if ln.strip() and ln not in seen]
    if fresh:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as f:
            f.write("\n".join(fresh) + "\n")
    return len(fresh)


def lock_is_stale(lock: dict | None, stale_minutes: float = 20.0,
                  now: float | None = None) -> bool:
    """A session lock is reclaimable if missing or its heartbeat is old."""
    if not lock:
        return True
    now = time.time() if now is None else now
    return (now - float(lock.get("heartbeat", 0))) / 60.0 >= stale_minutes


class StateStore:
    """JSON/JSONL state under Drive/state (profile.json, sessions.jsonl, session.lock)."""

    def __init__(self, state_dir: str | Path):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def read(self, name: str, default=None):
        p = self.dir / name
        if not p.exists():
            return default
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return default

    def write(self, name: str, payload) -> None:
        (self.dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def append(self, name: str, payload) -> None:
        with (self.dir / name).open("a") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def history(self) -> list[dict]:
        """Sessions log, deduped by session id (last write wins)."""
        p = self.dir / "sessions.jsonl"
        if not p.exists():
            return []
        rows: dict[str, dict] = {}
        for line in p.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[row.get("session", line)] = row
        return list(rows.values())

    # --- advisory session lock -------------------------------------------- #
    def acquire_lock(self, session_id: str, account: str, *,
                     stale_minutes: float = 20.0, force: bool = False) -> None:
        holder = self.read("session.lock")
        if holder and holder.get("session") != session_id and not force:
            if not lock_is_stale(holder, stale_minutes):
                idle = (time.time() - float(holder.get("heartbeat", 0))) / 60
                raise RuntimeError(
                    f"store is locked by {holder.get('session')} "
                    f"(account {holder.get('account')}), heartbeat {idle:.1f} min ago. "
                    "Close that session or set force=True if it is dead.")
        self.heartbeat(session_id, account)

    def heartbeat(self, session_id: str, account: str, **extra) -> None:
        payload = {"session": session_id, "account": account,
                   "heartbeat": time.time()}
        payload.update(extra)
        self.write("session.lock", payload)

    def release_lock(self) -> None:
        (self.dir / "session.lock").unlink(missing_ok=True)


def count_done_jobs(results_dir: str | Path) -> dict:
    """How many job files exist — the true 'what is finished' count.

    Matches the run-directory layout: jobs/{clean,attack,defense}. `attack` counts
    the expensive adversarial-image jobs; `defense` the per-defense evaluations.
    """
    r = Path(results_dir)
    return {"clean": len(list((r / "jobs" / "clean").glob("*.json"))),
            "attack": len(list((r / "jobs" / "attack").glob("*.json"))),
            "defense": len(list((r / "jobs" / "defense").glob("*.json")))}
