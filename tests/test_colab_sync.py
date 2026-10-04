"""Cross-account sync logic: merge-missing, jsonl merge, lock staleness, store."""
import json
import time

from src.colab.sync import (merge_missing, merge_jsonl, lock_is_stale,
                            StateStore, count_done_jobs)


def test_merge_missing_only_adds(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "a").mkdir(parents=True)
    (src / "a" / "1.json").write_text('{"v":1}')
    (src / "a" / "2.json").write_text('{"v":2}')
    (dst / "a").mkdir(parents=True)
    (dst / "a" / "2.json").write_text('{"v":"KEEP"}')   # pre-existing, must not change
    added = merge_missing(src, dst)
    assert added == 1
    assert (dst / "a" / "1.json").exists()
    assert json.loads((dst / "a" / "2.json").read_text())["v"] == "KEEP"


def test_merge_missing_is_idempotent(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src).mkdir(); (src / "x.json").write_text("{}")
    assert merge_missing(src, dst) == 1
    assert merge_missing(src, dst) == 0        # second run adds nothing


def test_merge_missing_respects_skip(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "session.lock").write_text("{}")
    (src / "keep.json").write_text("{}")
    added = merge_missing(src, dst, skip=("session.lock",))
    assert added == 1
    assert not (dst / "session.lock").exists()


def test_merge_jsonl_dedups_and_appends(tmp_path):
    src, dst = tmp_path / "s.jsonl", tmp_path / "d.jsonl"
    dst.write_text('{"session":"A"}\n')
    src.write_text('{"session":"A"}\n{"session":"B"}\n')   # A dup, B new
    added = merge_jsonl(src, dst)
    assert added == 1
    lines = dst.read_text().splitlines()
    assert len(lines) == 2


def test_lock_staleness():
    assert lock_is_stale(None) is True
    fresh = {"heartbeat": time.time()}
    stale = {"heartbeat": time.time() - 30 * 60}
    assert lock_is_stale(fresh, stale_minutes=20) is False
    assert lock_is_stale(stale, stale_minutes=20) is True


def test_state_store_lock_and_history(tmp_path):
    s = StateStore(tmp_path / "state")
    s.acquire_lock("acct1-001", "acct1")
    # a different live session cannot steal the lock
    try:
        s.acquire_lock("acct2-002", "acct2")
        assert False, "expected lock contention"
    except RuntimeError:
        pass
    # a stale lock can be reclaimed
    s.write("session.lock", {"session": "acct1-001", "account": "acct1",
                             "heartbeat": time.time() - 3600})
    s.acquire_lock("acct2-002", "acct2")       # no raise
    s.append("sessions.jsonl", {"session": "acct1-001", "account": "acct1"})
    s.append("sessions.jsonl", {"session": "acct1-001", "account": "acct1"})  # dup id
    s.append("sessions.jsonl", {"session": "acct2-002", "account": "acct2"})
    assert len(s.history()) == 2               # deduped by session id


def test_count_done_jobs(tmp_path):
    (tmp_path / "jobs" / "clean").mkdir(parents=True)
    (tmp_path / "jobs" / "attack").mkdir(parents=True)
    (tmp_path / "jobs" / "defense").mkdir(parents=True)
    (tmp_path / "jobs" / "clean" / "a__D0.json").write_text("{}")
    (tmp_path / "jobs" / "attack" / "a__eps0.0314.json").write_text("{}")
    (tmp_path / "jobs" / "attack" / "a__eps0.0627.json").write_text("{}")
    (tmp_path / "jobs" / "defense" / "a__eps0.0314__D0.json").write_text("{}")
    assert count_done_jobs(tmp_path) == {"clean": 1, "attack": 2, "defense": 1}


def test_restore_filters_pull_adv_arrays_but_light_filters_do_not(tmp_path):
    """A restore that skips adv/*.npy makes run.py redo every finished attack."""
    import shutil
    import subprocess
    from src.colab.sync import LIGHT_FILTERS, RESTORE_FILTERS
    if shutil.which("rsync") is None:
        import pytest
        pytest.skip("rsync not installed")
    src, light, full = tmp_path / "src", tmp_path / "light", tmp_path / "full"
    (src / "adv").mkdir(parents=True)
    (src / "jobs").mkdir()
    (src / "adv" / "a.npy").write_bytes(b"x")
    (src / "jobs" / "a.json").write_text("{}")
    (src / "model.bin").write_bytes(b"x")
    for dst, filt in ((light, LIGHT_FILTERS), (full, RESTORE_FILTERS)):
        subprocess.run(["rsync", "-a", *filt, f"{src}/", f"{dst}/"], check=True)
    assert not (light / "adv" / "a.npy").exists()
    assert (full / "adv" / "a.npy").exists() and (full / "jobs" / "a.json").exists()
    assert not (full / "model.bin").exists()
