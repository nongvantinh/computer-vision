"""Account-pool logic for cdrctl (pure functions; no live Colab)."""
import importlib.util
import time
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "cdrctl", Path(__file__).resolve().parents[1] / "scripts" / "cdrctl.py")
cdrctl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cdrctl)


def test_parse_balance():
    assert cdrctl.parse_balance("Current balance: 12.34 compute units") == 12.34
    assert cdrctl.parse_balance("Current balance: 0.00 compute units\nrate 1.07/hr") == 0.0
    assert cdrctl.parse_balance("nothing here") is None


def test_classify_failure():
    assert cdrctl.classify_failure("... : Service Unavailable") == "retry"
    assert cdrctl.classify_failure("TooManyAssignmentsError: ...") == "retry"
    assert cdrctl.classify_failure("You are out of compute units / quota") == "rotate"
    assert cdrctl.classify_failure("401 Unauthorized invalid_grant") == "reauth"
    assert cdrctl.classify_failure("some unknown thing") == "fail"


def test_pool_roundtrip_and_next_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(cdrctl, "CDR_HOME", tmp_path)
    monkeypatch.setattr(cdrctl, "POOL", tmp_path / "pool.json")

    pool = cdrctl.load_pool()
    cdrctl.set_state(pool, "A", state=cdrctl.READY)
    cdrctl.set_state(pool, "B", state=cdrctl.EXHAUSTED,
                     cooldown_until=time.time() + 9999)
    cdrctl.save_pool(pool)

    pool = cdrctl.load_pool()                     # persisted and reloaded
    assert pool["accounts"]["A"]["state"] == cdrctl.READY
    assert cdrctl.next_ready(pool) == "A"         # READY preferred

    cdrctl.set_state(pool, "A", state=cdrctl.EXHAUSTED,
                     cooldown_until=time.time() + 9999)
    assert cdrctl.next_ready(pool) is None        # both cooling down

    cdrctl.set_state(pool, "B", cooldown_until=0)  # B cooldown elapsed
    assert cdrctl.next_ready(pool) == "B"          # recycled


def test_next_ready_empty():
    assert cdrctl.next_ready({"accounts": {}, "active": None}) is None
