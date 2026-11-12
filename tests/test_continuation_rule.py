"""The pre-specified continuation-equivalence rule (scripts/validate_continuation.py)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("vc", Path(__file__).resolve().parents[1] / "scripts" / "validate_continuation.py")
vc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vc)


def row(l_u1_u2, l_u1_r, t_u1=1.0, t_u2=1.0, t_r=1.0, s=(1, 1, 1)):
    return {"image_id": "x", "linf_U1_U2": l_u1_u2, "linf_U1_R": l_u1_r,
            "loss_tail": {"U1": t_u1, "U2": t_u2, "R": t_r},
            "score": {"U1": {"D0": s[0]}, "U2": {"D0": s[1]}, "R": {"D0": s[2]}}}


def test_bitwise_deterministic_floor_requires_bitwise_continuation():
    assert vc.verdict_for(row(0.0, 0.0))["pass"]
    assert not vc.verdict_for(row(0.0, 1e-4))["pass"]


def test_nondeterministic_floor_allows_differences_up_to_the_floor():
    assert vc.verdict_for(row(0.02, 0.02))["pass"]
    assert vc.verdict_for(row(0.02, 0.0249))["pass"]
    assert not vc.verdict_for(row(0.02, 0.03))["pass"]


def test_loss_difference_beyond_tolerance_and_floor_fails():
    assert vc.verdict_for(row(0.0, 0.0, t_r=1.04))["pass"]          # 4% < 5%
    assert not vc.verdict_for(row(0.0, 0.0, t_r=1.10))["pass"]      # 10% > 5%, floor zero
    assert vc.verdict_for(row(0.02, 0.02, t_u2=1.20, t_r=1.20))["pass"]   # large floor widens tolerance


def test_success_disagreement_counts_only_when_the_floor_is_agreed():
    assert not vc.verdict_for(row(0.0, 0.0, s=(1, 1, 0)))["pass"]
    assert vc.verdict_for(row(0.0, 0.0, s=(1, 0, 0)))["pass"]       # U1 and U2 already disagree
