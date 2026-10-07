"""Confirmatory comparisons and validity gates, on synthetic rows."""
import numpy as np
import yaml
from pathlib import Path

from src.evaluation.adaptive_compare import paired_test, run_confirmatory
from src.evaluation import gates

PLAN = yaml.safe_load((Path(__file__).resolve().parents[1]
                       / "configs" / "analysis_plan.yaml").read_text())
E8, E16 = 0.0313725490, 0.0627450980


def _rows(defense, rate, n=200, eps=(E8, E16), seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for e in eps:
        s = rng.random(n) < rate
        out += [{"image_id": f"i{i}", "epsilon": e, "defense": defense,
                 "target_success": int(s[i]), "restored": 0, "error": None} for i in range(n)]
    return out


def test_confirmatory_family_has_six_tests_and_holm():
    cond = {"ada_jpeg50": _rows("jpeg50", 0.5, seed=1), "obl_jpeg50": _rows("jpeg50", 0.0),
            "ada_dangerzone_ll": _rows("dangerzone_ll", 0.2, seed=2),
            "obl_dangerzone_ll": _rows("dangerzone_ll", 0.0)}
    dmap = {"ada_jpeg50": "jpeg50", "obl_jpeg50": "jpeg50",
            "ada_dangerzone_ll": "dangerzone_ll", "obl_dangerzone_ll": "dangerzone_ll"}
    out = run_confirmatory(PLAN, cond, dmap)
    assert out["family_size"] == 6 and not out["skipped"]
    t = {x["id"]: x for x in out["tests"]}
    big = t[f"adaptive_vs_oblivious[jpeg50]@{E16:.4f}"]
    assert big["reject_h0"] and big["diff_a_minus_b"] > 0.3 and big["p_adjusted"] >= big["p_raw"]
    cmp_ = t[f"cdr_vs_jpeg_adaptive[dangerzone_ll vs jpeg50]@{E16:.4f}"]
    assert cmp_["diff_a_minus_b"] < 0                      # Dangerzone below JPEG here


def test_missing_conditions_are_skipped_not_invented():
    out = run_confirmatory(PLAN, {"ada_jpeg50": _rows("jpeg50", 0.3)}, {"ada_jpeg50": "jpeg50"})
    assert out["family_size"] == 0 and len(out["skipped"]) == 6


def test_a_null_result_reports_whether_it_resolves_the_minimum_effect():
    same = {"ada_jpeg50": _rows("jpeg50", 0.0), "obl_jpeg50": _rows("jpeg50", 0.0)}
    out = run_confirmatory(PLAN, same, {k: "jpeg50" for k in same})
    t = out["tests"][0]
    assert not t["reject_h0"] and t["diff_ci95"][0] < 0 < t["diff_ci95"][1]
    assert t["resolves_mei"]                                # CI is inside +-0.15 at N=200


def test_pairing_uses_only_shared_images():
    va, vb = {"a": 1, "b": 0, "c": 1}, {"b": 0, "c": 0, "d": 1}
    r = paired_test(va, vb)
    assert r["n_paired"] == 2 and r["n_a_only_images"] == 1 and r["n_b_only_images"] == 1


def _job(**kw):
    base = {"image_id": "x", "epsilon": 0.0627, "linf": 0.0627, "defense_in_loop": "jpeg50",
            "defense_calls": 200, "diag_grad_finite": True, "diag_grad_abs_mean": 0.1,
            "diag_loss_raw_clean": 5.0, "diag_loss_defended_clean": 6.0}
    base.update(kw)
    return base


def test_g1_fails_when_the_defense_was_not_called_every_step():
    assert gates.g1_defense_in_loop([_job()], 200)["passed"]
    assert not gates.g1_defense_in_loop([_job(defense_calls=0)], 200)["passed"]
    assert not gates.g1_defense_in_loop([_job(defense_in_loop=None)], 200)["passed"]
    assert not gates.g1_defense_in_loop([_job(diag_loss_defended_clean=5.0)], 200)["passed"]


def test_g2_g3_g4_g5_thresholds():
    rep = {"heldout": {"clean": {"psnr_db_mean": 44.0}, "adv": {"psnr_db_mean": 43.0}}}
    assert gates.g2_surrogate_fidelity(rep)["passed"]
    rep["heldout"]["adv"]["psnr_db_mean"] = 39.0
    assert not gates.g2_surrogate_fidelity(rep)["passed"]
    ada = {i: 1.0 for i in range(10)}
    obl = {i: 2.0 for i in range(8)} | {8: 0.5, 9: 0.5}
    assert gates.g3_attack_optimizes(ada, obl)["passed"]        # 8/10 = 0.8
    obl[7] = 0.5
    assert not gates.g3_attack_optimizes(ada, obl)["passed"]    # 7/10
    jobs = [_job(image_id=str(i)) for i in range(4)]
    assert gates.g4_no_silent_loss(jobs, [], 2, 2)["passed"]
    assert not gates.g4_no_silent_loss(jobs[:3], [], 2, 2)["passed"]
    assert not gates.g4_no_silent_loss([_job(linf=0.2)], [], 1, 1)["passed"]
    assert not gates.g4_no_silent_loss(jobs, [{"error": "x"}], 2, 2)["passed"]
    rows = [{"image_id": str(i), "epsilon": 0.0627, "defense": "D0", "answer": "cat"} for i in range(100)]
    assert gates.g5_baseline_reproduction(rows, rows)["passed"]
    changed = [dict(r, answer="dog") if i < 3 else r for i, r in enumerate(rows)]
    assert not gates.g5_baseline_reproduction(changed, rows)["passed"]
