from src.evaluation.staging import decide, smoothed_loss


def summ(n_success, loss, steps, n=10):
    return {"n_scored": n, "n_success": n_success, "median_defended_loss": loss,
            "cumulative_steps": steps}


def test_success_stops_first():
    assert decide(summ(8, 1.0, 500), 2.0)["reason"] == "S1 success"


def test_small_improvement_is_convergence():
    d = decide(summ(0, 2.5, 500), 2.6)         # 3.8% better
    assert d["action"] == "stop" and d["reason"] == "S2 converged"


def test_large_improvement_continues_until_the_last_budget():
    assert decide(summ(0, 1.5, 500), 2.6)["action"] == "continue"
    assert decide(summ(0, 0.5, 2000), 1.0)["reason"] == "S3 max budget"


def test_smoothing_uses_the_tail_of_the_curve():
    assert smoothed_loss({"loss_curve": [9.0] * 100 + [1.0] * 50}) == 1.0
