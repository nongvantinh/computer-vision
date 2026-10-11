"""G5 as registered (all rows, raw answers), G5a (target success, identical inputs), G5b."""
from src.evaluation import gates


def row(i, ans, succ, psnr, eps=0.0627, sha=None):
    r = {"image_id": f"c/{i}", "epsilon": eps, "defense": "D0", "answer": ans,
         "target_success": succ, "psnr_vs_clean": psnr, "error": None}
    if sha:
        r["adv_sha256"] = sha
    return r


def test_original_g5_still_fails_on_all_row_disagreement_while_g5a_scores_identical_inputs():
    base = [row(i, "fish", 0, 30.0) for i in range(100)]
    new = [row(i, "fish", 0, 30.0) for i in range(100)]
    for i in range(5):                       # different pixels: answers and success differ
        new[i] = row(i, "shih tzu", 1, 31.0)
    g5, g5a, g5b = gates.g5_reproducibility(new, base)
    assert g5["passed"] is False and abs(g5["detail"]["agreement"] - 0.95) < 1e-9
    assert g5a["passed"] is True and g5a["detail"]["n_identical_input"] == 95
    assert g5a["detail"]["n_input_differs"] == 5 and g5b["passed"] is True


def test_g5a_fails_when_identical_inputs_disagree():
    base = [row(i, "fish", 0, 30.0) for i in range(100)]
    new = [row(i, "fish", 0, 30.0) for i in range(100)]
    for i in range(3):                       # same pixels, different outcome: real nondeterminism
        new[i] = row(i, "shih tzu", 1, 30.0)
    _, g5a, g5b = gates.g5_reproducibility(new, base)
    assert g5a["passed"] is False and len(g5a["detail"]["discordant_identical_input"]) == 3
    assert g5b["passed"] is False


def test_hash_overrides_the_psnr_proxy_when_present():
    base = [row(0, "fish", 0, 30.0, sha="aa")]
    new = [row(0, "fish", 0, 30.0, sha="bb")]          # same psnr, different bytes
    _, g5a, _ = gates.g5_reproducibility(new, base)
    assert g5a["detail"]["n_identical_input"] == 0 and g5a["passed"] is False
