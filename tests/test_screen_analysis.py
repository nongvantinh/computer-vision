from src.evaluation import screen_analysis as sa


def rows(defense, eps, successes):
    return [{"image_id": f"c/{i}", "epsilon": eps, "defense": defense,
             "target_success": int(s), "error": None} for i, s in enumerate(successes)]


EPS = 0.0627450980
IDS = {f"c/{i}" for i in range(20)}


def test_adapters_stay_separate_and_rates_have_exact_intervals():
    non = rows("dangerzone_ll", EPS, [0] * 20) + rows("dangerzone", EPS, [0] * 20)
    ada = {"dangerzone_ll": rows("dangerzone_ll", EPS, [1] * 16 + [0] * 4),
           "dangerzone": rows("dangerzone", EPS, [1] * 8 + [0] * 12)}
    table = sa.rate_rows(non, ada, [EPS], IDS)
    by = {(r["attack"], r["scored_through"]): r for r in table}
    assert by[("adaptive_dangerzone_ll", "dangerzone_ll")]["k"] == 16
    assert by[("adaptive_dangerzone", "dangerzone")]["k"] == 8
    assert 0 < by[("adaptive_dangerzone_ll", "dangerzone_ll")]["ci95"][0] < 0.8


def test_family_is_holm_corrected_and_labelled_exploratory():
    non = (rows("jpeg50", EPS, [0] * 20) + rows("dangerzone_ll", EPS, [0] * 20))
    ada = {"jpeg50": rows("jpeg50", EPS, [0] * 20),
           "dangerzone_ll": rows("dangerzone_ll", EPS, [1] * 16 + [0] * 4)}
    fam = sa.exploratory_family(non, ada, [EPS], IDS)
    assert fam["status"] == "exploratory" and fam["family_size"] == 3
    t = {x["id"]: x for x in fam["tests"]}
    adapt = t[f"adaptive_vs_non_adaptive[dangerzone_ll]@{EPS:.4f}"]
    assert adapt["diff_a_minus_b"] == 0.8 and adapt["reject_h0"] is True
    assert t[f"adaptive_vs_non_adaptive[jpeg50]@{EPS:.4f}"]["reject_h0"] is False
    assert all(0 <= v <= 1 for v in sa.discordant_rate(fam["tests"]).values())
