"""Convergence study (docs/phase3_plan.md section 2): checkpointing, optimizer variant,
manifest rules and checkpoint scoring, all on CPU."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.attacks.targeted_pgd import PGDConfig, targeted_pgd, project_linf  # noqa: E402
from src.defenses.base import Identity  # noqa: E402
from src.experiment import store  # noqa: E402
from src.experiment.aggregate import aggregate  # noqa: E402
from src.experiment.layout import RunLayout  # noqa: E402
from src.utils.resume import sanitize_id  # noqa: E402

EPS = 0.062745098
ROOT = Path(__file__).resolve().parents[1]


# --- attack: checkpoints and the optimizer variant ------------------------------------- #

def test_ckpt_roundtrip(tmp_path):
    lay = RunLayout(tmp_path / "r")
    arr = np.full((4, 4, 3), 0.25, np.float32)
    store.save_ckpt_image(lay, "n01/a b.JPEG", EPS, 600, arr)
    back = store.load_ckpt_image(lay, "n01/a b.JPEG", EPS, 600)
    assert back is not None and np.allclose(back, arr)
    assert store.load_ckpt_image(lay, "n01/a b.JPEG", EPS, 700) is None
    store.save_ckpt_image(lay, "n01/a b.JPEG", EPS, 700, arr)
    store.save_ckpt_image(lay, "n01/a b.JPEG", EPS, 1500, arr)
    assert store.list_ckpt_steps(lay, "n01/a b.JPEG", EPS) == [600, 700, 1500]


def test_ckpt_score_job_path_is_unique_per_step(tmp_path):
    lay = RunLayout(tmp_path / "r")
    p1 = store.ckpt_score_job_path(lay, "a", EPS, 600, "jpeg50")
    p2 = store.ckpt_score_job_path(lay, "a", EPS, 700, "jpeg50")
    p3 = store.ckpt_score_job_path(lay, "a", EPS, 600, "dangerzone_ll")
    assert len({p1, p2, p3}) == 3


class _TorchStubModel:
    """Quadratic loss: the gradient changes sign as x crosses 0.5, so the constant-step
    sign update and the momentum variant follow genuinely different trajectories."""
    torch = torch
    device = "cpu"

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(1, 3, 8, 8, generator=g)

    def make_image_tensor(self, image_np, requires_grad=True):
        t = torch.tensor(np.asarray(image_np, np.float32))
        t.requires_grad_(requires_grad)
        return t

    def target_loss(self, image_01, question, target):
        x = image_01.permute(2, 0, 1).unsqueeze(0) if image_01.dim() == 3 else image_01
        return (self.W * (x - 0.5) ** 2).sum() / x.numel()


def _torch_model():
    return _TorchStubModel()


def _img(seed=3):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.8, (8, 8, 3)).astype(np.float32)


def test_checkpoints_saved_every_n_and_last_equals_final():
    m = _torch_model()
    img = _img()
    cfg = PGDConfig(epsilon=EPS, steps=30, step_size=1 / 255, seed=7,
                    checkpoint_every=10, random_start=True)
    r = targeted_pgd(m, img, "q", "t", cfg)
    assert [t for t, _ in r.checkpoints] == [10, 20, 30]
    for t, arr in r.checkpoints:
        assert np.max(np.abs(arr - img)) <= cfg.epsilon + 1e-6
    assert np.allclose(r.checkpoints[-1][1], r.adv_image, atol=1e-6)


def test_momentum_changes_the_trajectory_but_stays_in_ball():
    m = _torch_model()
    img = _img()
    cfg = dict(epsilon=EPS, steps=24, step_size=1 / 255, seed=7, checkpoint_every=1)
    base = targeted_pgd(m, img, "q", "t", PGDConfig(**cfg))
    mom = targeted_pgd(m, img, "q", "t", PGDConfig(momentum=0.9, **cfg))
    # identical at step 1 (velocity = gradient there); the trajectories must diverge later
    assert np.allclose(base.checkpoints[0][1], mom.checkpoints[0][1], atol=1e-6)
    assert any(not np.allclose(a, b, atol=1e-6)
               for (_, a), (_, b) in zip(base.checkpoints, mom.checkpoints))
    assert np.max(np.abs(mom.adv_image - img)) <= EPS + 1e-6


def test_step_decay_descends_the_loss():
    m = _torch_model()
    img = _img()
    r = targeted_pgd(m, img, "q", "t",
                     PGDConfig(epsilon=EPS, steps=25, step_size=2 / 255, seed=7,
                               step_decay=True))
    assert np.max(np.abs(r.adv_image - img)) <= EPS + 1e-6
    assert r.losses[-1] < r.losses[0]


# --- manifest rules --------------------------------------------------------------------- #

def test_conv_manifests_are_deterministic_and_committed(tmp_path):
    rc = subprocess.run([sys.executable, "scripts/make_conv_manifests.py", "--check"],
                        cwd=ROOT, capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    m100 = json.loads((ROOT / "configs/conv_100.json").read_text())
    m25 = json.loads((ROOT / "configs/conv_25.json").read_text())
    ids = [s["image_id"] for s in m100["samples"]]
    assert len(ids) == 100 and len(set(ids)) == 100
    from collections import Counter
    assert set(Counter(s["wnid"] for s in m100["samples"]).values()) == {5}
    ids25 = [s["image_id"] for s in m25["samples"]]
    assert len(ids25) == 25 and set(ids25) <= set(ids)


# --- checkpoint scoring ---------------------------------------------------------------- #

class _StubModel:
    class _Cfg:
        model_id = "stub/model"
    cfg = _Cfg()

    def generate(self, image, question, max_new_tokens=8):
        return "banana"


def _fake_run(tmp_path, ids=("n01/a.JPEG", "n01/b.JPEG")):
    lay = RunLayout(tmp_path / "conv_fake")
    lay.root.mkdir(parents=True)
    lay.attack_jobs.mkdir(parents=True, exist_ok=True)
    for iid in ids:
        meta = {"image_id": iid, "class": "cat", "target": "banana", "epsilon": EPS,
                "clean_answer": "cat", "steps_cumulative": 500}
        (lay.attack_jobs / f"{sanitize_id(iid)}__eps{EPS:.4f}.json").write_text(
            json.dumps(meta))
        arr = np.full((4, 4, 3), 0.25, np.float32)
        store.save_ckpt_image(lay, iid, EPS, 600, arr)
        store.save_ckpt_image(lay, iid, EPS, 700, arr + 0.01)
    (lay.root / "config.json").write_text(json.dumps({
        "dataset": {"manifest": "configs/conv_100.json"},
        "evaluation": {"max_new_tokens": 8},
        "model": {"model_id": "stub/model"}}))
    return lay


def _sc(monkeypatch):
    import scripts.score_checkpoints as sc
    monkeypatch.setattr(sc, "build_model", lambda cfg: _StubModel())
    monkeypatch.setattr(sc, "build_defenses", lambda cfg: {"jpeg50": Identity()})
    return sc


def test_score_checkpoints_array_mode_resumes_and_aggregates(tmp_path, monkeypatch):
    sc = _sc(monkeypatch)
    lay = _fake_run(tmp_path)
    rc = sc.main(["--run", str(lay.root), "--defense", "jpeg50"])
    assert rc == 0
    jobs = sorted((lay.ckpt_score_jobs).glob("*.json"))
    assert len(jobs) == 4                       # 2 images x 2 steps
    rows = [json.loads(l) for l in lay.ckpt_results.open()]
    assert len(rows) == 4
    for r in rows:
        assert r["defense"] == "jpeg50" and r["answer"] == "banana"
        assert r["target_success"] == 1 and r["ckpt_step"] in (600, 700)
        assert not r.get("error")
    # rerun: everything skipped, no duplicate rows
    rc = sc.main(["--run", str(lay.root), "--defense", "jpeg50"])
    assert rc == 0
    assert len([json.loads(l) for l in lay.ckpt_results.open()]) == 4
    # aggregate keeps ckpt rows out of the standard tables
    summary = aggregate(lay)
    assert summary["n_ckpt_score_jobs"] == 4 and summary["n_ckpt_score_failed"] == 0
    assert summary["n_defense_jobs"] == 0


def test_score_checkpoints_sanitized_mode(tmp_path, monkeypatch):
    from src.defenses.base import array_to_pil
    sc = _sc(monkeypatch)
    lay = _fake_run(tmp_path)
    dz = tmp_path / "dz_ll"
    (dz / "ckpt").mkdir(parents=True)
    for iid in ("n01/a.JPEG", "n01/b.JPEG"):
        arr = np.full((4, 4, 3), 0.5, np.float32)
        for step in (600, 700):
            array_to_pil(arr).save(
                dz / "ckpt" / f"{sanitize_id(iid)}__eps{EPS:.4f}__s{step}.png")
    rc = sc.main(["--run", str(lay.root), "--dz-dir", str(dz)])
    assert rc == 0
    rows = [json.loads(l) for l in lay.ckpt_results.open()]
    assert len(rows) == 4
    assert {r["defense"] for r in rows} == {"dangerzone_ll"}
    assert all(r["ckpt_step"] in (600, 700) for r in rows)
    # a png whose image has no attack job is refused loudly, not silently scored
    array_to_pil(arr).save(dz / "ckpt" / f"unknown__eps{EPS:.4f}__s600.png")
    rc = sc.main(["--run", str(lay.root), "--dz-dir", str(dz)])
    assert rc == 0 and len([json.loads(l) for l in lay.ckpt_results.open()]) == 4


# --- scheduler: cumulative checkpoint labels through init-from ------------------------- #

def test_run_experiment_labels_checkpoints_cumulatively(tmp_path, monkeypatch):
    import src.experiment.run as run_mod
    from src.attacks.targeted_pgd import AttackResult

    ids = ["n01/a.JPEG"]

    class _DS:
        prompt = "q?"

        def __init__(self):
            from types import SimpleNamespace
            self._s = [SimpleNamespace(image_id=i, class_name="cat",
                                       target_class_name="banana") for i in ids]

        def __iter__(self):
            yield from self._s

        def load_image(self, s, size=8):
            return np.full((8, 8, 3), 0.4, np.float32)

    class _Model:
        class _Cfg:
            model_id = "stub/model"
        cfg = _Cfg()

        def generate(self, image, question, max_new_tokens=8):
            return "cat"

    def fake_pgd(model, clean, question, target, cfg, defense=None):
        adv = project_linf(clean + 0.01, clean, cfg.epsilon)
        return AttackResult(adv_image=adv, losses=[1.0, 0.5], epsilon=cfg.epsilon,
                            steps=cfg.steps,
                            linf=float(np.max(np.abs(adv - clean))),
                            checkpoints=[(100, adv), (200, adv)])
    monkeypatch.setattr(run_mod, "targeted_pgd", fake_pgd)

    # the earlier stage: 500 cumulative steps, stored arrays
    prev = RunLayout(tmp_path / "prev")
    prev.root.mkdir(parents=True)
    prev.attack_jobs.mkdir(parents=True, exist_ok=True)
    store.save_adv_image(prev, ids[0], EPS, np.full((8, 8, 3), 0.4, np.float32),
                         np.full((8, 8, 3), 0.41, np.float32))
    (prev.attack_jobs / f"{sanitize_id(ids[0])}__eps{EPS:.4f}.json").write_text(json.dumps(
        {"image_id": ids[0], "epsilon": EPS, "steps": 500, "steps_cumulative": 500}))

    lay = RunLayout(tmp_path / "next")
    lay.root.mkdir(parents=True)
    lay.attack_jobs.mkdir(parents=True, exist_ok=True)
    made = run_mod.run_experiment(_Model(), _DS(), {"D0": Identity()}, lay,
                                  epsilons=[EPS], pgd_steps=200, pgd_step_size=1 / 255,
                                  pgd_random_start=False, seed=1234,
                                  init_source=prev, checkpoint_every=100)
    assert made["attack"] == 1
    # cumulative labels: 500 carried over from the init run
    assert store.list_ckpt_steps(lay, ids[0], EPS) == [600, 700]
    meta = store.read_attack_meta(lay, ids[0], EPS)
    assert meta["steps_cumulative"] == 700
    assert meta["checkpoints"] == [600, 700]
    assert meta["checkpoint_every"] == 100
