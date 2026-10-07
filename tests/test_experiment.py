"""End-to-end persistence/resume test for the experiment scheduler, on CPU.

A stub model (real torch autograd, trivial loss) and stub dataset let the whole
clean -> attack -> defense pipeline run without a GPU or the real VLM, so we can
assert: jobs are written, adversarial images are persisted, a rerun skips finished
work, a failing defense is isolated, and aggregate() reports honest coverage.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from src.defenses.base import Defense, Identity
from src.experiment.layout import create_run
import src.experiment.run as run_mod
from src.experiment.run import run_experiment
from src.experiment.aggregate import aggregate
from src.experiment import store
from src.attacks.targeted_pgd import AttackResult, project_linf


@pytest.fixture(autouse=True)
def stub_attack(monkeypatch):
    """Replace the torch PGD with a numpy stub so the scheduler is testable on CPU
    without torch. The attack math itself is covered by test_pgd_constraint."""
    def fake_pgd(model, clean, question, target, cfg, defense=None):
        rng = np.random.default_rng(cfg.seed)
        noise = rng.uniform(-cfg.epsilon, cfg.epsilon, size=clean.shape).astype(np.float32)
        adv = project_linf(clean + noise, clean, cfg.epsilon)
        return AttackResult(adv_image=adv, losses=[1.0, 0.1],
                            epsilon=cfg.epsilon, steps=cfg.steps,
                            linf=float(np.max(np.abs(adv - clean))))
    monkeypatch.setattr(run_mod, "targeted_pgd", fake_pgd)


class StubModel:
    class _Cfg:
        model_id = "stub/model"
    cfg = _Cfg()
    device = "cpu"

    def generate(self, image, question, max_new_tokens=16):
        return "banana" if float(np.asarray(image).mean()) > 0.5 else "cat"


@dataclass
class StubSample:
    image_id: str
    class_name: str
    target_class_name: str


class StubDS:
    prompt = "what is this?"

    def __init__(self, n=3, size=32):
        self.size = size
        self.samples = [StubSample(f"n01/img_{i}.JPEG", "cat", "banana")
                        for i in range(n)]

    def __iter__(self):
        yield from self.samples

    def load_image(self, s, size=32):
        v = 0.3 + 0.1 * int(s.image_id.split("_")[-1].split(".")[0])
        return np.full((self.size, self.size, 3), v, dtype=np.float32)


class FailingDefense(Defense):
    name = "boom"

    def _apply(self, img):
        raise RuntimeError("simulated defense failure")


def _defenses():
    return {"D0": Identity(), "boom": FailingDefense()}


def _run(tmp_path, **kw):
    cfg = {"model": {"model_id": "stub/model"}, "defenses": ["D0", "boom"],
           "dataset": {}}
    lay = create_run("t1", cfg, {"steps": 2}, base=tmp_path)
    made = run_experiment(
        StubModel(), StubDS(n=3), _defenses(), lay,
        epsilons=[8 / 255], pgd_steps=2, pgd_step_size=1 / 255,
        pgd_random_start=True, seed=1234, image_size=32, max_new_tokens=4, **kw)
    return lay, made


def test_create_run_writes_reproducibility_artifacts(tmp_path):
    lay = create_run("r", {"model": {"model_id": "m", "dtype": "float16"},
                           "defenses": ["D0"]}, {"steps": 1}, base=tmp_path)
    for p in (lay.config, lay.environment, lay.git_commit, lay.model_info,
              lay.defense_info, lay.progress):
        assert p.exists(), p
    import json
    assert json.loads(lay.model_info.read_text())["model_id"] == "m"


def test_full_pipeline_persists_jobs_and_adv_images(tmp_path):
    lay, made = _run(tmp_path)
    assert made["attack"] == 3                      # 3 images x 1 eps
    assert made["clean"] == 6                        # 3 images x 2 defenses
    assert made["defense"] == 6                       # 3 images x 1 eps x 2 defenses
    assert len(list(lay.adv.glob("*.npy"))) == 3      # persisted adversarial images
    assert len(list(lay.adv.glob("*.png"))) == 3
    # the persisted adv image loads back and sits in the L-inf ball
    adv = store.load_adv_image(lay, "n01/img_0.JPEG", 8 / 255)
    assert adv is not None and adv.shape == (32, 32, 3)


def test_failing_defense_is_isolated_and_recorded(tmp_path):
    lay, made = _run(tmp_path)
    assert made["defense_failed"] > 0
    summary = aggregate(lay)
    assert summary["n_defense_failed"] > 0
    assert any(f["defense"] == "boom" for f in summary["failures"])
    # the healthy defense still produced clean results
    assert summary["coverage_by_defense"]["D0"]["ok"] > 0
    assert summary["coverage_by_defense"]["boom"]["failed"] > 0


def test_resume_skips_finished_jobs(tmp_path):
    lay, made1 = _run(tmp_path)
    # second run over the same run dir: everything is done, nothing new is made
    made2 = run_experiment(
        StubModel(), StubDS(n=3), _defenses(), lay,
        epsilons=[8 / 255], pgd_steps=2, pgd_step_size=1 / 255,
        pgd_random_start=True, seed=1234, image_size=32, max_new_tokens=4)
    assert made2 == {"clean": 0, "attack": 0, "defense": 0, "defense_failed": 0,
                     "missing_adv": 0, "mismatched_adv": 0}


def test_aggregate_writes_derived_tables(tmp_path):
    lay, _ = _run(tmp_path)
    summary = aggregate(lay)
    assert lay.clean_results.exists()
    assert lay.attack_results.exists()
    assert lay.defense_results.exists()
    assert summary["adv_images"] == 3
    assert summary["n_attack_jobs"] == 3


def _defense_only_run(tmp_path, source, defenses, run_id="ctl"):
    cfg = {"model": {"model_id": "stub/model"}, "defenses": list(defenses), "dataset": {}}
    lay = create_run(run_id, cfg, {"steps": 2}, base=tmp_path,
                     meta={"experiment_id": run_id, "attack_source": str(source.root)})
    made = run_experiment(
        StubModel(), StubDS(n=3), defenses, lay,
        epsilons=[8 / 255], pgd_steps=2, pgd_step_size=1 / 255,
        pgd_random_start=True, seed=1234, image_size=32, max_new_tokens=4,
        attack_source=source)
    return lay, made


def test_attack_source_reuses_adv_and_crafts_nothing(tmp_path):
    base, _ = _run(tmp_path)
    before = sorted(p.name for p in base.adv.glob("*.npy"))
    lay, made = _defense_only_run(tmp_path, base, {"D0": Identity()})
    assert made["attack"] == 0 and made["defense"] == 3
    assert not list(lay.adv.glob("*.npy"))              # nothing crafted or copied
    assert not list(lay.attack_jobs.glob("*.json"))
    assert sorted(p.name for p in base.adv.glob("*.npy")) == before   # source untouched
    # the defense rows are the same adversarial pixels: D0 answers match the baseline
    import json
    a = {json.loads(p.read_text())["image_id"]: json.loads(p.read_text())["answer"]
         for p in lay.defense_jobs.glob("*__D0.json")}
    b = {json.loads(p.read_text())["image_id"]: json.loads(p.read_text())["answer"]
         for p in base.defense_jobs.glob("*__D0.json")}
    assert a == b and len(a) == 3
    meta = json.loads(lay.config.read_text())["meta"]
    assert meta["attack_source"] == str(base.root)


def test_attack_source_missing_adv_is_counted_not_crafted(tmp_path):
    base, _ = _run(tmp_path)
    next(base.adv.glob("*.npy")).unlink()
    lay, made = _defense_only_run(tmp_path, base, {"D0": Identity()})
    assert made["missing_adv"] == 1 and made["attack"] == 0 and made["defense"] == 2


def test_attack_source_rejects_adv_that_does_not_match_the_clean_image(tmp_path):
    base, _ = _run(tmp_path)
    npy = next(base.adv.glob("*.npy"))
    np.save(npy, np.ones((32, 32, 3), dtype=np.float32))     # far outside the eps ball
    lay, made = _defense_only_run(tmp_path, base, {"D0": Identity()})
    assert made["mismatched_adv"] == 1 and made["defense"] == 2
