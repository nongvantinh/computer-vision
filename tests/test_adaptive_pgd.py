"""targeted_pgd with a defense in the loop: it must really optimize through the defense."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.attacks.targeted_pgd import PGDConfig, targeted_pgd  # noqa: E402
from src.defenses.differentiable import DifferentiableJpeg  # noqa: E402


class StubModel:
    """Loss = -<W, image>: lower is better for the attacker. W is fixed white noise."""
    torch = torch
    device = "cpu"

    def __init__(self, n=64, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(1, 3, n, n, generator=g)
        self.seen = []                      # records every image the loss was taken on

    def make_image_tensor(self, image_np, requires_grad=True):
        t = torch.tensor(np.asarray(image_np, dtype=np.float32))
        t.requires_grad_(requires_grad)
        return t

    def target_loss(self, image_01, question, target):
        x = image_01.permute(2, 0, 1).unsqueeze(0) if image_01.dim() == 3 else image_01
        self.seen.append(x.detach().clone())
        return -(self.W * x).sum() / x.numel()


def _image(n=64, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    a = np.stack([0.5 + 0.3 * np.sin(5 * xx), 0.5 + 0.3 * np.cos(4 * yy), xx * yy], -1)
    return np.clip(np.rint((a + rng.normal(0, 0.01, a.shape)) * 255), 0, 255).astype(np.float32) / 255


CFG = PGDConfig(epsilon=16 / 255, steps=30, step_size=1 / 255, random_start=True, seed=7)


def _real_loss(model, adv, jpeg):
    with torch.no_grad():
        x = torch.from_numpy(adv).permute(2, 0, 1).unsqueeze(0)
        return float(-(model.W * jpeg.real(x)).sum() / x.numel())


def test_identity_defense_reproduces_the_oblivious_attack_exactly():
    img = _image()
    a = targeted_pgd(StubModel(), img, "q", "t", CFG)
    b = targeted_pgd(StubModel(), img, "q", "t", CFG, defense=lambda x: x)
    assert np.array_equal(a.adv_image, b.adv_image)       # only the defense differs
    assert a.defense_name is None and a.defense_calls == 0
    assert b.defense_calls == CFG.steps


def test_defense_is_called_every_step_and_the_loss_is_taken_on_its_output():
    model = StubModel()
    jpeg = DifferentiableJpeg(50)
    r = targeted_pgd(model, _image(), "q", "t", CFG, defense=jpeg)
    assert r.defense_calls == CFG.steps and r.defense_name == jpeg.name
    # the loss saw JPEG output (8-bit grid after a real JPEG), not the raw float iterate
    seen = model.seen[-1]
    assert torch.allclose(seen * 255, torch.round(seen * 255), atol=1e-4)


def test_adaptive_attack_stays_inside_the_budget():
    img = _image()
    r = targeted_pgd(StubModel(), img, "q", "t", CFG, defense=DifferentiableJpeg(50))
    assert r.linf <= CFG.epsilon + 1e-6
    assert r.adv_image.min() >= 0.0 and r.adv_image.max() <= 1.0


def test_adaptive_beats_oblivious_when_scored_through_the_real_defense():
    img = _image()
    jpeg = DifferentiableJpeg(50)
    oblivious = targeted_pgd(StubModel(), img, "q", "t", CFG)
    adaptive = targeted_pgd(StubModel(), img, "q", "t", CFG, defense=jpeg)
    m = StubModel()
    l_obl = _real_loss(m, oblivious.adv_image, jpeg)
    l_ada = _real_loss(m, adaptive.adv_image, jpeg)
    l_clean = _real_loss(m, img, jpeg)
    assert l_ada < l_obl < l_clean                       # lower loss = stronger attack


def test_eot_averages_over_the_requested_number_of_calls():
    calls = []

    def noisy(x):
        calls.append(1)
        return x + 0.0 * torch.randn_like(x)

    r = targeted_pgd(StubModel(), _image(), "q", "t",
                     PGDConfig(epsilon=8 / 255, steps=5), defense=noisy, eot_samples=4)
    assert len(calls) == 5 * 4 and r.defense_calls == 20


class _GenStub(StubModel):
    class _Cfg:
        model_id = "stub/model"
    cfg = _Cfg()

    def generate(self, image, question, max_new_tokens=16):
        return "cat"


class _DS:
    prompt = "what is this?"

    class _S:
        image_id = "n01/img_0.JPEG"
        class_name = "cat"
        target_class_name = "banana"

    def __iter__(self):
        yield self._S()

    def load_image(self, s, size=64):
        return _image()


def test_run_experiment_records_defense_in_the_attack_job(tmp_path):
    """The attack job carries the evidence that the defense was in the loss path."""
    import json
    from src.defenses.base import Identity
    from src.experiment.layout import create_run
    from src.experiment.run import run_experiment

    lay = create_run("ad", {"model": {"model_id": "stub/model"}, "defenses": ["D0"],
                            "dataset": {}}, {"steps": 4}, base=tmp_path,
                     meta={"attack_id": "pgd_adaptive_jpeg50", "threat_model": "adaptive"})
    made = run_experiment(_GenStub(), _DS(), {"D0": Identity()}, lay,
                          epsilons=[8 / 255], pgd_steps=4, pgd_step_size=1 / 255,
                          pgd_random_start=True, seed=1234, image_size=64,
                          max_new_tokens=4, attack_defense=DifferentiableJpeg(50))
    assert made["attack"] == 1
    job = json.loads(next(lay.attack_jobs.glob("*.json")).read_text())
    assert job["defense_in_loop"] == "jpeg50_diff" and job["defense_calls"] == 4
    assert job["diag_grad_finite"] is True and job["diag_grad_abs_mean"] > 0
    assert job["diag_loss_raw_clean"] != job["diag_loss_defended_clean"]   # defense matters
    assert json.loads(lay.config.read_text())["meta"]["threat_model"] == "adaptive"
