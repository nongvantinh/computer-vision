#!/usr/bin/env python
"""Loss of stored adversarial images THROUGH a defense (input to gate G3). GPU, no backward.

    python scripts/eval_defended_loss.py --adaptive-run runs/adaptive_jpeg50 \
        --oblivious-run runs/full_200 --defense jpeg50 --out defended_loss_jpeg50.json

For JPEG the defense forward is the real Pillow output; for Dangerzone it is the
fitted surrogate (so G2, the surrogate's fidelity, must also pass).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.defenses.differentiable import build_adaptive_defense  # noqa: E402
from src.evaluation.runner import build_model, build_dataset  # noqa: E402
from src.experiment.layout import RunLayout  # noqa: E402
from src.experiment import store  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, seed_everything  # noqa: E402

log = get_logger("defended_loss")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptive-run", required=True)
    ap.add_argument("--oblivious-run", required=True)
    ap.add_argument("--defense", required=True)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    ada, obl = RunLayout(Path(args.adaptive_run)), RunLayout(Path(args.oblivious_run))
    model, ds = build_model(cfg), build_dataset(cfg)
    defense = build_adaptive_defense(args.defense)
    torch = model.torch
    out = {"adaptive": {}, "oblivious": {}, "defense": args.defense}
    for s in ds:
        for ajp in sorted(ada.attack_jobs.glob(f"{store.sanitize_id(s.image_id)}__eps*.json")):
            job = json.loads(ajp.read_text())
            eps = job["epsilon"]
            key = f"{s.image_id}@{eps:.4f}"
            for tag, lay in (("adaptive", ada), ("oblivious", obl)):
                adv = store.load_adv_image(lay, s.image_id, eps)
                if adv is None:
                    continue
                with torch.no_grad():
                    x = model.make_image_tensor(adv, requires_grad=False)
                    xd = defense(x.permute(2, 0, 1).unsqueeze(0))
                    out[tag][key] = float(model.target_loss(xd, ds.prompt,
                                                            s.target_class_name).cpu())
    Path(args.out).write_text(json.dumps(out, indent=1))
    log.info("wrote %s (%d adaptive, %d oblivious)", args.out,
             len(out["adaptive"]), len(out["oblivious"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
