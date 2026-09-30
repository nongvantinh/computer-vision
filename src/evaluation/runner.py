"""Builders for the experiment: model, dataset, defenses.

The resumable orchestration lives in `src/experiment/` (layout, store, run,
aggregate). This module only constructs the pieces from a config dict, so both the
CLI scripts and the scheduler share one way of building things.

Metric design (model-relative, label-free): a general VLM will not emit a
fine-grained ImageNet class name for a clean image, so accuracy against the
ImageNet label is not meaningful. We use the model's own clean answer as the
reference:
  - preserved:  a defense on the CLEAN image keeps the model's clean answer.
  - restored:   a defense on the ADVERSARIAL image returns the answer to the clean
                answer (the defense undid the attack).
  - targeted success: the (defended) adversarial answer contains the target label.
"""
from __future__ import annotations

from ..defenses import build_defense
from ..datasets.imagenet_vqa import ImageNetVQA
from ..utils.logging import get_logger

log = get_logger("runner")


def build_model(cfg: dict):
    from ..models.llava import LlavaWrapper, LlavaConfig
    m = cfg["model"]
    return LlavaWrapper(LlavaConfig(
        model_id=m["model_id"], device=m["device"], dtype=m["dtype"],
        gradient_checkpointing=m.get("gradient_checkpointing", True),
        load_in_4bit=m.get("load_in_4bit", False)))


def build_defenses(cfg: dict) -> dict:
    """Build the configured defenses, skipping any that cannot be constructed.

    A defense whose external tool is unconfigured or blocked (e.g. ICDR without a
    licensed command) is logged and left out rather than crashing the run. The set
    of unavailable defenses is recorded on `build_defenses.unavailable` so the
    coverage summary stays honest; we never substitute a homemade filter for a
    missing real tool.
    """
    tools = cfg.get("tools", {})
    out, unavailable = {}, {}
    for name in cfg["defenses"]:
        kwargs = {}
        if name == "icdr":
            kwargs["cmd"] = tools.get("icdr_cmd")
        elif name == "dangerzone":
            kwargs["cli"] = tools.get("dangerzone_cmd", "dangerzone-cli")
        try:
            out[name] = build_defense(name, **kwargs)
        except Exception as e:
            unavailable[name] = repr(e)
            log.warning("defense %r unavailable, skipping: %s", name, e)
    if unavailable:
        log.warning("unavailable defenses (not run): %s", list(unavailable))
    build_defenses.unavailable = unavailable
    return out


def build_dataset(cfg: dict) -> ImageNetVQA:
    d = cfg["dataset"]
    return ImageNetVQA(d["manifest"], d["imagenet_val_dir"])
