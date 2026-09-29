"""Turn a VLM free-text answer into a class label prediction."""
from __future__ import annotations

import numpy as np

from .metrics import match_label


def predict(model, image_np: np.ndarray, prompt: str, class_names,
            max_new_tokens: int = 16) -> tuple[str, str | None]:
    """Return (raw_answer, matched_class_name_or_None)."""
    raw = model.generate(image_np, prompt, max_new_tokens=max_new_tokens)
    return raw, match_label(raw, class_names)
