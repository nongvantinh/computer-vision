"""Differentiable LLaVA-1.5-7B wrapper for white-box attacks (proposal §3.5).

Requirement: gradients must flow to the raw input image, so CLIP normalization is
applied *inside* the differentiable path. We feed a float image in [0, 1] at the
model's native 336x336 and compute pixel_values ourselves; the processor is used
only to build the token sequence (with expanded <image> tokens).

No CLIP surrogate: the loss is the teacher-forcing cross-entropy of the full VLM
toward a target answer, and the attack differentiates through the whole model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# CLIP ViT-L/14-336 normalization (LLaVA-1.5 vision tower)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_SIZE = 336


@dataclass
class LlavaConfig:
    model_id: str = "llava-hf/llava-1.5-7b-hf"
    device: str = "cuda"
    dtype: str = "float16"
    gradient_checkpointing: bool = True


class LlavaWrapper:
    def __init__(self, cfg: LlavaConfig | None = None):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.cfg = cfg or LlavaConfig()
        self.torch = torch
        dtype = getattr(torch, self.cfg.dtype)
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.cfg.model_id, torch_dtype=dtype, low_cpu_mem_usage=True)
        self.model.to(self.cfg.device).eval()
        self.model.requires_grad_(False)          # freeze weights; grad -> input only
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.dtype = dtype
        self.device = self.cfg.device
        self._mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)

    # ---- image -> normalized pixel_values (differentiable) ------------------ #
    def to_pixel_values(self, image_01):
        """image_01: (H,W,3) or (1,3,H,W) tensor in [0,1] -> normalized pixel_values."""
        torch = self.torch
        x = image_01
        if x.dim() == 3:                      # HWC -> CHW
            x = x.permute(2, 0, 1).unsqueeze(0)
        x = x.to(self.device)
        if x.shape[-1] != IMAGE_SIZE or x.shape[-2] != IMAGE_SIZE:
            x = torch.nn.functional.interpolate(
                x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", align_corners=False)
        x = (x - self._mean) / self._std
        return x.to(self.dtype)

    def make_image_tensor(self, image_np: np.ndarray, requires_grad: bool = True):
        """Wrap a numpy [0,1] image as a leaf tensor for the attack."""
        torch = self.torch
        t = torch.tensor(np.asarray(image_np, dtype=np.float32), device=self.device)
        t.requires_grad_(requires_grad)
        return t

    # ---- prompt / label construction --------------------------------------- #
    def _prompt_ids(self, question: str):
        """Build input_ids (with expanded image tokens) for the USER turn."""
        from PIL import Image
        conv = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}]}]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True)
        dummy = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        enc = self.processor(images=dummy, text=text, return_tensors="pt")
        return enc  # has input_ids, attention_mask, pixel_values (we replace pv)

    # ---- teacher-forcing loss toward a target answer ----------------------- #
    def target_loss(self, image_01, question: str, target_answer: str):
        """Cross-entropy of the VLM on `target_answer` tokens given the image."""
        torch = self.torch
        enc = self._prompt_ids(question)
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        tgt = self.processor.tokenizer(
            " " + target_answer.strip(), add_special_tokens=False,
            return_tensors="pt")["input_ids"].to(self.device)

        full_ids = torch.cat([input_ids, tgt], dim=1)
        full_attn = torch.cat([attn, torch.ones_like(tgt)], dim=1)
        labels = torch.cat(
            [torch.full_like(input_ids, -100), tgt], dim=1)

        pixel_values = self.to_pixel_values(image_01)
        out = self.model(input_ids=full_ids, attention_mask=full_attn,
                         pixel_values=pixel_values, labels=labels)
        return out.loss

    # ---- free-form generation for evaluation ------------------------------- #
    @property
    def no_grad(self):
        return self.torch.no_grad

    def generate(self, image_01, question: str, max_new_tokens: int = 16) -> str:
        torch = self.torch
        enc = self._prompt_ids(question)
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        if not torch.is_tensor(image_01):
            image_01 = self.make_image_tensor(image_01, requires_grad=False)
        with torch.no_grad():
            pixel_values = self.to_pixel_values(image_01)
            gen = self.model.generate(
                input_ids=input_ids, attention_mask=attn,
                pixel_values=pixel_values, do_sample=False,
                num_beams=1, max_new_tokens=max_new_tokens)
        text = self.processor.tokenizer.decode(
            gen[0, input_ids.shape[1]:], skip_special_tokens=True)
        return text.strip()
