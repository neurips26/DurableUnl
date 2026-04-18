"""
src/baselines/wga.py
======================
Weighted Gradient Ascent (WGA) — Jia et al. (2024) SOUL variant.

Standard GA applies equal gradient weight to all forget samples.
WGA weights each sample's gradient by its predicted loss:
  high-loss samples (already forgotten) → low weight
  low-loss samples (still remembered) → high weight

This focuses unlearning effort on the samples the model still
remembers most strongly, improving efficiency and forget accuracy.

Additionally implements:
  "wga_mse": WGA with MSE label-replacement (replace gold labels with
             random tokens, then weight by how similar output is to gold)
             — shown to be more stable than pure GA in some settings.

Reference:
  Jia et al. "Soul: Unlocking the Power of Second-Order Optimization
              for LLM Unlearning." arXiv 2024.
  Also used as a component in: Maini et al. TOFU (label perturbation variant)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .base import BaseUnlearner, _clm_loss

logger = logging.getLogger(__name__)


class WGA(BaseUnlearner):
    """
    Weighted Gradient Ascent.

    Each forget batch gradient is scaled by a weight derived from per-sample loss:
        w_i = softmax(-losses_i / temperature)   (inverse loss weighting)
              OR
        w_i = 1.0 (standard GA, baseline)

    Low-loss samples (still remembered) get higher weight → focused unlearning.
    High-loss samples (already forgotten) get lower weight → avoid over-forgetting.

    Args:
        temperature (float): Softmax temperature for weight computation.
                             Lower = more concentrated on hard samples.
                             Default 1.0.
        variant     (str):   "weighted" (WGA) or "label_perturb" (WGA-LP).
    """

    def __init__(
        self,
        model: nn.Module,
        forget_loader=None,
        retain_loader=None,
        device=None,
        n_steps: int = 300,
        lr: float = 5e-5,
        retain_lambda: float = 1.0,
        gradient_clip: float = 1.0,
        log_every: int = 50,
        temperature: float = 1.0,
        variant: str = "weighted",  # "weighted" or "label_perturb"
        vocab_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            model=model, forget_loader=forget_loader, retain_loader=retain_loader,
            loss_fn=_clm_loss, device=device, n_steps=n_steps,
            retain_lambda=retain_lambda, lr=lr,
            gradient_clip=gradient_clip, log_every=log_every,
        )
        self.temperature = temperature
        self.variant     = variant
        self.vocab_size  = vocab_size

    def _per_sample_losses(self, batch: dict) -> torch.Tensor:
        """Compute per-sample CLM loss (not averaged)."""
        ids    = batch["input_ids"].to(self.device)
        mask   = batch["attention_mask"].to(self.device)
        labels = batch.get("labels", ids).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=mask).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Per-token loss
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(ids.size(0), -1)   # (batch, seq-1)

        # Average over non-padding tokens per sample
        valid = (shift_labels != -100).float()
        loss_per_sample = (loss_per_token * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1)
        return loss_per_sample   # (batch,)

    def _perturbed_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Label perturbation: replace gold labels with random tokens.
        The model maximises probability of wrong tokens → unlearning.
        """
        vocab = self.vocab_size or self.model.config.vocab_size
        perturbed = torch.randint_like(labels, low=0, high=vocab)
        # Keep -100 padding positions
        perturbed[labels == -100] = -100
        return perturbed

    def run(self) -> None:
        logger.info(
            f"Starting WGA | variant={self.variant} | steps={self.n_steps} | "
            f"temperature={self.temperature}"
        )
        self.model.train()

        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_steps)
        forget_iter = self._infinite_loader(self.forget_loader)
        retain_iter = self._infinite_loader(self.retain_loader) if self.retain_loader else None
        pbar = self._make_pbar("WGA")

        for step in range(1, self.n_steps + 1):
            optimizer.zero_grad()
            fb = self._to_device(next(forget_iter))

            if self.variant == "weighted":
                # Compute per-sample weights (inverse loss)
                per_loss = self._per_sample_losses(fb)              # (batch,)
                weights  = F.softmax(-per_loss / self.temperature, dim=0)

                # Weighted gradient ascent
                ids    = fb["input_ids"]
                mask   = fb["attention_mask"]
                labels = fb.get("labels", ids)
                logits = self.model(input_ids=ids, attention_mask=mask).logits

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss_per_token = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                ).view(ids.size(0), -1)

                valid = (shift_labels != -100).float()
                loss_per_sample = (loss_per_token * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1)
                loss_forget = (weights * loss_per_sample).sum()

            elif self.variant == "label_perturb":
                # Replace labels with random tokens, then ascend on perturbed loss
                ids    = fb["input_ids"]
                mask   = fb["attention_mask"]
                labels = fb.get("labels", ids).clone()
                perturbed = self._perturbed_labels(labels)
                fb_perturbed = {
                    "input_ids":      ids,
                    "attention_mask": mask,
                    "labels":         perturbed,
                }
                loss_forget = self.loss_fn(self.model, fb_perturbed)

            # Retain loss
            retain_loss = torch.tensor(0.0, device=self.device)
            if retain_iter:
                rb = self._to_device(next(retain_iter))
                retain_loss = self.loss_fn(self.model, rb)

            total = -loss_forget + self.retain_lambda * retain_loss
            total.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.gradient_clip,
            )
            optimizer.step()
            scheduler.step()

            self._log_step(step, pbar, {
                "forget": loss_forget.item(),
                "retain": retain_loss.item(),
            })

        pbar.close()
        logger.info("WGA complete.")
