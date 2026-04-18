"""
src/baselines/langevin_unlearn.py
===================================
Noisy Gradient Unlearning (NG) / Stochastic Gradient Langevin Unlearning.

Adds Gaussian noise to forget gradients during gradient ascent.
This prevents the model from memorizing the exact "unlearned" direction,
providing stronger privacy guarantees at the cost of slightly noisier forgetting.

Two variants:
  "ng":       Gaussian noise injected at each gradient step.
              θ_{t+1} = θ_t + η(∇L_forget + ε) - η·λ·∇L_retain
              where ε ~ N(0, σ²I)

  "langevin": Full Stochastic Gradient Langevin Dynamics (SGLD).
              θ_{t+1} = θ_t - η∇L + √(2η)·N(0,I)
              This implements approximate Bayesian posterior sampling
              over the "unlearned" distribution.

Reference:
  Neel et al. "Descent-to-delete: Gradient-based methods for machine
               unlearning." ALT 2021.
  Welling & Teh. "Bayesian learning via stochastic gradient Langevin
                  dynamics." ICML 2011.
  Also see: Chien et al. "Langevin Unlearning." ICML 2024 Workshop.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseUnlearner, _clm_loss

logger = logging.getLogger(__name__)


class NoisyGradientUnlearning(BaseUnlearner):
    """
    Noisy Gradient / Langevin unlearning.

    Args:
        noise_std   (float): Std of injected Gaussian noise.
                             For "ng": absolute noise std.
                             For "langevin": scales with sqrt(2*lr).
                             Default 0.01.
        variant     (str):   "ng" (additive noise) or "langevin" (SGLD).
        noise_scale (str):   "fixed" or "adaptive" (scale noise by gradient norm).
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
        noise_std: float = 0.01,
        variant: str = "ng",
        noise_scale: str = "fixed",
        **kwargs,
    ):
        super().__init__(
            model=model, forget_loader=forget_loader, retain_loader=retain_loader,
            loss_fn=_clm_loss, device=device, n_steps=n_steps,
            retain_lambda=retain_lambda, lr=lr,
            gradient_clip=gradient_clip, log_every=log_every,
        )
        self.noise_std   = noise_std
        self.variant     = variant
        self.noise_scale = noise_scale

    def _inject_noise(self, step: int):
        """Inject Gaussian noise into current gradients."""
        with torch.no_grad():
            for p in self.model.parameters():
                if not p.requires_grad or p.grad is None:
                    continue

                if self.variant == "langevin":
                    # SGLD noise: N(0, 2*lr*I)
                    std = math.sqrt(2 * self.lr)
                elif self.variant == "ng":
                    std = self.noise_std
                    if self.noise_scale == "adaptive":
                        # Scale noise by gradient norm for stability
                        grad_norm = p.grad.norm().item()
                        std = self.noise_std * max(grad_norm, 1e-6)
                else:
                    std = self.noise_std

                noise = torch.randn_like(p.grad) * std
                p.grad.data.add_(noise)

    def run(self) -> None:
        logger.info(
            f"Starting NoisyGradientUnlearning | variant={self.variant} | "
            f"noise_std={self.noise_std} | noise_scale={self.noise_scale}"
        )
        self.model.train()

        # Use SGD for Langevin (no momentum), AdamW for NG
        if self.variant == "langevin":
            from torch.optim import SGD
            optimizer = SGD(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.lr,
            )
        else:
            from torch.optim import AdamW
            from torch.optim.lr_scheduler import CosineAnnealingLR
            optimizer = AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.lr,
            )

        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_steps)

        forget_iter = self._infinite_loader(self.forget_loader)
        retain_iter = self._infinite_loader(self.retain_loader) if self.retain_loader else None
        pbar = self._make_pbar("NoisyGA")

        for step in range(1, self.n_steps + 1):
            optimizer.zero_grad()
            fb = self._to_device(next(forget_iter))

            loss_forget = self.loss_fn(self.model, fb)
            retain_loss = torch.tensor(0.0, device=self.device)
            if retain_iter:
                rb = self._to_device(next(retain_iter))
                retain_loss = self.loss_fn(self.model, rb)

            total = -loss_forget + self.retain_lambda * retain_loss
            total.backward()

            # Inject noise AFTER backward, BEFORE optimizer step
            self._inject_noise(step)

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
        logger.info("NoisyGradientUnlearning complete.")
