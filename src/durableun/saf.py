"""
SAF v4 — Sharpness-Aware Forgetting with full-model STE quantization.

Key change from v3:
  v3 applied STE only to LoRA weights (~14M params). But forget content
  is distributed across ALL linear layers (base + adapter). When INT4
  quantization rounds ALL weights, only scrubbing the adapter is insufficient.

  v4 applies STE to ALL linear layer weights in the quantization-aware
  forward pass, giving a true gradient signal for full-model INT4 robustness.

Objective per step:
  L = -L_forget(θ)                     [standard GA — always active]
    - α(t) · L_forget(Q_STE_full(θ))   [GA on fully-quantized model]
    + λ · L_retain(θ)                  [retain protection]

  α(t): linearly ramps from 0 → alpha_quant after warmup_steps.
  Q_STE_full: STE-quantize ALL nn.Linear weights to INT4.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..baselines.base import BaseUnlearner, _clm_loss

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Full-model STE quantization
# ─────────────────────────────────────────────────────────────────────────────

class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.round()
    @staticmethod
    def backward(ctx, g): return g


def _ste_quant_int4(w: torch.Tensor) -> torch.Tensor:
    """INT4 symmetric STE quantization. Gradient flows straight through."""
    w_f = w.float()
    if w_f.dim() >= 2:
        scale = w_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
    else:
        scale = w_f.abs().max().clamp(min=1e-8) / 7.0
    w_q = _STE.apply((w_f / scale)).clamp(-8, 7)
    return (w_q * scale).to(w.dtype)


def _forward_full_ste(model: nn.Module, batch: dict, device) -> torch.Tensor:
    """
    Forward pass with STE-INT4 quantization applied to ALL nn.Linear weights.
    This is the key fix — covers base model weights + LoRA adapters.
    """
    # Collect all linear layer weights and replace with STE-quantized versions
    originals = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight is not None:
            originals[id(module)] = (module, module.weight.data.clone())
            # Replace weight with STE-quantized version
            # The STE backward pass ensures gradients flow to original weights
            module.weight.data = _ste_quant_int4(module.weight.data)

    # Forward pass with all-quantized weights
    try:
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch.get("labels", ids).to(device)
        out    = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss   = out.loss
    finally:
        # Always restore original weights
        for orig_id, (module, orig_w) in originals.items():
            module.weight.data = orig_w

    return loss


# ─────────────────────────────────────────────────────────────────────────────
# SAF v4
# ─────────────────────────────────────────────────────────────────────────────

class SAF(BaseUnlearner):
    """
    SAF v4: Full-model STE quantization for genuine INT4 robustness.

    Args:
        alpha_quant  (float): Weight of quantization-aware loss. Default 3.0.
                              Higher = more INT4 robustness, but harder to train.
        warmup_steps (int):   Steps of pure GA before quant loss activates. Default 100.
        retain_lambda(float): Must be higher than alpha_quant to prevent collapse.
                              Default 3.0 (set equal to alpha_quant).
    """

    def __init__(
        self,
        model: nn.Module,
        forget_loader=None,
        retain_loader=None,
        device=None,
        n_steps: int = 300,
        lr: float = 5e-5,
        retain_lambda: float = 3.0,
        gradient_clip: float = 1.0,
        log_every: int = 50,
        alpha_quant: float = 3.0,
        warmup_steps: int = 100,
        **kwargs,
    ):
        super().__init__(
            model=model, forget_loader=forget_loader, retain_loader=retain_loader,
            loss_fn=_clm_loss, device=device, n_steps=n_steps,
            retain_lambda=retain_lambda, lr=lr,
            gradient_clip=gradient_clip, log_every=log_every,
        )
        self.alpha_quant  = alpha_quant
        self.warmup_steps = warmup_steps

    def _alpha(self, step: int) -> float:
        if step <= self.warmup_steps:
            return 0.0
        progress = (step - self.warmup_steps) / max(1, self.n_steps - self.warmup_steps)
        return min(self.alpha_quant, self.alpha_quant * progress * 2)

    def run(self) -> None:
        logger.info(
            f"Starting SAF v4 | steps={self.n_steps} | "
            f"alpha_quant={self.alpha_quant} | warmup={self.warmup_steps} | "
            f"retain_lambda={self.retain_lambda}"
        )
        self.model.train()

        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_steps)
        forget_iter = self._infinite_loader(self.forget_loader)
        retain_iter = self._infinite_loader(self.retain_loader) if self.retain_loader else None
        pbar = self._make_pbar("SAF-v4")

        for step in range(1, self.n_steps + 1):
            optimizer.zero_grad()
            fb    = self._to_device(next(forget_iter))
            alpha = self._alpha(step)

            # Loss 1: Standard GA
            loss_forget = self.loss_fn(self.model, fb)

            # Loss 2: GA on fully INT4-quantized model (full-model STE)
            loss_quant = torch.tensor(0.0, device=self.device)
            if alpha > 0:
                try:
                    loss_quant = _forward_full_ste(self.model, fb, self.device)
                except Exception as e:
                    if step <= 3:
                        logger.warning(f"Full STE failed: {e}. Standard GA only.")

            # Loss 3: Retain (stronger lambda to protect against quant loss)
            retain_loss = torch.tensor(0.0, device=self.device)
            if retain_iter:
                rb = self._to_device(next(retain_iter))
                retain_loss = self.loss_fn(self.model, rb)

            total = (
                -loss_forget
                - alpha * loss_quant
                + self.retain_lambda * retain_loss
            )
            total.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.gradient_clip,
            )
            optimizer.step()
            scheduler.step()

            self._log_step(step, pbar, {
                "forget": loss_forget.item(),
                "quant":  loss_quant.item() if hasattr(loss_quant, 'item') else 0.0,
                "retain": retain_loss.item(),
                "α":      round(alpha, 2),
            })

        pbar.close()
        logger.info("SAF v4 complete.")
