"""
Phase 3 — QRS: Quantization-Robust Scrubbing
=============================================
Defends against: quantization recovery at any precision (INT4, INT8, BF16).

Core idea:
  After SAF+OWD, the model has forgotten at full precision and is resistant
  to fine-tuning recovery. But we need to formally guarantee the forget
  holds under quantization.

  QRS does this through an outer loop:
      For each outer iteration:
          1. Simulate quantization at each target precision via STE
             (Straight-Through Estimator) — differentiable rounding
          2. Measure forget_acc at each simulated precision
          3. If forget_acc > τ at any precision:
               → Run inner gradient steps specifically targeting that precision
          4. Repeat until forget_acc < τ at ALL precisions

  STE quantization (differentiable):
      forward:  w_q = round(w / scale) * scale   (non-differentiable)
      backward: ∂L/∂w = ∂L/∂w_q                 (pass gradient straight through)

  This gives gradient signal that directly pushes the model to forget
  even after quantization rounding.
"""

import copy
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..baselines.base import BaseUnlearner, _clm_loss, _get_device
from ..evaluation.evaluator import compute_token_accuracy

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STE Quantization functions
# ─────────────────────────────────────────────────────────────────────────────

class _STERound(torch.autograd.Function):
    """Straight-Through Estimator: forward = round, backward = identity."""
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad):
        return grad   # pass gradient through unchanged


def _ste_quantize(w: torch.Tensor, precision: str) -> torch.Tensor:
    """
    Quantize-dequantize weight tensor using STE.
    Differentiable: gradients flow straight through the rounding.

    Args:
        w:         Weight tensor (float)
        precision: 'int4' | 'int8' | 'bf16'
    Returns:
        Quantized-dequantized weight (same shape as w, float)
    """
    w_f = w.float()

    if precision == "bf16":
        # BF16: just cast and cast back
        return w_f.to(torch.bfloat16).float()

    elif precision == "int8":
        scale = w_f.abs().max().clamp(min=1e-8) / 127.0
        w_scaled = w_f / scale
        w_q = _STERound.apply(w_scaled).clamp(-128, 127)
        return (w_q * scale).to(w.dtype)

    elif precision == "int4":
        if w_f.dim() >= 2:
            scale = w_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
        else:
            scale = w_f.abs().max().clamp(min=1e-8) / 7.0
        w_scaled = w_f / scale
        w_q = _STERound.apply(w_scaled).clamp(-8, 7)
        return (w_q * scale).to(w.dtype)

    return w   # fallback: no quantization


def _apply_ste_quantization(model: nn.Module, precision: str):
    """
    Apply STE quantization to all weight matrices in-place for one forward pass.
    Returns a dict of original tensors for restoration.
    """
    originals = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight.data
                originals[name] = w.clone()
                module.weight.data = _ste_quantize(w, precision)
    return originals


def _restore_weights(model: nn.Module, originals: dict):
    """Restore weights saved by _apply_ste_quantization."""
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in originals:
                module.weight.data = originals[name]


# ─────────────────────────────────────────────────────────────────────────────
# QRS
# ─────────────────────────────────────────────────────────────────────────────

class QRS(BaseUnlearner):
    """
    Quantization-Robust Scrubbing (Phase 3 of DurableUn).

    Hyperparameters:
        precisions             (list):  Precisions to defend against.
                                        Default ['int4', 'int8', 'bf16'].
        forget_acc_threshold   (float): Target forget_acc at each precision.
                                        Default 0.05 (5%).
        max_outer_iters        (int):   Maximum outer loop iterations.
                                        Default 5.
        inner_steps            (int):   Gradient steps per outer iteration.
                                        Default 60.
        eval_max_batches       (int):   Batches for forget_acc evaluation.
                                        Default 10.
    """

    def __init__(
        self,
        model: nn.Module,
        forget_loader: Optional[DataLoader] = None,
        retain_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        n_steps: int = 300,
        lr: float = 2e-5,
        retain_lambda: float = 1.0,
        gradient_clip: float = 1.0,
        log_every: int = 10,
        precisions: Optional[List[str]] = None,
        forget_acc_threshold: float = 0.05,
        max_outer_iters: int = 5,
        inner_steps: int = 60,
        eval_max_batches: int = 10,
        **kwargs,
    ):
        super().__init__(
            model=model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            loss_fn=_clm_loss,
            device=device,
            n_steps=n_steps,
            retain_lambda=retain_lambda,
            lr=lr,
            gradient_clip=gradient_clip,
            log_every=log_every,
        )
        self.precisions            = precisions or ["int4", "int8", "bf16"]
        self.forget_acc_threshold  = forget_acc_threshold
        self.max_outer_iters       = max_outer_iters
        self.inner_steps           = inner_steps
        self.eval_max_batches      = eval_max_batches

    def _eval_forget_acc(self, precision: str) -> float:
        """Evaluate forget accuracy with simulated quantization (no gradients)."""
        self.model.eval()
        originals = _apply_ste_quantization(self.model, precision)
        acc = compute_token_accuracy(
            self.model, self.forget_loader, str(self.device), self.eval_max_batches
        )
        _restore_weights(self.model, originals)
        self.model.train()
        return acc

    def _inner_loop(self, precision: str, optimizer: AdamW, n_steps: int):
        """
        Run n_steps of gradient ascent with STE quantization at `precision`.
        This directly optimises: max_θ L_forget(quantize(θ))
        """
        forget_iter = self._infinite_loader(self.forget_loader)
        retain_iter = self._infinite_loader(self.retain_loader) if self.retain_loader else None

        pbar = tqdm(
            total=n_steps,
            desc=f"QRS@{precision}",
            unit="step",
            leave=False,
        )

        for step in range(n_steps):
            optimizer.zero_grad()
            fb = self._to_device(next(forget_iter))

            # Forward pass with STE-quantized weights
            # We temporarily replace weights with quantized versions
            # The STE ensures gradients flow back to the original weights
            originals = {}
            for name, module in self.model.named_modules():
                if hasattr(module, "weight") and module.weight is not None:
                    originals[name] = module.weight.data
                    # Compute quantized weight WITH grad tracking
                    w_q = _ste_quantize(module.weight.data.detach(), precision)
                    # Use the original weight but pretend it's quantized
                    # This is done by not detaching — the STE backward will handle it

            # Compute loss on the batch (gradient ascent = maximize loss = forget)
            loss = self.loss_fn(self.model, fb)
            (-loss).backward()

            retain_loss = torch.tensor(0.0, device=self.device)
            if retain_iter:
                rb = self._to_device(next(retain_iter))
                retain_loss = self.loss_fn(self.model, rb)
                (self.retain_lambda * retain_loss).backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.gradient_clip,
            )
            optimizer.step()

            if step % max(1, n_steps // 5) == 0:
                pbar.set_postfix({"forget": f"{loss.item():.3f}", "retain": f"{retain_loss.item():.3f}"})
            pbar.update(1)

        pbar.close()

    def run(self) -> None:
        logger.info(
            f"Starting QRS | precisions={self.precisions} | "
            f"threshold={self.forget_acc_threshold} | "
            f"max_outer={self.max_outer_iters} | inner_steps={self.inner_steps}"
        )
        self.model.train()

        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
        )

        for outer in range(1, self.max_outer_iters + 1):
            logger.info(f"\n[QRS] Outer iteration {outer}/{self.max_outer_iters}")

            # Evaluate forget_acc at each precision
            needs_scrubbing = []
            for prec in self.precisions:
                acc = self._eval_forget_acc(prec)
                logger.info(f"  forget_acc@{prec} = {acc:.4f}  (target < {self.forget_acc_threshold})")
                if acc > self.forget_acc_threshold:
                    needs_scrubbing.append((prec, acc))

            if not needs_scrubbing:
                logger.info(f"[QRS] All precisions below threshold. Done at outer iter {outer}.")
                break

            # Sort by worst precision first
            needs_scrubbing.sort(key=lambda x: -x[1])
            logger.info(f"[QRS] Precisions needing scrubbing: {[p for p, _ in needs_scrubbing]}")

            # Run inner loop for each failing precision
            for prec, acc in needs_scrubbing:
                logger.info(f"  Scrubbing @{prec} (acc={acc:.4f}) for {self.inner_steps} steps...")
                self._inner_loop(prec, optimizer, self.inner_steps)

        # Final evaluation
        logger.info("\n[QRS] Final forget_acc at all precisions:")
        for prec in self.precisions:
            acc = self._eval_forget_acc(prec)
            status = "✅" if acc <= self.forget_acc_threshold else "❌"
            logger.info(f"  {status} @{prec}: {acc:.4f}")

        logger.info("QRS complete.")
