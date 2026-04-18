"""
src/baselines/tv_distance.py
==============================
Task Vector Unlearning (Ilharco et al. 2023 / Yu et al. 2023 DARE).

Modern baseline: instead of gradient ascent during fine-tuning,
negate the "task vector" — the difference between the fine-tuned
and pre-trained weights.

For LoRA models:
  task_vector = lora_weights (since base weights are frozen)
  unlearned   = base_model + (-scale * lora_weights)

Variants:
  "negate":  θ_unlearn = θ_base - scale * (θ_finetuned - θ_base)
             = standard task vector negation (Ilharco et al.)

  "dare":    θ_unlearn = θ_base + DARE_prune(-(θ_finetuned - θ_base))
             DARE randomly drops p% of task vector elements before negating
             (Yu et al. 2023)

This is a training-free method — no gradient steps needed.
Runtime: ~30 seconds (just weight arithmetic).

Reference:
  Ilharco et al. "Editing Models with Task Arithmetic." ICLR 2023.
  Yu et al. "DARE: Language Model Delta Weights without Parameters."
            arXiv 2023.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from .base import BaseUnlearner, UnlearningResult

logger = logging.getLogger(__name__)


class TaskVectorUnlearning(BaseUnlearner):
    """
    Task Vector negation for unlearning LoRA-finetuned models.

    For a 4-bit base model + LoRA: the "task vector" is the LoRA adapter weights.
    Negating it sets LoRA weights to their negative, reversing the fine-tuning direction.

    Args:
        scale      (float): Scaling factor for task vector negation.
                            1.0 = full negation. 0.5 = half-step back.
        method     (str):   "negate" (standard) or "dare" (with random pruning).
        dare_p     (float): Fraction of task vector to randomly drop (DARE method).
                            Default 0.9 (drop 90%, keep 10%).
    """

    def __init__(
        self,
        model: nn.Module,
        forget_loader=None,
        retain_loader=None,
        device=None,
        n_steps: int = 0,           # not used — training-free
        lr: float = 0.0,            # not used
        retain_lambda: float = 0.0, # not used
        gradient_clip: float = 1.0, # not used
        log_every: int = 1,
        scale: float = 1.0,
        method: str = "negate",     # "negate" or "dare"
        dare_p: float = 0.9,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            model=model, forget_loader=forget_loader, retain_loader=retain_loader,
            loss_fn=None, device=device, n_steps=n_steps,
            retain_lambda=retain_lambda, lr=lr,
            gradient_clip=gradient_clip, log_every=log_every,
        )
        self.scale  = scale
        self.method = method
        self.dare_p = dare_p
        self.seed   = seed

    def run(self) -> None:
        """
        Apply task vector negation — no training loop.
        Directly modifies LoRA adapter weights in-place.
        """
        logger.info(
            f"Starting TaskVectorUnlearning | method={self.method} | "
            f"scale={self.scale}"
            + (f" | dare_p={self.dare_p}" if self.method == "dare" else "")
        )

        n_modified = 0
        total_params = 0

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Only modify LoRA adapter weights (not base model frozen weights)
                if not param.requires_grad:
                    continue
                if param.data.numel() == 0:
                    continue

                total_params += param.data.numel()
                task_vec = param.data.clone()   # LoRA weights ARE the task vector

                if self.method == "negate":
                    # θ_new = -scale * task_vector
                    param.data = -self.scale * task_vec

                elif self.method == "dare":
                    # DARE: randomly prune task vector before negating
                    torch.manual_seed(self.seed + hash(name) % 10000)
                    mask = (torch.rand_like(task_vec) > self.dare_p).float()
                    # Scale kept weights to maintain expectation
                    rescale = 1.0 / (1.0 - self.dare_p + 1e-8)
                    param.data = -self.scale * (task_vec * mask * rescale)

                n_modified += param.data.numel()

        logger.info(
            f"TaskVectorUnlearning complete. "
            f"Modified {n_modified:,} / {total_params:,} trainable params."
        )
