"""
src/baselines/gradient_difference.py
======================================
Gradient Difference (GradDiff) — Maini et al. (2024) TOFU paper baseline.

This is the method proposed specifically FOR the TOFU benchmark and should
be included as it is the paper-recommended baseline for this dataset.

Objective:
  L = -L_forget(θ) + λ · |L_retain(θ) - L_retain(θ_original)|

  Where θ_original is the original fine-tuned model (frozen reference).
  This penalises DEVIATION from the original retain loss, not just high retain loss.
  This is subtly different from standard GA which only penalises low retain accuracy.

Reference:
  Maini et al. "TOFU: A Task of Fictitious Unlearning for LLMs." COLM 2024.
  https://arxiv.org/abs/2401.06121
"""

import logging
import copy
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .base import BaseUnlearner, _clm_loss

logger = logging.getLogger(__name__)


class GradDiff(BaseUnlearner):
    """
    Gradient Difference unlearning (Maini et al., 2024).
    
    The retain term penalises deviation from the reference model's retain loss
    rather than just encouraging high retain accuracy. This is the TOFU paper's
    recommended approach for the benchmark.
    
    Args:
        model:          The model to unlearn (LoRA-equipped).
        ref_model:      Frozen reference model (original fine-tuned). If None,
                        copies the input model at init time.
        forget_loader:  DataLoader for forget set.
        retain_loader:  DataLoader for retain set.
        grad_diff_coeff (float): Weight of gradient difference term. Default 1.0.
    """
    
    def __init__(
        self,
        model: nn.Module,
        forget_loader: Optional[DataLoader] = None,
        retain_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        n_steps: int = 300,
        lr: float = 5e-5,
        retain_lambda: float = 1.0,
        gradient_clip: float = 1.0,
        log_every: int = 50,
        ref_model: Optional[nn.Module] = None,
        grad_diff_coeff: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            model=model, forget_loader=forget_loader, retain_loader=retain_loader,
            loss_fn=_clm_loss, device=device, n_steps=n_steps,
            retain_lambda=retain_lambda, lr=lr,
            gradient_clip=gradient_clip, log_every=log_every,
        )
        self.grad_diff_coeff = grad_diff_coeff
        
        # Reference model: frozen copy of the model before unlearning
        if ref_model is not None:
            self.ref_model = ref_model
        else:
            logger.info("GradDiff: Creating frozen reference model copy...")
            self.ref_model = copy.deepcopy(model)
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
            self.ref_model.eval()
    
    def run(self) -> None:
        logger.info(
            f"Starting GradDiff | steps={self.n_steps} | "
            f"grad_diff_coeff={self.grad_diff_coeff} | lr={self.lr}"
        )
        self.model.train()
        self.ref_model.eval()
        
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_steps)
        
        forget_iter = self._infinite_loader(self.forget_loader)
        retain_iter = self._infinite_loader(self.retain_loader) if self.retain_loader else None
        pbar = self._make_pbar("GradDiff")
        
        for step in range(1, self.n_steps + 1):
            optimizer.zero_grad()
            
            # 1. Forget loss: maximize loss on forget set (gradient ascent)
            fb = self._to_device(next(forget_iter))
            loss_forget = self.loss_fn(self.model, fb)
            
            # 2. Retain gradient difference
            grad_diff_loss = torch.tensor(0.0, device=self.device)
            if retain_iter:
                rb = self._to_device(next(retain_iter))
                
                # Current model retain loss
                loss_retain_current = self.loss_fn(self.model, rb)
                
                # Reference model retain loss (no gradient)
                with torch.no_grad():
                    loss_retain_ref = self.loss_fn(self.ref_model, rb)
                
                # Gradient difference: penalise deviation from reference
                grad_diff_loss = (loss_retain_current - loss_retain_ref).abs()
            
            # Total loss: -forget + λ * |retain - retain_ref|
            total = -loss_forget + self.retain_lambda * grad_diff_loss
            total.backward()
            
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.gradient_clip,
            )
            optimizer.step()
            scheduler.step()
            
            self._log_step(step, pbar, {
                "forget": loss_forget.item(),
                "grad_diff": grad_diff_loss.item(),
            })
        
        pbar.close()
        logger.info("GradDiff complete.")
