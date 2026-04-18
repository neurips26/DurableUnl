"""
BaseUnlearner — abstract base for all unlearning methods.

Every baseline implements run(). The public entry point is unlearn().
A tqdm progress bar shows steps in real time so you can see it's alive.
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..utils.logging_utils import get_logger, ResultLogger


# ─────────────────────────────────────────────────────────────────────────────
# CLM loss — the ONLY correct loss signature for these baselines
# ─────────────────────────────────────────────────────────────────────────────

def _clm_loss(model: nn.Module, batch: dict) -> torch.Tensor:
    """
    Causal LM cross-entropy.
    Signature: (model, batch_dict) -> scalar tensor.
    batch must contain: input_ids, attention_mask, labels.
    """
    device = _get_device(model)
    out = model(
        input_ids      = batch["input_ids"].to(device),
        attention_mask = batch["attention_mask"].to(device),
        labels         = batch.get("labels", batch["input_ids"]).to(device),
    )
    return out.loss


def _get_device(model: nn.Module) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UnlearningResult:
    method_name: str = ""
    started_at: str = ""
    finished_at: str = ""
    wall_time_seconds: float = -1.0
    peak_gpu_memory_gb: float = -1.0
    total_gradient_steps: int = -1
    forget_accuracy: float = -1.0
    retain_accuracy: float = -1.0
    mia_auc: float = -1.0
    quant_recovery: Dict[str, float] = field(default_factory=dict)
    ft_recovery: Dict[int, float] = field(default_factory=dict)
    adv_recovery: float = -1.0
    forget_loss_history: List[float] = field(default_factory=list)
    retain_loss_history: List[float] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method":         self.method_name,
            "started_at":     self.started_at,
            "finished_at":    self.finished_at,
            "wall_time_min":  round(self.wall_time_seconds / 60, 1),
            "peak_gpu_gb":    round(self.peak_gpu_memory_gb, 2),
            "gradient_steps": self.total_gradient_steps,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class BaseUnlearner:
    """
    Abstract base. Subclasses implement run(). Scripts call unlearn().

    A tqdm progress bar is shown during training so you can see
    each step completing in real time.
    """

    def __init__(
        self,
        model: nn.Module,
        forget_loader: Optional[DataLoader] = None,
        retain_loader: Optional[DataLoader] = None,
        loss_fn=None,
        device: Optional[torch.device] = None,
        n_steps: int = 300,
        retain_lambda: float = 1.0,
        lr: float = 5e-5,
        lr_forget: Optional[float] = None,
        gradient_clip: float = 1.0,
        log_every: int = 50,
        result_logger: Optional[ResultLogger] = None,
        **kwargs,   # absorb unknown keys silently
    ):
        self.model         = model
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self.n_steps       = n_steps
        self.retain_lambda = retain_lambda
        self.lr            = lr_forget if lr_forget is not None else lr
        self.gradient_clip = gradient_clip
        self.log_every     = log_every
        self.result_logger = result_logger

        # Always use the correct CLM loss — reject nn.CrossEntropyLoss
        if loss_fn is None or isinstance(loss_fn, nn.CrossEntropyLoss):
            self.loss_fn = _clm_loss
        else:
            self.loss_fn = loss_fn

        # Device: skip meta params (device_map="auto" artefact)
        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device
        else:
            self.device = _get_device(model)

        self.logger = get_logger(self.__class__.__name__)

    # ── Public entry point ────────────────────────────────────────────────────

    def unlearn(
        self,
        forget_loader: Optional[DataLoader] = None,
        retain_loader: Optional[DataLoader] = None,
        **kwargs,
    ) -> UnlearningResult:
        if forget_loader is not None:
            self.forget_loader = forget_loader
        if retain_loader is not None:
            self.retain_loader = retain_loader

        if self.forget_loader is None:
            raise ValueError("forget_loader required")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t0 = time.time()

        self.run()

        finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        wall_time   = time.time() - t0
        peak_mem    = (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available() else 0.0
        )

        self.logger.info(
            f"Finished at {finished_at} | "
            f"Time: {wall_time/60:.1f} min | Peak GPU: {peak_mem:.2f} GB"
        )

        return UnlearningResult(
            method_name          = self.__class__.__name__,
            started_at           = started_at,
            finished_at          = finished_at,
            wall_time_seconds    = wall_time,
            peak_gpu_memory_gb   = peak_mem,
            total_gradient_steps = self.n_steps,
        )

    # ── Subclasses implement ──────────────────────────────────────────────────

    def run(self) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} must implement run()")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _infinite_loader(self, loader: DataLoader):
        """Cycle through a DataLoader forever. Raises if loader is empty."""
        if len(loader.dataset) == 0:
            raise RuntimeError(
                f"DataLoader dataset is empty! "
                "Check your HuggingFace token and dataset split."
            )
        while True:
            for batch in loader:
                yield batch

    def _to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _make_pbar(self, desc: str) -> tqdm:
        """Return a tqdm progress bar for the training loop."""
        return tqdm(
            total=self.n_steps,
            desc=f"{desc:>12}",
            unit="step",
            dynamic_ncols=True,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
        )

    def _log_step(self, step: int, pbar: tqdm, losses: Dict[str, float]):
        """Update tqdm bar and log every log_every steps."""
        pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()}, refresh=True)
        pbar.update(1)
        if step % self.log_every == 0:
            ts  = datetime.now().strftime("%H:%M:%S")
            msg = " | ".join(f"{k}={v:.4f}" for k, v in losses.items())
            self.logger.info(f"[{ts}] Step {step:4d}/{self.n_steps} | {msg}")
