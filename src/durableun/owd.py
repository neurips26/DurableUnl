"""
Phase 2 — OWD: Orthogonal Weight Deactivation
==============================================
Defends against: fine-tuning recovery.

Core idea:
  After fine-tuning on unrelated data (Alpaca, C4), a model's weight
  updates lie predominantly in the subspace spanned by the gradients
  of the downstream task. If the forget update also lies in this subspace,
  fine-tuning can accidentally "undo" the unlearning.

  OWD projects the forget gradient update into the NULL SPACE of the
  downstream task gradient subspace:

      V_downstream = top-k right singular vectors of G_downstream
      V_orth       = null space of V_downstream (complement)
      update       = (I - V_downstream V_downstream^T) · ∇L_forget

  This guarantees: fine-tuning on downstream data cannot restore the
  forget content because the forget update is orthogonal to the
  downstream fine-tuning direction.

  Builds on: AlphaEdit (Fang et al., 2024) but uses downstream task
  gradients instead of retain gradients for the null projection.
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..baselines.base import BaseUnlearner, _clm_loss, _get_device
from ..data.data_utils import get_downstream_dataloader

logger = logging.getLogger(__name__)


class OWD(BaseUnlearner):
    """
    Orthogonal Weight Deactivation (Phase 2 of DurableUn).

    Hyperparameters:
        svd_rank               (int):  Rank of downstream subspace to block.
                                       Default 64.
        downstream_datasets    (list): Datasets for computing downstream subspace.
                                       Default ['alpaca', 'c4'].
        n_downstream_samples   (int):  Samples per dataset for subspace computation.
                                       Default 200.
        n_subspace_batches     (int):  Batches to accumulate for subspace gradient.
                                       Default 20.
    """

    def __init__(
        self,
        model: nn.Module,
        forget_loader: Optional[DataLoader] = None,
        retain_loader: Optional[DataLoader] = None,
        tokenizer=None,
        device: Optional[torch.device] = None,
        n_steps: int = 300,
        lr: float = 5e-5,
        retain_lambda: float = 1.0,
        gradient_clip: float = 1.0,
        log_every: int = 50,
        svd_rank: int = 64,
        downstream_datasets: Optional[List[str]] = None,
        n_downstream_samples: int = 200,
        n_subspace_batches: int = 20,
        max_length: int = 256,
        cache_dir: Optional[str] = None,
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
        self.svd_rank             = svd_rank
        self.downstream_datasets  = downstream_datasets or ["alpaca", "c4"]
        self.n_downstream_samples = n_downstream_samples
        self.n_subspace_batches   = n_subspace_batches
        self.max_length           = max_length
        self.cache_dir            = cache_dir
        self.tokenizer            = tokenizer
        self._null_proj           = {}

    # ── Compute downstream subspace ───────────────────────────────────────────

    def _compute_downstream_subspace(self):
        """
        Accumulate gradients on downstream tasks, SVD → null projectors.
        """
        logger.info(
            f"OWD: Computing downstream subspace | "
            f"datasets={self.downstream_datasets} | rank={self.svd_rank}"
        )
        if self.tokenizer is None:
            logger.warning("OWD: No tokenizer provided — using retain set as proxy for downstream.")
            return self._compute_retain_subspace()

        try:
            downstream_loader = get_downstream_dataloader(
                self.tokenizer,
                datasets=self.downstream_datasets,
                n_samples_per_dist=self.n_downstream_samples,
                max_length=self.max_length,
                batch_size=4,
                num_workers=0,
                cache_dir=self.cache_dir,
            )
        except Exception as e:
            logger.warning(f"OWD: Downstream loader failed ({e}), using retain as proxy.")
            return self._compute_retain_subspace()

        return self._accumulate_and_project(downstream_loader, self.n_subspace_batches)

    def _compute_retain_subspace(self):
        """Fallback: use retain loader if downstream datasets unavailable."""
        logger.info("OWD: Using retain set as downstream proxy.")
        return self._accumulate_and_project(self.retain_loader, self.n_subspace_batches)

    def _accumulate_and_project(self, loader: DataLoader, n_batches: int) -> dict:
        """Accumulate gradients from loader → compute null projectors."""
        self.model.eval()

        grad_accum = {
            name: torch.zeros_like(p.data)
            for name, p in self.model.named_parameters()
            if p.requires_grad and p.dim() >= 2
        }

        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            self.model.zero_grad()
            self.loss_fn(self.model, self._to_device(batch)).backward()
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None and name in grad_accum:
                    grad_accum[name] += p.grad.data.clone()

        self.model.zero_grad()

        # SVD → null projector for each param group
        proj = {}
        n_computed = 0
        for name, G in grad_accum.items():
            G2d  = G.view(G.shape[0], -1).float()
            rank = min(self.svd_rank, *G2d.shape)
            try:
                _, _, Vh  = torch.linalg.svd(G2d, full_matrices=False)
                V         = Vh[:rank].T                # (cols, rank)
                P_null    = torch.eye(V.shape[0], device=V.device) - V @ V.T
                proj[name] = P_null.to(G.dtype)
                n_computed += 1
            except Exception as e:
                logger.debug(f"OWD: SVD failed for {name}: {e}")
                proj[name] = None

        logger.info(f"OWD: Null projectors computed for {n_computed} param groups.")
        self.model.train()
        return proj

    def _apply_null_projection(self):
        """Project gradients into null space of downstream subspace."""
        if not self._null_proj:
            return
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if not (p.requires_grad and p.grad is not None and name in self._null_proj):
                    continue
                P = self._null_proj[name]
                if P is None:
                    continue
                g = p.grad.data.view(p.shape[0], -1).float()
                if g.shape[1] == P.shape[0]:
                    p.grad.data.copy_((g @ P).view_as(p.grad).to(p.grad.dtype))

    # ── Main training loop ────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info(f"Starting OWD | steps={self.n_steps} | svd_rank={self.svd_rank}")

        # Compute null projectors once before training
        self._null_proj = self._compute_downstream_subspace()

        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_steps)

        forget_iter = self._infinite_loader(self.forget_loader)
        retain_iter = self._infinite_loader(self.retain_loader) if self.retain_loader else None
        pbar        = self._make_pbar("OWD")

        for step in range(1, self.n_steps + 1):
            optimizer.zero_grad()
            fb = self._to_device(next(forget_iter))
            lf = self.loss_fn(self.model, fb)
            (-lf).backward()

            # Project forget gradient into null space of downstream subspace
            self._apply_null_projection()

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
            scheduler.step()
            self._log_step(step, pbar, {"forget": lf.item(), "retain": retain_loss.item()})

        pbar.close()
        logger.info("OWD complete.")
