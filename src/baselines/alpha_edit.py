"""AlphaEdit — Null-Space Constrained Unlearning."""
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .base import BaseUnlearner


class AlphaEdit(BaseUnlearner):
    def __init__(self, *args, svd_rank: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.svd_rank = svd_rank
        self._proj    = {}

    def _compute_null(self, n_batches: int = 10):
        self.logger.info("AlphaEdit: Computing retain gradient subspace...")
        self.model.eval()
        gs = {n: torch.zeros_like(p.data)
              for n, p in self.model.named_parameters()
              if p.requires_grad and p.dim() >= 2}

        for i, batch in enumerate(self.retain_loader):
            if i >= n_batches: break
            self.model.zero_grad()
            self.loss_fn(self.model, self._to_device(batch)).backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None and n in gs:
                    gs[n] += p.grad.data.clone()

        self.model.zero_grad()
        proj = {}
        for n, G in gs.items():
            G2d  = G.view(G.shape[0], -1).float()
            rank = min(self.svd_rank, *G2d.shape)
            try:
                _, _, Vh = torch.linalg.svd(G2d, full_matrices=False)
                V        = Vh[:rank].T
                P        = torch.eye(V.shape[0], device=V.device) - V @ V.T
                proj[n]  = P.to(G.dtype)
            except Exception:
                proj[n] = None
        self.logger.info(f"AlphaEdit: {len(proj)} projectors computed.")
        self.model.train()
        return proj

    def run(self) -> None:
        self.logger.info(f"Starting AlphaEdit | steps={self.n_steps}")
        if self.retain_loader:
            self._proj = self._compute_null()

        optimizer   = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)
        scheduler   = CosineAnnealingLR(optimizer, T_max=self.n_steps)
        forget_iter = self._infinite_loader(self.forget_loader)
        pbar        = self._make_pbar("AlphaEdit")

        for step in range(1, self.n_steps + 1):
            optimizer.zero_grad()
            fb = self._to_device(next(forget_iter))
            lf = self.loss_fn(self.model, fb)
            (-lf).backward()

            if self._proj:
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        if p.requires_grad and p.grad is not None and n in self._proj:
                            P = self._proj[n]
                            if P is None: continue
                            g = p.grad.data.view(p.shape[0], -1).float()
                            if g.shape[1] == P.shape[0]:
                                p.grad.data.copy_((g @ P).view_as(p.grad).to(p.grad.dtype))

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.gradient_clip
            )
            optimizer.step(); scheduler.step()
            self._log_step(step, pbar, {"forget": lf.item()})

        pbar.close()
        self.logger.info("AlphaEdit complete.")
