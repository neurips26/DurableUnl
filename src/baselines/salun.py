"""SalUn — Saliency-based unlearning."""
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .base import BaseUnlearner


class SalUn(BaseUnlearner):
    def __init__(self, *args, saliency_threshold: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.saliency_threshold = saliency_threshold

    def _compute_mask(self, n_batches: int = 10):
        self.logger.info("Computing saliency mask...")
        self.model.eval()
        accum = {n: torch.zeros_like(p.data)
                 for n, p in self.model.named_parameters() if p.requires_grad}

        for i, batch in enumerate(self.forget_loader):
            if i >= n_batches: break
            self.model.zero_grad()
            self.loss_fn(self.model, self._to_device(batch)).backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    accum[n] += p.grad.data.abs()

        self.model.zero_grad()
        all_g = torch.cat([g.view(-1) for g in accum.values()])
        thr   = torch.quantile(all_g, 1.0 - self.saliency_threshold)
        masks = {n: (g >= thr).float() for n, g in accum.items()}
        sel   = sum(m.sum().item() for m in masks.values())
        tot   = sum(m.numel()      for m in masks.values())
        self.logger.info(f"Saliency mask: {sel:.0f}/{tot} ({100*sel/tot:.1f}%)")
        self.model.train()
        return masks

    def run(self) -> None:
        self.logger.info(f"Starting SalUn | steps={self.n_steps}")
        masks     = self._compute_mask()
        optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_steps)

        forget_iter = self._infinite_loader(self.forget_loader)
        retain_iter = self._infinite_loader(self.retain_loader) if self.retain_loader else None
        pbar = self._make_pbar("SalUn")

        for step in range(1, self.n_steps + 1):
            optimizer.zero_grad()
            fb = self._to_device(next(forget_iter))
            lf = self.loss_fn(self.model, fb)
            (-lf).backward()
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None and n in masks:
                        p.grad.mul_(masks[n].to(p.grad.device))

            rl = torch.tensor(0.0, device=self.device)
            if retain_iter:
                rb = self._to_device(next(retain_iter))
                rl = self.loss_fn(self.model, rb)
                (self.retain_lambda * rl).backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.gradient_clip
            )
            optimizer.step(); scheduler.step()
            self._log_step(step, pbar, {"forget": lf.item(), "retain": rl.item()})

        pbar.close()
        self.logger.info("SalUn complete.")
