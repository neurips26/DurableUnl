"""Gradient Ascent (GA) baseline unlearner."""
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .base import BaseUnlearner


class GradientAscent(BaseUnlearner):
    def __init__(self, *args, lr_retain: float = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_retain = lr_retain or self.lr * 0.5

    def run(self) -> None:
        self.logger.info(f"Starting GA | steps={self.n_steps} | lr={self.lr}")
        self.model.train()
        optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_steps)

        forget_iter = self._infinite_loader(self.forget_loader)
        retain_iter = self._infinite_loader(self.retain_loader) if self.retain_loader else None
        pbar = self._make_pbar("GA")

        for step in range(1, self.n_steps + 1):
            optimizer.zero_grad()
            fb = self._to_device(next(forget_iter))
            lf = self.loss_fn(self.model, fb)

            lr = torch.tensor(0.0, device=self.device)
            if retain_iter:
                rb = self._to_device(next(retain_iter))
                lr = self.loss_fn(self.model, rb)

            (-lf + self.retain_lambda * lr).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.gradient_clip
            )
            optimizer.step(); scheduler.step()
            self._log_step(step, pbar, {"forget": lf.item(), "retain": lr.item()})

        pbar.close()
        self.logger.info("GA complete.")
