"""
SCRUB baseline: gradient ascent + KL-retain from frozen reference.
OPTIMISED: KL computed on a small subset of the retain batch to avoid
the 66s/step issue seen with full 4-bit model KL computation.
"""
import copy
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .base import BaseUnlearner


class SCRUB(BaseUnlearner):
    def __init__(self, *args, gamma: float = 0.5, msteps: int = 1,
                 kl_batch_cap: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma       = gamma
        self.msteps      = msteps
        self.kl_batch_cap = kl_batch_cap  # cap KL to first N samples to save time

        self.logger.info("SCRUB: Creating frozen reference model...")
        self.ref_model = copy.deepcopy(self.model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()
        self.logger.info("SCRUB: reference model frozen.")

    def _kl(self, rb):
        """KL divergence on a capped batch size to keep step time reasonable."""
        ids  = rb["input_ids"].to(self.device)
        mask = rb["attention_mask"].to(self.device)

        # Cap to kl_batch_cap sequences — KL is expensive with 4-bit models
        if ids.shape[0] > self.kl_batch_cap:
            ids  = ids[:self.kl_batch_cap]
            mask = mask[:self.kl_batch_cap]

        with torch.no_grad():
            ref_lp = F.log_softmax(
                self.ref_model(input_ids=ids, attention_mask=mask).logits, dim=-1
            )
        cur_lp = F.log_softmax(
            self.model(input_ids=ids, attention_mask=mask).logits, dim=-1
        )
        kl = (ref_lp.exp() * (ref_lp - cur_lp)).sum(dim=-1)
        return (kl * mask.float()).sum() / mask.float().sum().clamp(min=1)

    def run(self) -> None:
        self.logger.info(
            f"Starting SCRUB | steps={self.n_steps} | gamma={self.gamma} | "
            f"kl_batch_cap={self.kl_batch_cap}"
        )
        self.model.train()
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_steps)

        forget_iter = self._infinite_loader(self.forget_loader)
        retain_iter = self._infinite_loader(self.retain_loader) if self.retain_loader else None
        pbar        = self._make_pbar("SCRUB")

        for step in range(1, self.n_steps + 1):
            # Ascent on forget
            af = 0.0
            for _ in range(self.msteps):
                optimizer.zero_grad()
                fb = self._to_device(next(forget_iter))
                lf = self.loss_fn(self.model, fb)
                (-self.gamma * lf).backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.gradient_clip,
                )
                optimizer.step()
                af += lf.item()
            af /= self.msteps

            # KL-retain (capped batch)
            kl = torch.tensor(0.0, device=self.device)
            if retain_iter:
                optimizer.zero_grad()
                rb  = self._to_device(next(retain_iter))
                kl  = self._kl(rb)
                (self.retain_lambda * kl).backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.gradient_clip,
                )
                optimizer.step()

            scheduler.step()
            self._log_step(step, pbar, {"forget": af, "kl": kl.item()})

        pbar.close()
        self.logger.info("SCRUB complete.")
