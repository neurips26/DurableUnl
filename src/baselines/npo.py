"""NPO (Negative Preference Optimisation) baseline."""
import copy
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .base import BaseUnlearner


class NPO(BaseUnlearner):
    def __init__(self, *args, beta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.logger.info("NPO: Creating frozen reference model...")
        self.ref_model = copy.deepcopy(self.model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()
        self.logger.info("NPO reference model frozen.")

    def _seq_lp(self, model, batch):
        device = self.device
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl  = batch.get("labels", ids).to(device)
        with torch.set_grad_enabled(model.training):
            logits = model(input_ids=ids, attention_mask=mask).logits
        lp     = F.log_softmax(logits[:, :-1, :], dim=-1)
        tgt    = lbl[:, 1:].clamp(min=0)
        tok_lp = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
        valid  = (lbl[:, 1:] != -100).float()
        return (tok_lp * valid).sum(dim=-1)

    def run(self) -> None:
        self.logger.info(f"Starting NPO | steps={self.n_steps} | beta={self.beta}")
        self.model.train()
        optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_steps)

        forget_iter = self._infinite_loader(self.forget_loader)
        retain_iter = self._infinite_loader(self.retain_loader) if self.retain_loader else None
        pbar = self._make_pbar("NPO")

        for step in range(1, self.n_steps + 1):
            optimizer.zero_grad()
            fb  = self._to_device(next(forget_iter))
            lpc = self._seq_lp(self.model, fb)
            with torch.no_grad():
                lpr = self._seq_lp(self.ref_model, fb)
            npo = -F.logsigmoid(self.beta * (lpc - lpr.detach())).mean()

            rl = torch.tensor(0.0, device=self.device)
            if retain_iter:
                rb = self._to_device(next(retain_iter))
                rl = self.loss_fn(self.model, rb)

            (npo + self.retain_lambda * rl).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.gradient_clip
            )
            optimizer.step(); scheduler.step()
            self._log_step(step, pbar, {"npo": npo.item(), "retain": rl.item()})

        pbar.close()
        self.logger.info("NPO complete.")
