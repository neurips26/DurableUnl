"""RMU — Representation Misdirection for Unlearning (Li et al., 2024)."""
import copy
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .base import BaseUnlearner


class RMU(BaseUnlearner):
    def __init__(self, *args, steering_coef: float = 20.0,
                 alpha_rmu: float = 1200.0, layer_id: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.steering_coef = steering_coef
        self.alpha_rmu     = alpha_rmu
        layers             = self._get_layers(self.model)
        self.layer_id      = layer_id if layer_id is not None else len(layers) // 2
        self.logger.info(f"RMU: targeting layer {self.layer_id} / {len(layers)}")
        self.logger.info("RMU: Creating frozen reference model...")
        self.ref_model = copy.deepcopy(self.model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()
        self._target = None

    @staticmethod
    def _get_layers(model):
        """
        Robustly find transformer layers regardless of PEFT/LoRA wrapping.

        PEFT wraps the model as:
          model                        <- PeftModelForCausalLM
            .base_model               <- LoraModel
              .model                  <- LlamaForCausalLM
                .model                <- LlamaModel
                  .layers             <- ModuleList  ← we want this

        Plain HuggingFace:
          model                        <- LlamaForCausalLM
            .model                    <- LlamaModel
              .layers                 <- ModuleList  ← we want this
        """
        # Unwrap PEFT layers until we find something with .layers or .h
        base = model
        for _ in range(5):   # max 5 levels of unwrapping
            # LLaMA / Mistral style
            if hasattr(base, "layers"):
                return base.layers
            # GPT-2 / Falcon style
            if hasattr(base, "h"):
                return base.h
            # Go one level deeper
            if hasattr(base, "base_model"):
                base = base.base_model
            elif hasattr(base, "model"):
                base = base.model
            else:
                break

        raise ValueError(
            "Cannot auto-detect transformer layers. "
            f"Model type: {type(model).__name__}. "
            "Set layer_id explicitly if needed."
        )

    def _hook(self, store):
        def fn(m, inp, out):
            store.append(out[0] if isinstance(out, tuple) else out)
        return fn

    def run(self) -> None:
        self.logger.info(f"Starting RMU | steps={self.n_steps} | layer={self.layer_id}")
        self.model.train()
        layers  = self._get_layers(self.model)
        rlayers = self._get_layers(self.ref_model)
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_steps)

        forget_iter = self._infinite_loader(self.forget_loader)
        retain_iter = self._infinite_loader(self.retain_loader) if self.retain_loader else None
        pbar        = self._make_pbar("RMU")

        for step in range(1, self.n_steps + 1):
            optimizer.zero_grad()
            fb = self._to_device(next(forget_iter))

            hf = []
            h  = layers[self.layer_id].register_forward_hook(self._hook(hf))
            self.model(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"])
            h.remove()

            if hf:
                if self._target is None:
                    t = torch.randn_like(hf[0][0:1, 0:1, :])
                    self._target = t / t.norm() * self.steering_coef
                    self.logger.info(
                        f"RMU: steering target norm={self._target.norm().item():.2f}"
                    )
                ml = F.mse_loss(hf[0], self._target.expand_as(hf[0]).detach())
            else:
                ml = torch.tensor(0.0, device=self.device, requires_grad=True)

            rl = torch.tensor(0.0, device=self.device)
            if retain_iter:
                rb  = self._to_device(next(retain_iter))
                hr  = []
                hrf = []
                h1  = layers[self.layer_id].register_forward_hook(self._hook(hr))
                self.model(input_ids=rb["input_ids"], attention_mask=rb["attention_mask"])
                h1.remove()
                h2  = rlayers[self.layer_id].register_forward_hook(self._hook(hrf))
                with torch.no_grad():
                    self.ref_model(input_ids=rb["input_ids"], attention_mask=rb["attention_mask"])
                h2.remove()
                if hr and hrf:
                    rl = F.mse_loss(hr[0], hrf[0].detach())

            (ml + self.alpha_rmu * rl).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.gradient_clip,
            )
            optimizer.step()
            scheduler.step()
            self._log_step(step, pbar, {"misdirect": ml.item(), "retain": rl.item()})

        pbar.close()
        self.logger.info("RMU complete.")
