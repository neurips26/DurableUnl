"""
experiments/ste_augmented_baselines.py
=======================================
Tests STE-augmented versions of strong baselines (SalUn+STE, GA+STE).
Addresses reviewer: "testing STE-augmented strong baselines."

Key insight: If STE alone (without SAF's warmup schedule) also closes the
Q-INT4 gap for SalUn, that weakens our methodological contribution.
If it doesn't, it strengthens our claim that the warmup + full-model STE
is the critical design choice.

Usage:
  python experiments/ste_augmented_baselines.py \
      --config configs/base_config.yaml

Expected results:
  - SalUn+STE (no warmup): Q-INT4 may improve but FA likely degrades
  - GA+STE (no warmup): Q-INT4 may improve but FA likely worse than SAF
  - This validates that warmup + STE together are necessary

Runtime: ~40 min per method.
"""

import argparse, csv, json, logging, os, sys
import torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.logging_utils import setup_root_logger, now_str, file_ts
from src.data.data_utils import set_seed
from src.data.tofu_dataset import get_tofu_dataloaders
from src.models.model_utils import load_model_with_lora
from src.evaluation.evaluator import (
    compute_token_accuracy, compute_quantization_recovery, compute_mia_auc
)


# ─── STE quantisation (same as SAF) ─────────────────────────────────────────

class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.round()
    @staticmethod
    def backward(ctx, g): return g

def _ste_quant_int4(w):
    w_f = w.float()
    scale = (w_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
             if w_f.dim() >= 2 else w_f.abs().max().clamp(min=1e-8) / 7.0)
    return (_STE.apply(w_f / scale).clamp(-8, 7) * scale).to(w.dtype)

def _clm_loss_ste(model, batch, device):
    """CLM loss with STE-INT4 quantized weights on ALL linear layers."""
    import torch.nn as nn
    originals = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight is not None:
            originals[id(module)] = (module, module.weight.data.clone())
            module.weight.data = _ste_quant_int4(module.weight.data)
    try:
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch.get("labels", ids).to(device)
        loss   = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    finally:
        for _, (module, orig) in originals.items():
            module.weight.data = orig
    return loss


# ─── STE-augmented GA ────────────────────────────────────────────────────────

class GA_STE:
    """Gradient Ascent + STE quantization-aware term (no warmup, same α for all steps)."""
    def __init__(self, model, forget_loader, retain_loader, device,
                 n_steps=300, lr=5e-5, retain_lambda=2.0, alpha_ste=1.0):
        self.model          = model
        self.forget_loader  = forget_loader
        self.retain_loader  = retain_loader
        self.device         = device
        self.n_steps        = n_steps
        self.lr             = lr
        self.retain_lambda  = retain_lambda
        self.alpha_ste      = alpha_ste

    def _clm(self, batch):
        ids = batch["input_ids"].to(self.device)
        mask = batch["attention_mask"].to(self.device)
        labels = batch.get("labels", ids).to(self.device)
        return self.model(input_ids=ids, attention_mask=mask, labels=labels).loss

    def _infinite(self, loader):
        while True:
            for b in loader: yield b

    def run(self):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from tqdm import tqdm
        import logging as _logging
        logger = _logging.getLogger("GA_STE")
        logger.info(f"GA+STE | steps={self.n_steps} | alpha_ste={self.alpha_ste}")
        self.model.train()
        opt  = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)
        sch  = CosineAnnealingLR(opt, T_max=self.n_steps)
        fi   = self._infinite(self.forget_loader)
        ri   = self._infinite(self.retain_loader)
        pbar = tqdm(total=self.n_steps, desc="GA+STE")
        for step in range(1, self.n_steps + 1):
            opt.zero_grad()
            fb = {k: v.to(self.device) if hasattr(v,"to") else v for k, v in next(fi).items()}
            lf  = self._clm(fb)
            lq  = _clm_loss_ste(self.model, fb, self.device)
            rb  = {k: v.to(self.device) if hasattr(v,"to") else v for k, v in next(ri).items()}
            lr_ = self._clm(rb)
            total = -lf - self.alpha_ste * lq + self.retain_lambda * lr_
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0)
            opt.step(); sch.step()
            if step % 50 == 0:
                pbar.set_postfix({"f": f"{lf.item():.2f}", "q": f"{lq.item():.2f}"})
            pbar.update(1)
        pbar.close()
        logger.info("GA+STE complete.")


class SalUn_STE(GA_STE):
    """SalUn (gradient masking) + STE quantization-aware term."""

    def run(self):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from tqdm import tqdm
        import logging as _logging
        logger = _logging.getLogger("SalUn_STE")
        logger.info(f"SalUn+STE | steps={self.n_steps} | alpha_ste={self.alpha_ste}")

        # Compute saliency mask on forget set
        logger.info("Computing saliency mask...")
        self.model.eval()
        grad_accum = {}
        for b in self.forget_loader:
            self.model.zero_grad()
            fb = {k: v.to(self.device) if hasattr(v,"to") else v for k,v in b.items()}
            loss = self._clm(fb)
            loss.backward()
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if name not in grad_accum:
                        grad_accum[name] = p.grad.data.abs().clone()
                    else:
                        grad_accum[name] += p.grad.data.abs()
            break  # single batch for efficiency

        # Threshold at top-20% saliency
        all_grads = torch.cat([g.flatten() for g in grad_accum.values()])
        threshold = torch.quantile(all_grads, 0.80)
        masks = {name: (g >= threshold).float() for name, g in grad_accum.items()}
        self.model.train()

        # Training with masked gradients + STE
        opt  = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)
        sch  = CosineAnnealingLR(opt, T_max=self.n_steps)
        fi   = self._infinite(self.forget_loader)
        ri   = self._infinite(self.retain_loader)
        pbar = tqdm(total=self.n_steps, desc="SalUn+STE")

        for step in range(1, self.n_steps + 1):
            opt.zero_grad()
            fb = {k: v.to(self.device) if hasattr(v,"to") else v for k,v in next(fi).items()}
            lf  = self._clm(fb)
            lq  = _clm_loss_ste(self.model, fb, self.device)
            rb  = {k: v.to(self.device) if hasattr(v,"to") else v for k,v in next(ri).items()}
            lr_ = self._clm(rb)
            total = -lf - self.alpha_ste * lq + self.retain_lambda * lr_
            total.backward()

            # Apply saliency mask to gradients
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None and name in masks:
                    p.grad.data *= masks[name].to(p.grad.device)

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0)
            opt.step(); sch.step()
            if step % 50 == 0:
                pbar.set_postfix({"f": f"{lf.item():.2f}", "q": f"{lq.item():.2f}"})
            pbar.update(1)
        pbar.close()
        logger.info("SalUn+STE complete.")


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="configs/base_config.yaml")
    p.add_argument("--alpha",   type=float, default=1.0,
                   help="STE alpha weight for augmented baselines")
    p.add_argument("--methods", nargs="+", default=["ga_ste", "salun_ste"],
                   choices=["ga_ste","salun_ste"])
    return p.parse_args()


def load_config(path):
    with open(path) as f: return yaml.safe_load(f)


def _get(cfg, *keys, default=None):
    for k in keys:
        v = cfg
        try:
            for part in k.split("."): v = v[part]
            return v
        except (KeyError, TypeError): pass
    return default


def _real_device(model):
    for p in model.parameters():
        if p.device.type != "meta": return p.device
    return torch.device("cuda")


def main():
    args   = parse_args()
    config = load_config(args.config)

    setup_root_logger(_get(config, "paths.logs", default="logs"))
    logger = logging.getLogger("ste_augmented")
    os.makedirs(_get(config, "paths.results", default="results"), exist_ok=True)
    results_csv = os.path.join(
        _get(config, "paths.results", default="results"),
        f"ste_augmented_{file_ts()}.csv"
    )

    set_seed(42)

    for method_name in args.methods:
        logger.info(f"\n{'='*50}")
        logger.info(f"  {method_name.upper()} | alpha_ste={args.alpha}")
        logger.info(f"{'='*50}")

        model, tokenizer = load_model_with_lora(
            _get(config, "model.name", default="meta-llama/Meta-Llama-3-8B-Instruct"),
            lora_config=config.get("lora"),
            dtype=_get(config, "model.dtype", default="bfloat16"),
            device_map=_get(config, "model.device_map", default="cuda:0"),
            load_in_4bit=_get(config, "model.load_in_4bit", default=True),
        )
        device = _real_device(model)

        fl, rl, _ = get_tofu_dataloaders(
            tokenizer,
            forget_split=_get(config, "dataset.forget_split", default="forget10"),
            retain_split=_get(config, "dataset.retain_split", default="retain90"),
            batch_size=4, max_length=256, num_workers=0,
        )

        retain_lambda = max(2.0, args.alpha + 1.0)

        if method_name == "ga_ste":
            unlearner = GA_STE(model=model, forget_loader=fl, retain_loader=rl,
                                device=device, n_steps=300, lr=5e-5,
                                retain_lambda=retain_lambda, alpha_ste=args.alpha)
        elif method_name == "salun_ste":
            unlearner = SalUn_STE(model=model, forget_loader=fl, retain_loader=rl,
                                   device=device, n_steps=300, lr=5e-5,
                                   retain_lambda=retain_lambda, alpha_ste=args.alpha)
        unlearner.run()

        dev   = str(device)
        max_b = 30
        fa    = compute_token_accuracy(model, fl, dev, max_b)
        ra    = compute_token_accuracy(model, rl, dev, max_b)
        mia   = compute_mia_auc(model, fl, rl, dev)
        quant = compute_quantization_recovery(model, fl, dev, ["bf16","int8","int4"], max_b)

        row = {
            "method": method_name, "alpha_ste": args.alpha,
            "retain_lambda": retain_lambda, "seed": 42,
            "forget_acc": round(fa, 4), "retain_acc": round(ra, 4),
            "mia_auc": round(mia, 4),
            "quant_bf16": round(quant.get("bf16",-1), 4),
            "quant_int8": round(quant.get("int8",-1), 4),
            "quant_int4": round(quant.get("int4",-1), 4),
        }

        logger.info(f"  FA={fa:.4f}  RA={ra:.4f}  Q_INT4={quant.get('int4',-1):.4f}")

        write_hdr = not os.path.exists(results_csv)
        with open(results_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
            if write_hdr: w.writeheader()
            w.writerow(row)

        del model; torch.cuda.empty_cache()

    # Print comparison
    logger.info("\n" + "="*55)
    logger.info("STE-AUGMENTED BASELINE COMPARISON")
    logger.info("="*55)
    logger.info(f"{'Method':<16} {'FA↓':>7} {'RA↑':>7} {'Q_INT4↓':>9}")
    logger.info("-"*42)
    refs = [("GA",0.028,0.521,0.262),("SalUn",0.011,0.541,0.051),
            ("DurableUn-SAF",0.008,0.495,0.239)]
    for n,fa,ra,qi4 in refs:
        logger.info(f"  {n:<14} {fa:>7.4f} {ra:>7.4f} {qi4:>9.4f}  ← baseline")
    logger.info(f"  Results saved: {results_csv}")


if __name__ == "__main__":
    import logging
    main()
