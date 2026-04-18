"""
Evaluation harness for DurableUn Phase 0.
Computes: Forget Acc, Retain Acc, MIA AUC, Quant Recovery, FT Recovery.
"""

import copy
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Token accuracy
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> float:
    """Fraction of non-masked tokens predicted correctly (next-token accuracy)."""
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Token Acc", leave=False)):
            if max_batches and i >= max_batches:
                break
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl  = batch["labels"].to(device)

            logits = model(input_ids=ids, attention_mask=mask).logits
            pred   = logits[:, :-1, :].argmax(dim=-1)   # (B, L-1)
            shift  = lbl[:, 1:]                           # (B, L-1)
            valid  = shift != -100

            correct += (pred[valid] == shift[valid]).sum().item()
            total   += valid.sum().item()

    return correct / max(total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MIA — loss-based membership inference
# ─────────────────────────────────────────────────────────────────────────────

def compute_mia_auc(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    device: str = "cuda",
    n_samples: int = 200,
) -> float:
    """
    Simple loss-based MIA. Lower loss → member.
    AUC ~0.5 = model has forgotten (good). AUC > 0.5 = model still remembers.
    """
    model.eval()

    def _get_losses(loader, n):
        losses = []
        with torch.no_grad():
            for batch in loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lbl  = batch["labels"].to(device)

                out    = model(input_ids=ids, attention_mask=mask, labels=lbl)
                B      = ids.shape[0]
                # Per-sample loss approximation from the batch loss
                losses.extend([out.loss.item()] * B)
                if len(losses) >= n:
                    break
        return losses[:n]

    f_losses = _get_losses(forget_loader, n_samples // 2)
    r_losses = _get_losses(retain_loader, n_samples // 2)

    # Member (forget) = lower loss → higher score = -loss
    scores = [-l for l in f_losses] + [-l for l in r_losses]
    labels = [1] * len(f_losses)   + [0] * len(r_losses)

    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 3. Quantization recovery attack
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_quantize(model: nn.Module, precision: str) -> nn.Module:
    """
    Return a deep copy of model with weights quantize-dequantized.
    Simulates the rounding error of INT8 / INT4 / BF16 deployment.
    """
    q_model = copy.deepcopy(model)
    with torch.no_grad():
        for module in q_model.modules():
            if not (hasattr(module, "weight") and module.weight is not None):
                continue
            w = module.weight.data.float()

            if precision == "bf16":
                wq = w.to(torch.bfloat16).float()
            elif precision in ("int8", "int8_absmax"):
                scale = w.abs().max().clamp(min=1e-8) / 127.0
                wq    = (w / scale).round().clamp(-128, 127) * scale
            elif precision in ("int4", "nf4"):
                # Per-row INT4 symmetric
                if w.dim() >= 2:
                    scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
                else:
                    scale = w.abs().max().clamp(min=1e-8) / 7.0
                wq = (w / scale).round().clamp(-8, 7) * scale
            else:
                wq = w

            module.weight.data.copy_(wq.to(module.weight.dtype))
    return q_model


def compute_quantization_recovery(
    model: nn.Module,
    forget_loader: DataLoader,
    device: str = "cuda",
    precisions: Optional[List[str]] = None,
    max_batches: int = 20,
) -> Dict[str, float]:
    """
    For each precision: quantize model copy → measure forget accuracy.
    High value = quantization restored the memory (bad).
    """
    if precisions is None:
        precisions = ["bf16", "int8", "int4"]

    results = {}
    for prec in precisions:
        logger.info(f"    Quant attack @ {prec}...")
        q_model = _simulate_quantize(model, prec)
        q_model.to(device)
        acc = compute_token_accuracy(q_model, forget_loader, device, max_batches)
        results[prec] = acc
        logger.info(f"    forget_acc@{prec} = {acc:.4f}")
        del q_model
        torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fine-tuning recovery attack
# ─────────────────────────────────────────────────────────────────────────────

def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def compute_finetuning_recovery(
    model: nn.Module,
    tokenizer,
    forget_loader: DataLoader,
    finetune_loader: DataLoader,
    device: str = "cuda",
    steps_list: Optional[List[int]] = None,
    lr: float = 2e-5,
    max_eval_batches: int = 20,
) -> Dict[int, float]:
    """
    Fine-tune unlearned model on unrelated data, measure forget accuracy.
    High value = fine-tuning restored memory (bad).
    """
    if steps_list is None:
        steps_list = [50, 100, 500]

    model_copy = copy.deepcopy(model)
    model_copy.to(device)
    model_copy.train()

    optimizer    = AdamW(
        [p for p in model_copy.parameters() if p.requires_grad], lr=lr
    )
    ft_iter      = _infinite(finetune_loader)
    results      = {}
    current_step = 0

    for target in sorted(steps_list):
        while current_step < target:
            batch = next(ft_iter)
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl  = batch.get("labels", ids).to(device)

            optimizer.zero_grad()
            out = model_copy(input_ids=ids, attention_mask=mask, labels=lbl)
            out.loss.backward()
            optimizer.step()
            current_step += 1

        model_copy.eval()
        acc = compute_token_accuracy(model_copy, forget_loader, device, max_eval_batches)
        results[target] = acc
        logger.info(f"    FT@{target} steps: forget_acc = {acc:.4f}")
        model_copy.train()

    del model_copy
    torch.cuda.empty_cache()
    return results
