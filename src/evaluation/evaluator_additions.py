"""
src/evaluation/evaluator_additions.py
======================================
Add these functions to your existing src/evaluation/evaluator.py
OR import them from here alongside existing evaluator functions.

Adds:
  1. compute_token_accuracy_quantized()  — RA-INT4 metric
  2. compute_full_eval()                 — single call for all metrics
  3. compute_dataset_generalization()    — second dataset validation
"""

import logging
import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# INT4 quantization (matches paper Eq.1)
# ─────────────────────────────────────────────────────────────────────────────

def _quant_int4_inplace(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Apply symmetric per-row INT4 quantization. Returns originals for restoration."""
    originals = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight is not None:
            w = module.weight.data.float()
            if w.dim() >= 2:
                scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
            else:
                scale = w.abs().max().clamp(min=1e-8) / 7.0
            originals[name] = module.weight.data.clone()
            module.weight.data = (torch.round(w / scale).clamp(-8, 7) * scale).to(module.weight.dtype)
    return originals


def _restore_weights(model: nn.Module, originals: Dict[str, torch.Tensor]):
    """Restore weights after quantization evaluation."""
    for name, module in model.named_modules():
        if name in originals:
            module.weight.data = originals[name]


# ─────────────────────────────────────────────────────────────────────────────
# RA-INT4: Retain accuracy AFTER INT4 quantization (new metric)
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_accuracy_quantized(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    precision: str,
    max_batches: int = 30,
) -> float:
    """
    Compute token accuracy after applying quantization.
    
    Used for RA-INT4: retain accuracy under deployment quantization.
    A key finding: for baselines, RA-INT4 ≈ RA (quantization doesn't harm retain).
    For DurableUn-SAF α=3, RA-INT4 ≈ RA ≈ 0.045 (the trilemma).
    
    Args:
        model:      Unlearned model to evaluate.
        dataloader: DataLoader (forget set for Q-INT4, retain set for RA-INT4).
        device:     Device string.
        precision:  'int4', 'int8', or 'bf16'.
        max_batches: Max batches to evaluate.
    
    Returns:
        Token-level accuracy after quantization.
    """
    model.eval()
    
    if precision == 'bf16':
        # No quantization needed - just evaluate normally
        return _compute_accuracy_no_quant(model, dataloader, device, max_batches)
    
    if precision == 'int4':
        originals = _quant_int4_inplace(model)
        acc = _compute_accuracy_no_quant(model, dataloader, device, max_batches)
        _restore_weights(model, originals)
        return acc
    
    if precision == 'int8':
        originals = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight is not None:
                w = module.weight.data.float()
                scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
                originals[name] = module.weight.data.clone()
                module.weight.data = (torch.round(w / scale).clamp(-128, 127) * scale).to(module.weight.dtype)
        acc = _compute_accuracy_no_quant(model, dataloader, device, max_batches)
        _restore_weights(model, originals)
        return acc
    
    raise ValueError(f"Unknown precision: {precision}. Use 'bf16', 'int8', or 'int4'.")


def _compute_accuracy_no_quant(model, dataloader, device, max_batches):
    """Token accuracy without any quantization modification."""
    correct = total = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch.get("labels", ids).to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            valid = (shift_labels != -100) & mask[:, 1:].bool()
            preds = shift_logits.argmax(dim=-1)
            correct += (preds[valid] == shift_labels[valid]).sum().item()
            total   += valid.sum().item()
    return correct / max(total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation: all metrics in one call
# ─────────────────────────────────────────────────────────────────────────────

def compute_full_eval(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    device: str,
    max_batches: int = 30,
    compute_ra_quantized: bool = True,
) -> Dict[str, float]:
    """
    Compute all metrics in a single function call.
    
    Returns dict with keys:
      forget_acc, retain_acc, mia_auc,
      quant_bf16, quant_int8, quant_int4,
      ra_int4, ra_int8  (if compute_ra_quantized=True)
    
    Usage:
        from src.evaluation.evaluator import compute_token_accuracy, compute_quantization_recovery, compute_mia_auc
        from src.evaluation.evaluator_additions import compute_full_eval
        
        metrics = compute_full_eval(model, forget_loader, retain_loader, device)
    """
    from src.evaluation.evaluator import (
        compute_token_accuracy, compute_quantization_recovery, compute_mia_auc
    )
    
    results = {}
    
    # Standard metrics
    results["forget_acc"] = round(compute_token_accuracy(model, forget_loader, device, max_batches), 4)
    results["retain_acc"] = round(compute_token_accuracy(model, retain_loader, device, max_batches), 4)
    results["mia_auc"]    = round(compute_mia_auc(model, forget_loader, retain_loader, device), 4)
    
    logger.info(f"  FA={results['forget_acc']:.4f}  RA={results['retain_acc']:.4f}  MIA={results['mia_auc']:.4f}")
    
    # Quantization recovery (forget set)
    quant = compute_quantization_recovery(model, forget_loader, device,
                                          ["bf16", "int8", "int4"], max_batches)
    results["quant_bf16"] = round(quant.get("bf16", -1), 4)
    results["quant_int8"] = round(quant.get("int8", -1), 4)
    results["quant_int4"] = round(quant.get("int4", -1), 4)
    
    logger.info(f"  Q-BF16={results['quant_bf16']:.4f}  Q-INT8={results['quant_int8']:.4f}  Q-INT4={results['quant_int4']:.4f}")
    
    # RA under quantization (retain set) — new metric for reviewer
    if compute_ra_quantized:
        results["ra_int4"] = round(
            compute_token_accuracy_quantized(model, retain_loader, device, "int4", max_batches), 4
        )
        results["ra_int8"] = round(
            compute_token_accuracy_quantized(model, retain_loader, device, "int8", max_batches), 4
        )
        logger.info(f"  RA-INT8={results['ra_int8']:.4f}  RA-INT4={results['ra_int4']:.4f}")
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Second dataset evaluation (generalization check)
# ─────────────────────────────────────────────────────────────────────────────

def load_wikitext_forget_proxy(
    tokenizer,
    n_samples: int = 200,
    max_length: int = 256,
    batch_size: int = 4,
    seed: int = 42,
) -> DataLoader:
    """
    Load WikiText-2 as a proxy 'forget set' for second-dataset generalization.
    
    This tests a different scenario: after unlearning on TOFU, does the model
    still behave normally on unrelated text?
    
    Used in Table 3 (generalization evaluation):
    - High RA on WikiText = model didn't over-forget general knowledge
    - Q-INT4 on WikiText ≈ BF16 = no INT4 effect on general text (expected)
    """
    from datasets import load_dataset
    import random
    
    random.seed(seed)
    
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [row["text"] for row in dataset if len(row["text"].strip()) > 100]
        random.shuffle(texts)
        texts = texts[:n_samples]
        
        encodings = tokenizer(
            texts, truncation=True, max_length=max_length,
            padding="max_length", return_tensors="pt",
        )
        labels = encodings["input_ids"].clone()
        labels[encodings["attention_mask"] == 0] = -100
        
        dataset_obj = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            labels,
        )
        
        def collate_fn(batch):
            ids, mask, lbl = zip(*batch)
            return {
                "input_ids":      torch.stack(ids),
                "attention_mask": torch.stack(mask),
                "labels":         torch.stack(lbl),
            }
        
        return DataLoader(dataset_obj, batch_size=batch_size,
                          shuffle=False, collate_fn=collate_fn)
    
    except Exception as e:
        logger.warning(f"Failed to load WikiText-2: {e}")
        return None


def load_tofu_forget05(tokenizer, max_length=256, batch_size=4):
    """Load TOFU forget05 split for additional validation."""
    from src.data.tofu_dataset import get_tofu_dataloaders
    fl, rl, _ = get_tofu_dataloaders(
        tokenizer,
        forget_split="forget05",
        retain_split="retain95",
        batch_size=batch_size,
        max_length=max_length,
        num_workers=0,
    )
    return fl, rl


def compute_dataset_generalization(
    model: nn.Module,
    tokenizer,
    device: str,
    max_batches: int = 20,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate generalization across two additional datasets.
    
    Returns dict:
      'tofu_forget05': {forget_acc, quant_int4}  — smaller forget split
      'wikitext':      {retain_acc, ra_int4}      — general knowledge
    """
    results = {}
    
    # TOFU forget05 — smaller forget split, tests robustness to split size
    try:
        fl05, rl05 = load_tofu_forget05(tokenizer)
        fa05  = _compute_accuracy_no_quant(model, fl05, device, max_batches)
        qi4_05 = compute_token_accuracy_quantized(model, fl05, device, "int4", max_batches)
        results["tofu_forget05"] = {
            "forget_acc":  round(fa05, 4),
            "quant_int4":  round(qi4_05, 4),
        }
        logger.info(f"  TOFU forget05: FA={fa05:.4f}  Q-INT4={qi4_05:.4f}")
    except Exception as e:
        logger.warning(f"TOFU forget05 evaluation failed: {e}")
        results["tofu_forget05"] = {"forget_acc": -1, "quant_int4": -1}
    
    # WikiText — general knowledge preservation
    wt_loader = load_wikitext_forget_proxy(tokenizer, max_length=256)
    if wt_loader is not None:
        wt_acc  = _compute_accuracy_no_quant(model, wt_loader, device, max_batches)
        wt_qi4  = compute_token_accuracy_quantized(model, wt_loader, device, "int4", max_batches)
        results["wikitext"] = {
            "retain_acc": round(wt_acc, 4),
            "ra_int4":    round(wt_qi4, 4),
        }
        logger.info(f"  WikiText: RA={wt_acc:.4f}  RA-INT4={wt_qi4:.4f}")
    else:
        results["wikitext"] = {"retain_acc": -1, "ra_int4": -1}
    
    return results
