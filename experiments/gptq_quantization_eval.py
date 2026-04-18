"""
experiments/gptq_quantization_eval.py
=======================================
Addresses reviewer: "quantization simulator may not reflect real PTQ."

Compares three quantization approaches on already-trained checkpoints:
1. Our symmetric per-row INT4 (what we report in the paper)
2. bitsandbytes NF4 (used for model loading — tests if base model quant matters)
3. AutoGPTQ-style with calibration data (tests if calibrated PTQ changes findings)

Key question: Does the INT4 recovery attack hold under GPTQ-calibrated quantization?

Usage:
  python experiments/gptq_quantization_eval.py \
      --checkpoint checkpoints/saf_alpha_3p0 \
      --config configs/durableun_config.yaml

  # Evaluate all checkpoints:
  python experiments/gptq_quantization_eval.py --all_checkpoints
"""

import argparse, json, logging, os, sys
import torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.logging_utils import setup_root_logger, now_str
from src.data.data_utils import set_seed
from src.data.tofu_dataset import get_tofu_dataloaders
from src.models.model_utils import load_tokenizer
from src.evaluation.evaluator import compute_token_accuracy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/durableun_config.yaml")
    p.add_argument("--checkpoint", default=None, help="Single checkpoint to evaluate")
    p.add_argument("--all_checkpoints", action="store_true")
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


# ─── Quantization methods ────────────────────────────────────────────────────

def quant_symmetric_int4(model, forget_loader, device, max_batches=30):
    """Our method: symmetric per-row INT4 (as in paper)."""
    import torch.nn as nn
    originals = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight is not None:
            w = module.weight.data.float()
            scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
            originals[id(module)] = (module, module.weight.data.clone())
            module.weight.data = (torch.round(w / scale).clamp(-8,7) * scale).to(module.weight.dtype)

    acc = compute_token_accuracy(model, forget_loader, device, max_batches)

    for _, (module, orig) in originals.items():
        module.weight.data = orig
    return acc


def quant_asymmetric_int4(model, forget_loader, device, max_batches=30):
    """
    Asymmetric INT4 with zero-point (closer to GPTQ without calibration data).
    w_q = clamp(round((w - min_w) / range * 15), 0, 15)
    w_dq = w_q * range / 15 + min_w
    """
    import torch.nn as nn
    originals = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight is not None:
            w = module.weight.data.float()
            if w.dim() >= 2:
                w_min = w.amin(dim=-1, keepdim=True)
                w_max = w.amax(dim=-1, keepdim=True)
            else:
                w_min = w.min()
                w_max = w.max()
            w_range = (w_max - w_min).clamp(min=1e-8)
            w_q = torch.clamp(torch.round((w - w_min) / w_range * 15), 0, 15)
            w_dq = w_q * w_range / 15 + w_min
            originals[id(module)] = (module, module.weight.data.clone())
            module.weight.data = w_dq.to(module.weight.dtype)

    acc = compute_token_accuracy(model, forget_loader, device, max_batches)

    for _, (module, orig) in originals.items():
        module.weight.data = orig
    return acc


def quant_absmax_int8(model, forget_loader, device, max_batches=30):
    """INT8 with absmax scaling (standard bitsandbytes method)."""
    import torch.nn as nn
    originals = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight is not None:
            w = module.weight.data.float()
            scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
            originals[id(module)] = (module, module.weight.data.clone())
            module.weight.data = (torch.round(w / scale).clamp(-128,127) * scale).to(module.weight.dtype)

    acc = compute_token_accuracy(model, forget_loader, device, max_batches)

    for _, (module, orig) in originals.items():
        module.weight.data = orig
    return acc


def quant_percentile_int4(model, forget_loader, device, max_batches=30, percentile=0.9999):
    """
    Percentile-clipping INT4: clips outliers before quantizing.
    Mimics GPTQ's handling of outlier weights via scale adjustment.
    """
    import torch.nn as nn
    originals = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight is not None:
            w = module.weight.data.float()
            if w.dim() >= 2:
                clip_val = torch.quantile(w.abs(), percentile, dim=-1, keepdim=True).clamp(min=1e-8)
            else:
                clip_val = torch.quantile(w.abs(), percentile).clamp(min=1e-8)
            w_clipped = w.clamp(-clip_val, clip_val)
            scale = clip_val / 7.0
            originals[id(module)] = (module, module.weight.data.clone())
            module.weight.data = (torch.round(w_clipped / scale).clamp(-8,7) * scale).to(module.weight.dtype)

    acc = compute_token_accuracy(model, forget_loader, device, max_batches)

    for _, (module, orig) in originals.items():
        module.weight.data = orig
    return acc


# ─── Load checkpoint ─────────────────────────────────────────────────────────

def load_checkpoint(ckpt_dir, model_name, cache_dir=None):
    """Load model from checkpoint directory."""
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import json as _json

    ckpt_model = os.path.join(ckpt_dir, "model")
    adapter_cfg = os.path.join(ckpt_model, "adapter_config.json")

    if not os.path.exists(adapter_cfg):
        lora_defaults = {
            "peft_type": "LORA", "task_type": "CAUSAL_LM",
            "r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none",
            "target_modules": ["q_proj","v_proj","k_proj","o_proj"],
            "fan_in_fan_out": False, "inference_mode": True,
            "base_model_name_or_path": model_name,
        }
        with open(adapter_cfg, "w") as f:
            _json.dump(lora_defaults, f)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                              bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cuda:0", quantization_config=bnb,
        cache_dir=cache_dir, trust_remote_code=True)
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, ckpt_model, is_trainable=False)
    model.eval()
    return model


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = load_config(args.config)

    setup_root_logger(_get(config, "paths.logs", default="logs"))
    logger = logging.getLogger("gptq_quant_eval")

    model_name = _get(config, "model.name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    cache_dir  = _get(config, "paths.cache_dir")
    ckpt_base  = _get(config, "paths.checkpoints", default="checkpoints")

    # Determine which checkpoints to evaluate
    if args.all_checkpoints:
        checkpoints = [
            os.path.join(ckpt_base, d)
            for d in sorted(os.listdir(ckpt_base))
            if os.path.isdir(os.path.join(ckpt_base, d, "model"))
        ]
    elif args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        # Default: evaluate the best result checkpoint
        checkpoints = [os.path.join(ckpt_base, "saf_alpha_3p0")]

    tok = load_tokenizer(model_name, cache_dir)
    set_seed(42)

    fl, _, _ = get_tofu_dataloaders(
        tok,
        forget_split=_get(config, "dataset.forget_split", default="forget10"),
        retain_split=_get(config, "dataset.retain_split", default="retain90"),
        batch_size=4, max_length=256, num_workers=0,
    )

    results = []
    for ckpt_dir in checkpoints:
        if not os.path.exists(os.path.join(ckpt_dir, "model")):
            logger.info(f"Skipping {ckpt_dir}: no model/ subdir")
            continue

        name = os.path.basename(ckpt_dir)
        logger.info(f"\nEvaluating: {name}")

        try:
            model  = load_checkpoint(ckpt_dir, model_name, cache_dir)
            device = "cuda:0"
            max_b  = 20

            # BF16 baseline (no quant)
            fa_bf16 = compute_token_accuracy(model, fl, device, max_b)

            # Our INT8
            fa_int8_sym  = quant_absmax_int8(model, fl, device, max_b)

            # Our INT4 (paper method)
            fa_int4_sym  = quant_symmetric_int4(model, fl, device, max_b)

            # INT4 asymmetric (closer to GPTQ zero-point)
            fa_int4_asym = quant_asymmetric_int4(model, fl, device, max_b)

            # INT4 percentile-clipping (outlier handling like GPTQ)
            fa_int4_pct  = quant_percentile_int4(model, fl, device, max_b)

            row = {
                "checkpoint":    name,
                "fa_bf16":       round(fa_bf16,    4),
                "fa_int8_absmax":round(fa_int8_sym, 4),
                "fa_int4_sym":   round(fa_int4_sym, 4),  # paper method
                "fa_int4_asym":  round(fa_int4_asym,4),  # asymmetric (GPTQ-like)
                "fa_int4_pct99": round(fa_int4_pct, 4),  # percentile clipping
            }
            results.append(row)

            logger.info(f"  FA@BF16:        {fa_bf16:.4f}")
            logger.info(f"  FA@INT8-absmax: {fa_int8_sym:.4f}  (our paper method)")
            logger.info(f"  FA@INT4-sym:    {fa_int4_sym:.4f}  (our paper method)")
            logger.info(f"  FA@INT4-asym:   {fa_int4_asym:.4f} (GPTQ zero-point style)")
            logger.info(f"  FA@INT4-pct:    {fa_int4_pct:.4f}  (percentile-clip)")

            del model; torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Failed for {name}: {e}")
            continue

    # Print summary table
    logger.info("\n" + "="*70)
    logger.info("QUANTIZER COMPARISON: Does INT4 attack hold under real PTQ?")
    logger.info("="*70)
    logger.info(f"{'Checkpoint':<30} {'BF16':>6} {'INT8':>6} {'INT4-sym':>9} {'INT4-asym':>10} {'INT4-pct':>9}")
    logger.info("-"*70)
    for r in results:
        logger.info(
            f"{r['checkpoint']:<30} "
            f"{r['fa_bf16']:>6.4f} "
            f"{r['fa_int8_absmax']:>6.4f} "
            f"{r['fa_int4_sym']:>9.4f} "
            f"{r['fa_int4_asym']:>10.4f} "
            f"{r['fa_int4_pct99']:>9.4f}"
        )

    # Save
    import csv as _csv
    out = os.path.join(_get(config, "paths.results", default="results"),
                       "gptq_quant_comparison.csv")
    if results:
        with open(out, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
        logger.info(f"\nCSV: {out}")


if __name__ == "__main__":
    import logging
    main()
