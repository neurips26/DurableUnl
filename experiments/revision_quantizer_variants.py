"""
experiments/revision_quantizer_variants.py
============================================
Reviewer ask: "quantization simulation omits realistic PTQ/QAT configurations."

Tests four quantizer variants on all saved TOFU checkpoints:
  1. Symmetric per-row INT4        — paper method (conservative)
  2. Asymmetric zero-point INT4    — closer to GPTQ/bitsandbytes
  3. Percentile-clipped INT4       — handles outliers like AWQ
  4. Group-wise INT4 (g=128)       — group quantization like GPTQ default

Key expected finding: the INT4 attack holds under all quantizers.
The asymmetric quantizer may give slightly lower Q-INT4 (less quantization
error due to better scale fitting), so our symmetric simulator is conservative.

Usage:
  py -m experiments.revision_quantizer_variants

No training — evaluates existing checkpoints only. Fast (~5 min total).
"""

import json, logging, os, sys, csv
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.logging_utils import setup_root_logger, now_str, file_ts

logger = logging.getLogger(__name__)


# ── Quantizer implementations ─────────────────────────────────────────────────

def _symmetric_int4(w: torch.Tensor) -> torch.Tensor:
    """Paper method: symmetric per-row scale."""
    w = w.float()
    scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0 \
        if w.dim() >= 2 else w.abs().max().clamp(min=1e-8) / 7.0
    return (torch.round(w / scale).clamp(-8, 7) * scale).to(torch.bfloat16)


def _asymmetric_zp_int4(w: torch.Tensor) -> torch.Tensor:
    """Asymmetric zero-point INT4 (closer to GPTQ/bitsandbytes NF4)."""
    w = w.float()
    if w.dim() >= 2:
        w_min = w.amin(dim=-1, keepdim=True)
        w_max = w.amax(dim=-1, keepdim=True)
    else:
        w_min, w_max = w.min(), w.max()
    w_range = (w_max - w_min).clamp(min=1e-8)
    # Quantize to [0, 15] then dequantize
    w_q   = torch.clamp(torch.round((w - w_min) / w_range * 15), 0, 15)
    w_dq  = w_q * w_range / 15 + w_min
    return w_dq.to(torch.bfloat16)


def _percentile_int4(w: torch.Tensor, pct: float = 0.9999) -> torch.Tensor:
    """Percentile-clipped INT4 (handles outliers like AWQ)."""
    w = w.float()
    if w.dim() >= 2:
        clip = torch.quantile(w.abs(), pct, dim=-1, keepdim=True).clamp(min=1e-8)
    else:
        clip = torch.quantile(w.abs(), pct).clamp(min=1e-8)
    w_clipped = w.clamp(-clip, clip)
    scale = clip / 7.0
    return (torch.round(w_clipped / scale).clamp(-8, 7) * scale).to(torch.bfloat16)


def _groupwise_int4(w: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Group-wise INT4 (GPTQ default: g=128)."""
    w = w.float()
    if w.dim() < 2 or w.shape[-1] < group_size:
        return _symmetric_int4(w)
    orig_shape = w.shape
    # Reshape to (..., n_groups, group_size)
    cols = w.shape[-1]
    n_groups = cols // group_size
    w_trunc = w[..., :n_groups * group_size]
    w_r = w_trunc.reshape(*w.shape[:-1], n_groups, group_size)
    scale = w_r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
    w_q = (torch.round(w_r / scale).clamp(-8, 7) * scale)
    w_out = w_q.reshape(*w.shape[:-1], n_groups * group_size)
    # Handle remainder
    if cols > n_groups * group_size:
        remainder = _symmetric_int4(w[..., n_groups * group_size:])
        w_out = torch.cat([w_out, remainder], dim=-1)
    return w_out.to(torch.bfloat16)


# ── Apply quantizer to model, evaluate, restore ──────────────────────────────

def _quant_eval(model, forget_loader, device, quantizer_fn, max_batches=25):
    """Apply quantizer to all Linear weights, eval FA, restore."""
    originals = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight is not None:
            try:
                orig = module.weight.data.clone()
                originals[name] = orig
                orig_dtype = orig.dtype
                quantized = quantizer_fn(module.weight.data)
                # Restore original dtype — prevents float/bfloat16 mismatch
                module.weight.data = quantized.to(orig_dtype)
            except Exception:
                pass  # Skip layers that fail (e.g., meta tensors)

    correct = total = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(forget_loader):
            if i >= max_batches: break
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch.get("labels", ids).to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            valid  = (shift_labels != -100) & mask[:, 1:].bool()
            preds  = shift_logits.argmax(dim=-1)
            correct += (preds[valid] == shift_labels[valid]).sum().item()
            total   += valid.sum().item()

    for name, module in model.named_modules():
        if name in originals:
            module.weight.data = originals[name]

    return correct / max(total, 1)


# ── Load checkpoint ───────────────────────────────────────────────────────────

def load_ckpt(ckpt_dir, model_name, cache_dir=None):
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    ckpt_model = os.path.join(ckpt_dir, "model")
    adapter_cfg = os.path.join(ckpt_model, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        with open(adapter_cfg, "w") as f:
            json.dump({
                "peft_type":"LORA","task_type":"CAUSAL_LM",
                "r":16,"lora_alpha":32,"lora_dropout":0.05,"bias":"none",
                "target_modules":["q_proj","v_proj","k_proj","o_proj"],
                "fan_in_fan_out":False,"inference_mode":True,
                "base_model_name_or_path": model_name,
            }, f)

    bnb  = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                               bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cuda:0", quantization_config=bnb,
        cache_dir=cache_dir, trust_remote_code=True)
    base.config.use_cache = False
    base  = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, ckpt_model, is_trainable=False)
    model.eval()
    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import yaml
    setup_root_logger("logs")
    logger = logging.getLogger("revision_quantizer_variants")

    cfg_path   = os.path.join(ROOT, "configs", "base_config.yaml")
    with open(cfg_path) as f: cfg = yaml.safe_load(f)

    def _get(cfg, *keys, default=None):
        for k in keys:
            v = cfg
            try:
                for part in k.split("."): v = v[part]
                return v
            except: pass
        return default

    model_name = _get(cfg, "model.name",
                      default="meta-llama/Meta-Llama-3-8B-Instruct")
    ckpt_base  = _get(cfg, "paths.checkpoints", default="checkpoints")
    cache_dir  = _get(cfg, "paths.cache_dir")
    res_dir    = _get(cfg, "paths.results", default="results")
    os.makedirs(res_dir, exist_ok=True)

    from src.models.model_utils import load_tokenizer
    from src.data.tofu_dataset import get_tofu_dataloaders

    tok = load_tokenizer(model_name, cache_dir)
    fl, _, _ = get_tofu_dataloaders(
        tok, forget_split="forget10", retain_split="retain90",
        batch_size=4, max_length=256, num_workers=0,
    )
    device = "cuda:0"

    # Which checkpoints to evaluate
    checkpoints = {
        "GA":           os.path.join(ckpt_base, "ga_tofu_s42"),
        "SalUn":        os.path.join(ckpt_base, "salun_tofu_s42"),
        "GradDiff":     os.path.join(ckpt_base, "graddiff_tofu_s42"),
        "SAF_alpha3":   os.path.join(ckpt_base, "saf_alpha3p0_tofu_s42"),
    }

    quantizers = {
        "Sym-INT4 (paper)":   _symmetric_int4,
        "Asym-ZP-INT4":       _asymmetric_zp_int4,
        "Pct-INT4 (99.99%)":  _percentile_int4,
        "GroupW-INT4 (g=128)":lambda w: _groupwise_int4(w, 128),
    }

    csv_path = os.path.join(res_dir, f"revision_quantizers_{file_ts()}.csv")
    all_rows = []

    for method_name, ckpt_dir in checkpoints.items():
        if not os.path.exists(os.path.join(ckpt_dir, "model")):
            logger.info(f"SKIP {method_name}: checkpoint not found at {ckpt_dir}")
            continue

        logger.info(f"\n{'='*55}\n  {method_name}\n{'='*55}")
        try:
            model = load_ckpt(ckpt_dir, model_name, cache_dir)
        except Exception as e:
            logger.error(f"Failed to load {method_name}: {e}")
            continue

        row = {"method": method_name}

        # BF16 baseline (no quantization)
        fa_bf16 = _quant_eval(model, fl, device, lambda w: w)
        row["fa_bf16"] = round(fa_bf16, 4)
        logger.info(f"  FA@BF16: {fa_bf16:.4f}")

        for qname, qfn in quantizers.items():
            fa_q = _quant_eval(model, fl, device, qfn)
            key  = qname.replace(" ", "_").replace("(","").replace(")","").replace("%","").replace(".","p").replace("-","_")
            row[f"fa_{key}"] = round(fa_q, 4)
            cert = "Y" if fa_q <= 0.05 else "N"
            logger.info(f"  FA@{qname}: {fa_q:.4f}  cert={cert}")

        all_rows.append(row)
        del model; torch.cuda.empty_cache()

        write_hdr = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
            if write_hdr: w.writeheader()
            w.writerow(row)

    # Print summary
    logger.info(f"\n{'='*75}")
    logger.info("QUANTIZER VARIANT COMPARISON (for paper)")
    logger.info(f"{'='*75}")
    logger.info(f"{'Method':<14} {'BF16':>6} {'Sym-INT4':>9} {'Asym-ZP':>8} {'Pct-99':>7} {'GroupW-128':>11}")
    logger.info("-"*60)
    for r in all_rows:
        keys = list(r.keys())
        vals = [r.get(k, -1) for k in [
            "fa_bf16",
            "fa_Sym_INT4_paper_",
            "fa_Asym_ZP_INT4",
            "fa_Pct_INT4_99p99_",
            "fa_GroupW_INT4_g128_",
        ]]
        logger.info(
            f"  {r['method']:<12} "
            + "  ".join(f"{v:>7.4f}" if v >= 0 else f"{'?':>7}" for v in vals)
        )

    logger.info(f"\nCSV: {csv_path}")
    logger.info("\nKey message for paper:")
    logger.info("  If Sym-INT4 ≈ Asym-ZP ≈ Pct-INT4, then our symmetric")
    logger.info("  simulator is conservative (real PTQ is no more lenient).")
    logger.info("  If Sym-INT4 > Asym-ZP, then real PTQ is EASIER for the")
    logger.info("  attacker — strengthening our paper's claim.")


if __name__ == "__main__":
    main()
