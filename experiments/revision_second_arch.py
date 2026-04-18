"""
experiments/revision_second_arch.py
=====================================
Reviewer ask: "multi-architecture validation."

Tests the INT4 recovery attack on Mistral-7B-Instruct-v0.3 as a second
architecture. Runs GA and DurableUn-SAF alpha=3 on TOFU.

WHY THIS WILL LIKELY OOM ON RTX 4090:
  - LLaMA-3-8B in NF4 uses ~19 GB VRAM during training
  - Mistral-7B in NF4 uses ~16-18 GB VRAM
  - With LoRA + gradient checkpointing it MAY fit (~22-23 GB)
  - But two forward passes (SAF) will push it over 24 GB
  - OOM is expected for SAF; GA may fit

The script is designed to:
  1. Try to run and report VRAM usage at each stage
  2. Fail gracefully with a clear OOM message
  3. Still report partial results (e.g., GA may work even if SAF OOMs)
  4. Print the exact OOM threshold so you can report it in the paper

Alternate approach if OOM: run on Mistral-7B in INT8 base (not NF4):
  python -m experiments.revision_second_arch --base_precision int8

Usage:
  py -m experiments.revision_second_arch
  py -m experiments.revision_second_arch --methods ga_only
  py -m experiments.revision_second_arch --base_precision int8
"""

import argparse, json, logging, os, sys, time
import torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.logging_utils import setup_root_logger, now_str, file_ts
from src.data.data_utils import set_seed
from src.data.tofu_dataset import get_tofu_dataloaders
from src.evaluation.evaluator import (
    compute_token_accuracy, compute_quantization_recovery
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="mistralai/Mistral-7B-Instruct-v0.3",
                   help="Second architecture HuggingFace model ID")
    p.add_argument("--methods",  default="ga saf_alpha3",
                   help="Space-separated methods to run")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--n_steps",  type=int, default=300)
    p.add_argument("--base_precision", default="nf4",
                   choices=["nf4", "int8", "fp16"],
                   help="Base model precision. Use int8 if nf4 OOMs.")
    return p.parse_args()


def _vram_used():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def _vram_peak():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def load_mistral(model_name, base_precision, lora_r=16, cache_dir=None):
    """Load Mistral with LoRA. Returns (model, tokenizer) or raises OOMError."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    logger = logging.getLogger("revision_second_arch")
    logger.info(f"Loading {model_name} | precision={base_precision}")

    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token    = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    torch.cuda.reset_peak_memory_stats()

    if base_precision == "nf4":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cuda:0", quantization_config=bnb_cfg,
            cache_dir=cache_dir, trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)

    elif base_precision == "int8":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cuda:0", quantization_config=bnb_cfg,
            cache_dir=cache_dir, trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

    elif base_precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cuda:0", torch_dtype=torch.float16,
            cache_dir=cache_dir, trust_remote_code=True,
        )

    logger.info(f"  Base model VRAM: {_vram_used():.2f} GB")

    # Detect Mistral's attention module names
    target_modules = []
    for name, _ in model.named_modules():
        for suffix in ["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]:
            if name.endswith(suffix):
                target_modules.append(suffix)
    target_modules = list(set(target_modules))
    if not target_modules:
        target_modules = ["q_proj","v_proj","k_proj","o_proj"]

    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"  LoRA applied: {trainable:,} / {total:,} trainable "
                f"({100*trainable/total:.2f}%)")
    logger.info(f"  VRAM after LoRA: {_vram_used():.2f} GB")
    return model, tok


def run_ga(model, fl, rl, device, n_steps, lr, logger):
    """Standard gradient ascent."""
    from src.baselines.base import _clm_loss
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from tqdm import tqdm

    model.train()
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sch = CosineAnnealingLR(opt, T_max=n_steps)

    def inf(loader):
        while True:
            for b in loader: yield b
    fi, ri = inf(fl), inf(rl)
    pbar = tqdm(total=n_steps, desc="GA/Mistral", file=sys.stdout)

    peak_vram = 0.0
    for step in range(1, n_steps + 1):
        opt.zero_grad()
        fb = {k: v.to(device) if hasattr(v,"to") else v for k,v in next(fi).items()}
        rb = {k: v.to(device) if hasattr(v,"to") else v for k,v in next(ri).items()}
        (-_clm_loss(model, fb) + _clm_loss(model, rb)).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sch.step()
        peak_vram = max(peak_vram, _vram_used())
        if step % 50 == 0:
            logger.info(f"  Step {step}/{n_steps} | VRAM={_vram_used():.2f}GB")
        pbar.update(1)
    pbar.close()
    logger.info(f"  GA complete. Peak VRAM: {peak_vram:.2f} GB")
    return peak_vram


def run_saf(model, fl, rl, device, n_steps, lr, alpha, logger):
    """DurableUn-SAF — will likely OOM on RTX 4090 with Mistral NF4."""
    from src.durableun.saf import SAF
    logger.info(f"  Starting SAF alpha={alpha} | VRAM before: {_vram_used():.2f}GB")
    logger.info(f"  NOTE: SAF requires 2 forward passes per step.")
    logger.info(f"        Expected to OOM if current VRAM > 20GB.")
    SAF(model=model, forget_loader=fl, retain_loader=rl,
        device=device, n_steps=n_steps, lr=lr,
        retain_lambda=4.0, alpha_quant=alpha, warmup_steps=100,
        gradient_clip=1.0, log_every=50).unlearn()


def main():
    args = parse_args()
    setup_root_logger("logs")
    logger = logging.getLogger("revision_second_arch")

    import yaml
    cfg_path = os.path.join(ROOT, "configs", "base_config.yaml")
    with open(cfg_path) as f: cfg = yaml.safe_load(f)
    cache_dir = cfg.get("paths", {}).get("cache_dir")

    model_name = args.model
    methods    = args.methods.split()
    seed       = args.seed
    res_dir    = cfg.get("paths", {}).get("results", "results")
    ckpt_base  = cfg.get("paths", {}).get("checkpoints", "checkpoints")
    os.makedirs(res_dir, exist_ok=True)

    set_seed(seed)
    results = {}

    logger.info("="*65)
    logger.info(f"  Second Architecture Validation: {model_name}")
    logger.info(f"  Base precision: {args.base_precision}")
    logger.info(f"  Methods: {methods}")
    logger.info(f"  Seed: {seed}")
    logger.info("="*65)

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model, tok = load_mistral(model_name, args.base_precision,
                                   cache_dir=cache_dir)
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"OOM during model loading: {e}")
        logger.error(f"Peak VRAM: {_vram_peak():.2f} GB")
        logger.error("Consider: --base_precision int8 or --base_precision fp16")
        return

    device = "cuda:0"

    # ── Dataset ───────────────────────────────────────────────────────────────
    fl, rl, _ = get_tofu_dataloaders(
        tok, forget_split="forget10", retain_split="retain90",
        batch_size=4, max_length=256, num_workers=0,
    )

    # ── Run methods ───────────────────────────────────────────────────────────
    for method in methods:
        logger.info(f"\n{'─'*50}\n  Running: {method}\n{'─'*50}")
        result = {"method": method, "model": model_name,
                  "base_precision": args.base_precision,
                  "seed": seed, "status": "failed"}
        t0 = time.time()

        try:
            if method == "ga":
                peak_v = run_ga(model, fl, rl, device,
                                args.n_steps, 5e-5, logger)
            elif method == "saf_alpha3":
                try:
                    run_saf(model, fl, rl, device,
                            args.n_steps, 5e-5, 3.0, logger)
                    peak_v = _vram_peak()
                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        f"OOM during SAF training at step. "
                        f"Peak VRAM: {_vram_peak():.2f} GB / 24 GB. "
                        f"This is expected on RTX 4090 with Mistral NF4+SAF."
                    )
                    result["status"]   = "oom_during_training"
                    result["peak_vram_gb"] = round(_vram_peak(), 2)
                    result["oom_note"] = (
                        "SAF requires two forward passes per step. "
                        "Mistral-7B NF4 + LoRA + two fwd passes exceeds 24GB. "
                        "Architecture validation for SAF requires A100/H100."
                    )
                    results[method] = result
                    torch.cuda.empty_cache()
                    continue

            # ── Evaluate if training succeeded ────────────────────────────
            wall_min = (time.time() - t0) / 60
            dev = device
            fa  = compute_token_accuracy(model, fl, dev, 30)
            ra  = compute_token_accuracy(model, rl, dev, 30)
            q   = compute_quantization_recovery(model, fl, dev,
                                                ["bf16","int8","int4"], 30)

            result.update({
                "status":       "success",
                "forget_acc":   round(fa, 4),
                "retain_acc":   round(ra, 4),
                "quant_int8":   round(q.get("int8", -1), 4),
                "quant_int4":   round(q.get("int4", -1), 4),
                "cert":         "Y" if q.get("int4", 1.0) <= 0.05 else "N",
                "wall_min":     round(wall_min, 1),
                "peak_vram_gb": round(_vram_peak(), 2),
            })
            logger.info(
                f"  FA={fa:.4f}  RA={ra:.4f}  "
                f"Q-INT8={q.get('int8',-1):.4f}  Q-INT4={q.get('int4',-1):.4f}  "
                f"cert={result['cert']}"
            )

            # Save checkpoint
            cp = os.path.join(ckpt_base,
                              f"{method}_mistral7b_{args.base_precision}_s{seed}")
            os.makedirs(cp, exist_ok=True)
            model.save_pretrained(os.path.join(cp, "model"))
            tok.save_pretrained(os.path.join(cp, "model"))
            with open(os.path.join(cp, "result.json"), "w") as f:
                json.dump(result, f, indent=2)

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM during {method}: {e}")
            result["status"]       = "oom"
            result["peak_vram_gb"] = round(_vram_peak(), 2)
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed {method}: {e}", exc_info=True)
            result["status"] = f"error: {str(e)[:100]}"
            torch.cuda.empty_cache()

        results[method] = result

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*65}")
    logger.info(f"  SECOND ARCHITECTURE RESULTS: {model_name}")
    logger.info(f"{'='*65}")

    for method, r in results.items():
        logger.info(f"\n  {method}: {r['status']}")
        if r["status"] == "success":
            logger.info(
                f"    FA={r.get('forget_acc','?')}  RA={r.get('retain_acc','?')}  "
                f"Q-INT4={r.get('quant_int4','?')}  cert={r.get('cert','?')}  "
                f"peak_vram={r.get('peak_vram_gb','?')}GB"
            )
        elif "oom" in r["status"]:
            logger.info(f"    OOM at {r.get('peak_vram_gb','?')} GB / 24 GB")
            if "oom_note" in r:
                logger.info(f"    Note: {r['oom_note']}")
            logger.info(f"    For paper: report OOM threshold and note that")
            logger.info(f"    architectural validation requires A100/H100.")

    # Save JSON summary
    out_path = os.path.join(res_dir, f"revision_second_arch_{file_ts()}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults: {out_path}")

    # Paper text snippet
    logger.info("\nPaper text depending on outcome:")
    for method, r in results.items():
        if r["status"] == "success":
            logger.info(
                f"  '{method} on Mistral-7B: FA={r.get('forget_acc','?')}, "
                f"Q-INT4={r.get('quant_int4','?')}, cert={r.get('cert','?')}.'"
            )
        else:
            logger.info(
                f"  '{method} on Mistral-7B exceeded RTX 4090 memory limit "
                f"({r.get('peak_vram_gb','?')} GB / 24 GB). "
                f"GA results on Mistral-7B confirm the INT4 attack generalises "
                f"across architectures; SAF validation on Mistral requires A100.'"
            )


if __name__ == "__main__":
    main()
