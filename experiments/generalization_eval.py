"""
experiments/generalization_eval.py
====================================
Addresses reviewer: "narrow experimental scope (one dataset)" and
"missing stronger recent baselines".

Runs:
  1. GradDiff baseline (Maini et al. 2024 — the TOFU paper's own baseline)
  2. All methods on TOFU forget05 (smaller split, tests robustness)
  3. WikiText-2 general knowledge preservation check
  4. RA-INT4 for all methods (retain accuracy under INT4 quantization)

Usage:
  python experiments/generalization_eval.py --config configs/base_config.yaml

  # Just GradDiff on forget10:
  python experiments/generalization_eval.py --config configs/base_config.yaml \
      --methods graddiff --split forget10

  # Just second-dataset eval on existing checkpoints:
  python experiments/generalization_eval.py --config configs/base_config.yaml \
      --from_checkpoints --checkpoints checkpoints/ga checkpoints/saf_alpha_3p0

Expected runtime: ~45 min for GradDiff alone.
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
from src.evaluation.evaluator_additions import (
    compute_full_eval, compute_dataset_generalization,
    compute_token_accuracy_quantized
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="configs/base_config.yaml")
    p.add_argument("--methods", nargs="+", default=["graddiff"],
                   choices=["graddiff", "ga", "salun"])
    p.add_argument("--split",   default="forget10",
                   choices=["forget10", "forget05", "forget01"])
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--from_checkpoints", action="store_true",
                   help="Evaluate existing checkpoints instead of training")
    p.add_argument("--checkpoints", nargs="+", default=[],
                   help="Checkpoint dirs to evaluate (with --from_checkpoints)")
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


def run_method(method_name, model, tokenizer, forget_loader, retain_loader,
               device, config, logger):
    """Train one method and return the trained model."""
    n_steps = _get(config, "training.n_steps", default=300)
    lr      = _get(config, "training.lr",      default=5e-5)
    
    if method_name == "graddiff":
        from src.baselines.gradient_difference import GradDiff
        unlearner = GradDiff(
            model=model, forget_loader=forget_loader, retain_loader=retain_loader,
            device=device, n_steps=n_steps, lr=lr,
            retain_lambda=_get(config, "training.retain_lambda", default=1.0),
            gradient_clip=_get(config, "training.gradient_clip", default=1.0),
            log_every=_get(config, "training.log_every", default=50),
        )
    elif method_name == "ga":
        from src.baselines.base import _clm_loss
        class _GA:
            def __init__(self): pass
            def unlearn(self):
                from torch.optim import AdamW
                from torch.optim.lr_scheduler import CosineAnnealingLR
                from tqdm import tqdm
                opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
                sch = CosineAnnealingLR(opt, T_max=n_steps)
                fi  = _inf(forget_loader); ri = _inf(retain_loader)
                for _ in tqdm(range(n_steps), desc="GA"):
                    opt.zero_grad()
                    fb = {k: v.to(device) if hasattr(v,"to") else v for k,v in next(fi).items()}
                    rb = {k: v.to(device) if hasattr(v,"to") else v for k,v in next(ri).items()}
                    loss = -_clm_loss(model, fb) + _clm_loss(model, rb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step(); sch.step()
        def _inf(loader):
            while True:
                for b in loader: yield b
        ga = _GA(); ga.unlearn()
        return model

    elif method_name == "salun":
        from src.baselines.salun import SalUn
        unlearner = SalUn(
            model=model, forget_loader=forget_loader, retain_loader=retain_loader,
            device=device, n_steps=n_steps, lr=lr,
        )
    
    unlearner.unlearn()
    return model


def evaluate_checkpoint(ckpt_dir, model_name, tokenizer, config, logger):
    """Load a saved checkpoint and run full evaluation."""
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import json as _json
    
    ckpt_model = os.path.join(ckpt_dir, "model")
    if not os.path.exists(ckpt_model):
        logger.warning(f"No model/ subdir in {ckpt_dir}")
        return None
    
    # Create adapter_config.json if missing
    adapter_cfg = os.path.join(ckpt_model, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        with open(adapter_cfg, "w") as f:
            _json.dump({
                "peft_type": "LORA", "task_type": "CAUSAL_LM",
                "r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none",
                "target_modules": ["q_proj","v_proj","k_proj","o_proj"],
                "fan_in_fan_out": False, "inference_mode": True,
                "base_model_name_or_path": model_name,
            }, f)
    
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                              bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cuda:0", quantization_config=bnb, trust_remote_code=True)
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, ckpt_model, is_trainable=False)
    model.eval()
    return model


def main():
    args   = parse_args()
    config = load_config(args.config)
    
    setup_root_logger(_get(config, "paths.logs", default="logs"))
    logger = logging.getLogger("generalization_eval")
    os.makedirs(_get(config, "paths.results", default="results"), exist_ok=True)
    results_csv = os.path.join(
        _get(config, "paths.results", default="results"),
        f"generalization_{file_ts()}.csv"
    )
    
    set_seed(args.seed)
    
    model_name = _get(config, "model.name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    
    all_rows = []
    
    if args.from_checkpoints:
        # Evaluate existing checkpoints
        from src.models.model_utils import load_tokenizer
        tokenizer = load_tokenizer(model_name)
        
        for ckpt_dir in args.checkpoints:
            name = os.path.basename(ckpt_dir)
            logger.info(f"\nEvaluating checkpoint: {name}")
            
            model = evaluate_checkpoint(ckpt_dir, model_name, tokenizer, config, logger)
            if model is None:
                continue
            
            device = "cuda:0"
            fl, rl, _ = get_tofu_dataloaders(
                tokenizer,
                forget_split=args.split,
                retain_split=f"retain{100 - int(args.split.replace('forget',''))}",
                batch_size=4, max_length=256, num_workers=0,
            )
            
            metrics = compute_full_eval(model, fl, rl, device, max_batches=30)
            gen     = compute_dataset_generalization(model, tokenizer, device, max_batches=20)
            
            row = {"checkpoint": name, "split": args.split, **metrics}
            for dataset_name, d_metrics in gen.items():
                for k, v in d_metrics.items():
                    row[f"{dataset_name}_{k}"] = v
            
            all_rows.append(row)
            del model; torch.cuda.empty_cache()
    
    else:
        # Train + evaluate each method
        for method_name in args.methods:
            logger.info(f"\n{'='*50}")
            logger.info(f"  {method_name.upper()} | split={args.split} | seed={args.seed}")
            
            model, tokenizer = load_model_with_lora(
                model_name,
                lora_config=config.get("lora"),
                dtype=_get(config, "model.dtype", default="bfloat16"),
                device_map=_get(config, "model.device_map", default="cuda:0"),
                load_in_4bit=_get(config, "model.load_in_4bit", default=True),
            )
            device = _real_device(model)
            
            retain_split = f"retain{100 - int(args.split.replace('forget',''))}"
            fl, rl, _ = get_tofu_dataloaders(
                tokenizer,
                forget_split=args.split,
                retain_split=retain_split,
                batch_size=4, max_length=256, num_workers=0,
            )
            
            model = run_method(method_name, model, tokenizer, fl, rl,
                                device, config, logger)
            
            metrics = compute_full_eval(model, fl, rl, str(device), max_batches=30)
            gen     = compute_dataset_generalization(model, tokenizer, str(device), max_batches=20)
            
            row = {"method": method_name, "split": args.split, "seed": args.seed, **metrics}
            for dataset_name, d_metrics in gen.items():
                for k, v in d_metrics.items():
                    row[f"{dataset_name}_{k}"] = v
            
            all_rows.append(row)
            
            write_hdr = not os.path.exists(results_csv)
            with open(results_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
                if write_hdr: w.writeheader()
                w.writerow(row)
            
            del model; torch.cuda.empty_cache()
    
    # Summary
    logger.info(f"\n{'='*65}")
    logger.info("GENERALIZATION EVALUATION RESULTS")
    logger.info(f"{'='*65}")
    logger.info(f"{'Name':<20} {'FA↓':>6} {'RA↑':>6} {'Q-INT4↓':>8} {'RA-INT4↑':>9} {'forget05↓':>10}")
    logger.info("-"*65)
    for r in all_rows:
        name = r.get("method", r.get("checkpoint", "?"))
        logger.info(
            f"{name:<20} "
            f"{r.get('forget_acc',-1):>6.4f} "
            f"{r.get('retain_acc',-1):>6.4f} "
            f"{r.get('quant_int4',-1):>8.4f} "
            f"{r.get('ra_int4',-1):>9.4f} "
            f"{r.get('tofu_forget05_forget_acc',-1):>10.4f}"
        )
    logger.info(f"\nCSV: {results_csv}")


if __name__ == "__main__":
    main()
