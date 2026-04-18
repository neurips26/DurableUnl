"""
experiments/pareto_sweep.py — Pareto sweep for DurableUn-SAF.
Now supports --retain_lambda override for targeted runs.
"""

import argparse, csv, json, logging, os, sys, glob
from datetime import datetime
import torch, yaml
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.logging_utils import setup_root_logger, now_str, file_ts
from src.data.data_utils import set_seed
from src.data.tofu_dataset import get_tofu_dataloaders
from src.models.model_utils import load_model_with_lora
from src.durableun.saf import SAF
from src.evaluation.evaluator import (
    compute_token_accuracy, compute_quantization_recovery, compute_mia_auc,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="configs/durableun_config.yaml")
    p.add_argument("--alphas",  nargs="+", type=float, default=[0.0, 1.0, 3.0])
    p.add_argument("--n_steps", type=int, default=300)
    p.add_argument("--retain_lambda", type=float, default=None,
                   help="Override retain_lambda (default: alpha+1). Try 6.0 for α=2.0")
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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_one_alpha(alpha, config, args, logger):
    logger.info(f"\n{'='*55}")
    logger.info(f"  alpha_quant = {alpha}")
    logger.info(f"{'='*55}")

    set_seed(_get(config, "training.seed", default=42))

    model, tokenizer = load_model_with_lora(
        _get(config, "model.name", default="meta-llama/Meta-Llama-3-8B-Instruct"),
        lora_config=config.get("lora"),
        dtype=_get(config, "model.dtype", default="bfloat16"),
        device_map=_get(config, "model.device_map", default="cuda:0"),
        load_in_4bit=_get(config, "model.load_in_4bit", default=True),
        cache_dir=_get(config, "paths.cache_dir", default=None),
    )
    device = _real_device(model)

    forget_loader, retain_loader, _ = get_tofu_dataloaders(
        tokenizer,
        forget_split=_get(config, "dataset.forget_split", default="forget10"),
        retain_split=_get(config, "dataset.retain_split", default="retain90"),
        batch_size=_get(config, "dataset.batch_size", default=4),
        max_length=_get(config, "dataset.max_length", default=256),
        num_workers=0,
    )

    # retain_lambda: use override if provided, else alpha+1
    retain_lambda = args.retain_lambda if args.retain_lambda else max(1.0, alpha + 1.0)
    logger.info(f"  retain_lambda = {retain_lambda}")

    saf = SAF(
        model=model, forget_loader=forget_loader, retain_loader=retain_loader,
        device=device, n_steps=args.n_steps,
        lr=_get(config, "training.lr", default=5e-5),
        retain_lambda=retain_lambda,
        gradient_clip=_get(config, "training.gradient_clip", default=1.0),
        log_every=_get(config, "training.log_every", default=50),
        alpha_quant=alpha,
        warmup_steps=_get(config, "saf.warmup_steps", default=100),
    )
    result = saf.unlearn()

    dev   = str(device)
    max_b = _get(config, "eval.max_batches", default=30)
    fa    = compute_token_accuracy(model, forget_loader, dev, max_b)
    ra    = compute_token_accuracy(model, retain_loader, dev, max_b)
    mia   = compute_mia_auc(model, forget_loader, retain_loader, dev)
    quant = compute_quantization_recovery(model, forget_loader, dev,
                                          ["bf16", "int8", "int4"], max_b)

    row = {
        "alpha_quant":   alpha,
        "retain_lambda": retain_lambda,
        "n_steps":       args.n_steps,
        "forget_acc":    round(fa,  4),
        "retain_acc":    round(ra,  4),
        "mia_auc":       round(mia, 4),
        "quant_bf16":    round(quant.get("bf16", -1), 4),
        "quant_int8":    round(quant.get("int8", -1), 4),
        "quant_int4":    round(quant.get("int4", -1), 4),
        "wall_time_min": round(result.wall_time_seconds / 60, 1),
        "evaluated_at":  now_str(),
    }
    logger.info(f"  α={alpha} λ={retain_lambda} | FA={fa:.4f} | RA={ra:.4f} | Q_INT4={quant.get('int4',-1):.4f}")

    ckpt_dir = os.path.join(
        _get(config, "paths.checkpoints", default="checkpoints"),
        f"saf_alpha_{str(alpha).replace('.','p')}_lambda_{str(retain_lambda).replace('.','p')}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(os.path.join(ckpt_dir, "model"))
    tokenizer.save_pretrained(os.path.join(ckpt_dir, "model"))
    with open(os.path.join(ckpt_dir, "result.json"), "w") as f:
        json.dump(row, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return row


def main():
    args   = parse_args()
    config = load_config(args.config)

    setup_root_logger(_get(config, "paths.logs", default="logs"))
    logger = logging.getLogger("pareto_sweep")
    os.makedirs(_get(config, "paths.results", default="results"), exist_ok=True)
    results_csv = os.path.join(
        _get(config, "paths.results", default="results"),
        f"pareto_sweep_{file_ts()}.csv"
    )

    logger.info(f"Pareto sweep | alphas={args.alphas} | n_steps={args.n_steps}")
    if args.retain_lambda:
        logger.info(f"retain_lambda override: {args.retain_lambda}")

    all_rows = []
    for alpha in sorted(args.alphas):
        row = run_one_alpha(alpha, config, args, logger)
        all_rows.append(row)
        write_header = not os.path.exists(results_csv)
        with open(results_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
            if write_header: w.writeheader()
            w.writerow(row)

    logger.info(f"\n{'='*60}")
    logger.info("PARETO SWEEP RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"{'α':>6} {'λ':>5} {'FA↓':>7} {'RA↑':>7} {'Q_INT4↓':>9} {'Time':>6}")
    logger.info("-" * 50)
    for r in all_rows:
        logger.info(
            f"{r['alpha_quant']:>6.1f} {r['retain_lambda']:>5.1f} "
            f"{r['forget_acc']:>7.4f} {r['retain_acc']:>7.4f} "
            f"{r['quant_int4']:>9.4f} {r['wall_time_min']:>5.0f}m"
        )
    logger.info(f"\nBaselines:")
    logger.info(f"  GA    FA=0.028  RA=0.521  Q_INT4=0.262")
    logger.info(f"  SalUn FA=0.011  RA=0.541  Q_INT4=0.051")
    logger.info(f"\nCSV: {results_csv}")


if __name__ == "__main__":
    main()
