"""
experiments/phase0_baseline_audit.py
======================================
Phase 0: Reproduce ICLR 2025 Baseline Failure.

Runs all baseline unlearning methods and attacks each with:
  - Quantization attack   (INT4, INT8, BF16)
  - Fine-tuning attack    (50, 100, 500 steps on Alpaca)

Features:
  ✓ Checkpoint saved after every method (resume where you left off)
  ✓ Full timestamps in every log line
  ✓ Results CSV updated incrementally
  ✓ --resume flag skips already-completed methods
  ✓ Quick mode: use quick_config.yaml for a 20-minute sanity check

Usage (QUICK — do this first):
  python experiments/phase0_baseline_audit.py --config configs/quick_config.yaml

Usage (FULL):
  python experiments/phase0_baseline_audit.py --config configs/base_config.yaml

Usage (RESUME after crash):
  python experiments/phase0_baseline_audit.py --config configs/base_config.yaml --resume

Usage (single method):
  python experiments/phase0_baseline_audit.py --config configs/base_config.yaml --methods ga
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import yaml

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.logging_utils import setup_root_logger, now_str, file_ts
from src.utils.checkpoint import CheckpointManager
from src.data.data_utils import set_seed, get_downstream_dataloader
from src.data.tofu_dataset import get_tofu_dataloaders
from src.models.model_utils import load_model_with_lora
from src.baselines import get_baseline, BASELINE_MAP
from src.evaluation.evaluator import (
    compute_token_accuracy,
    compute_quantization_recovery,
    compute_finetuning_recovery,
    compute_mia_auc,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="configs/base_config.yaml",
                   help="Config file. Use quick_config.yaml for a fast sanity check.")
    p.add_argument("--methods", nargs="+",
                   default=["ga", "npo", "scrub", "salun", "rmu", "alpha_edit"])
    p.add_argument("--resume",  action="store_true",
                   help="Skip methods that already have checkpoints.")
    p.add_argument("--n_steps", type=int, default=None,
                   help="Override training steps from config.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get(cfg, *keys, default=None):
    """Resolve a key that may live under different section names."""
    for k in keys:
        v = cfg
        try:
            for part in k.split("."):
                v = v[part]
            return v
        except (KeyError, TypeError):
            pass
    return default


# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────

def _real_device(model) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_row(row: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Per-method runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one_method(
    method_name: str,
    config: dict,
    args,
    logger: logging.Logger,
    ckpt: CheckpointManager,
) -> dict:
    """
    Load model → unlearn → eval → checkpoint → return metrics dict.
    """
    started = now_str()
    logger.info(f"\n{'='*65}")
    logger.info(f"  METHOD: {method_name.upper()}")
    logger.info(f"  Started: {started}")
    logger.info(f"{'='*65}")

    seed       = _get(config, "training.seed", default=42)
    set_seed(seed)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name = _get(config, "model.name",        default="meta-llama/Meta-Llama-3-8B-Instruct")
    dtype      = _get(config, "model.dtype",       default="bfloat16")
    device_map = _get(config, "model.device_map",  default="cuda:0")
    in_4bit    = _get(config, "model.load_in_4bit", default=False)
    in_8bit    = _get(config, "model.load_in_8bit", default=False)
    lora_cfg   = config.get("lora", None)
    cache_dir  = _get(config, "paths.cache_dir",   default=None)

    logger.info(f"  [{now_str()}] Loading model: {model_name}")
    model, tokenizer = load_model_with_lora(
        model_name,
        lora_config=lora_cfg,
        dtype=dtype,
        device_map=device_map,
        load_in_4bit=in_4bit,
        load_in_8bit=in_8bit,
        cache_dir=cache_dir,
    )
    device = _real_device(model)
    logger.info(f"  [{now_str()}] Model on device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    forget_split = _get(config, "dataset.forget_split", default="forget10")
    retain_split = _get(config, "dataset.retain_split", default="retain90")
    max_length   = _get(config, "dataset.max_length",   default=512)
    batch_size   = _get(config, "dataset.batch_size",   default=4)
    num_workers  = _get(config, "dataset.num_workers",  default=0)

    logger.info(f"  [{now_str()}] Loading TOFU ({forget_split} / {retain_split})")
    forget_loader, retain_loader, _ = get_tofu_dataloaders(
        tokenizer,
        forget_split=forget_split,
        retain_split=retain_split,
        batch_size=batch_size,
        max_length=max_length,
        cache_dir=cache_dir,
        num_workers=num_workers,
    )

    # ── Hyperparameters ───────────────────────────────────────────────────────
    n_steps       = (args.n_steps
                     or _get(config, "training.n_steps", "training.forget_steps", default=300))
    lr_forget     = _get(config, "training.lr_forget",    default=5e-5)
    lr_retain     = _get(config, "training.lr_retain",    default=1e-5)
    retain_lambda = _get(config, "training.retain_lambda", default=1.0)
    gradient_clip = _get(config, "training.gradient_clip", default=1.0)
    log_every     = _get(config, "training.log_every",    default=50)

    hparams = dict(
        n_steps=n_steps, lr=lr_forget, lr_forget=lr_forget, lr_retain=lr_retain,
        retain_lambda=retain_lambda, gradient_clip=gradient_clip, log_every=log_every,
        # Method-specific (absorbed silently if not needed)
        beta=0.1, gamma=0.5, msteps=1, saliency_threshold=0.5,
        steering_coef=20.0, alpha_rmu=1200.0, layer_id=None, svd_rank=128,
    )

    # ── Unlearn ───────────────────────────────────────────────────────────────
    logger.info(f"  [{now_str()}] Creating {method_name.upper()} unlearner ({n_steps} steps)")
    unlearner = get_baseline(
        method_name,
        model=model,
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        device=device,
        **hparams,
    )

    logger.info(f"  [{now_str()}] Training started")
    result = unlearner.unlearn(forget_loader, retain_loader)
    logger.info(f"  [{now_str()}] Training finished in {result.wall_time_seconds/60:.1f} min")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    max_b        = _get(config, "eval.max_batches", default=30)
    skip_ft      = _get(config, "eval.skip_ft_attack", default=False)
    quant_prec   = _get(config, "eval.quant_precisions", default=["bf16", "int8", "int4"])
    ft_steps     = _get(config, "eval.ft_attack_steps",  default=[50, 100, 500])
    dev_str      = str(device)

    logger.info(f"  [{now_str()}] Evaluating forget/retain accuracy...")
    forget_acc = compute_token_accuracy(model, forget_loader, dev_str, max_b)
    retain_acc = compute_token_accuracy(model, retain_loader, dev_str, max_b)
    mia_auc    = compute_mia_auc(model, forget_loader, retain_loader, dev_str)

    logger.info(f"  Forget Acc : {forget_acc:.4f}  (↓ good)")
    logger.info(f"  Retain Acc : {retain_acc:.4f}  (↑ good)")
    logger.info(f"  MIA AUC    : {mia_auc:.4f}  (0.5 = perfect forget)")

    logger.info(f"  [{now_str()}] Running quantization recovery attack...")
    quant_rec = compute_quantization_recovery(model, forget_loader, dev_str, quant_prec, max_b)

    ft_rec = {}
    if not skip_ft:
        logger.info(f"  [{now_str()}] Running fine-tuning recovery attack...")
        try:
            alpaca = get_downstream_dataloader(
                tokenizer, datasets=["alpaca"],
                n_samples_per_dist=500,
                max_length=max_length, batch_size=batch_size,
                cache_dir=cache_dir, num_workers=0,
            )
            ft_rec = compute_finetuning_recovery(
                model, tokenizer, forget_loader, alpaca, dev_str,
                steps_list=ft_steps, max_eval_batches=max_b,
            )
        except Exception as e:
            logger.warning(f"  FT attack failed: {e}")
    else:
        logger.info("  [skip_ft_attack=true] Skipping fine-tuning attack.")

    # ── Build metric dict ─────────────────────────────────────────────────────
    finished = now_str()
    metrics  = {
        "method":          method_name,
        "started_at":      started,
        "finished_at":     finished,
        "wall_time_min":   round(result.wall_time_seconds / 60, 1),
        "peak_gpu_gb":     round(result.peak_gpu_memory_gb, 2),
        "gradient_steps":  n_steps,
        "forget_acc":      round(forget_acc, 4),
        "retain_acc":      round(retain_acc, 4),
        "mia_auc":         round(mia_auc,    4),
    }
    for p, v in quant_rec.items():
        metrics[f"quant_{p}"] = round(v, 4)
    for k_steps, v in ft_rec.items():
        metrics[f"ft_{k_steps}steps"] = round(v, 4)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    logger.info(f"  [{now_str()}] Saving checkpoint...")
    ckpt.save(method_name, model, tokenizer, metrics, config)

    # ── Free GPU ──────────────────────────────────────────────────────────────
    del model
    torch.cuda.empty_cache()

    logger.info(f"  [{now_str()}] Done: {method_name.upper()}")
    logger.info(f"  Summary: forget={forget_acc:.4f} | retain={retain_acc:.4f} | "
                f"quant_int4={quant_rec.get('int4', -1):.4f}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure1(results_path: str, out_path: str):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np

        df = pd.read_csv(results_path)
        m  = df["method"].tolist()
        x  = np.arange(len(m))
        w  = 0.25

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            "Figure 1 — Baseline Failure Under Recovery Attacks\n"
            "(lower = model has forgotten / is durable)",
            fontsize=12, fontweight="bold",
        )

        ax = axes[0]
        for i, (col, lbl) in enumerate(
            [("quant_bf16","BF16"),("quant_int8","INT8"),("quant_int4","INT4")]
        ):
            if col in df.columns:
                ax.bar(x + (i-1)*w, df[col], w, label=lbl)
        ax.axhline(0.05, color="green", ls="--", lw=2, label="Target 5%")
        ax.set_xticks(x); ax.set_xticklabels(m, rotation=30, ha="right")
        ax.set_ylabel("Forget Acc After Quantization ↓")
        ax.set_title("Quantization Attack"); ax.legend(); ax.set_ylim(0,1)

        ax = axes[1]
        for i, k in enumerate([50, 100, 500]):
            col = f"ft_{k}steps"
            if col in df.columns:
                ax.bar(x + (i-1)*w, df[col], w, label=f"{k} steps")
        ax.axhline(0.10, color="green", ls="--", lw=2, label="Target 10%")
        ax.set_xticks(x); ax.set_xticklabels(m, rotation=30, ha="right")
        ax.set_ylabel("Forget Acc After Fine-Tuning ↓")
        ax.set_title("Fine-Tuning Attack"); ax.legend(); ax.set_ylim(0,1)

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[{now_str()}] Figure 1 saved: {out_path}")
        plt.close()
    except ImportError:
        print("matplotlib/pandas not installed — skipping Figure 1.")
    except Exception as e:
        print(f"Figure 1 failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = load_config(args.config)

    # Dirs
    log_dir  = _get(config, "paths.logs",        default="logs")
    res_dir  = _get(config, "paths.results",     default="results")
    ckpt_dir = _get(config, "paths.checkpoints", default="checkpoints")
    fig_dir  = _get(config, "paths.figures",     default="figures")

    # Setup logging (console + file with timestamp)
    log_path = setup_root_logger(log_dir)
    logger   = logging.getLogger("phase0")

    os.makedirs(res_dir, exist_ok=True)
    results_csv = os.path.join(res_dir, f"baseline_recovery_{file_ts()}.csv")

    ckpt = CheckpointManager(ckpt_dir)

    logger.info(f"[{now_str()}] ===== Phase 0: Baseline Failure Audit =====")
    logger.info(f"[{now_str()}] Config  : {args.config}")
    logger.info(f"[{now_str()}] Methods : {args.methods}")
    logger.info(f"[{now_str()}] Resume  : {args.resume}")
    logger.info(f"[{now_str()}] Log     : {log_path}")
    logger.info(f"[{now_str()}] Results : {results_csv}")

    if args.resume:
        completed = ckpt.list_completed()
        logger.info(f"[{now_str()}] Already completed: {completed}")

    all_results = []
    for method_name in args.methods:
        if method_name not in BASELINE_MAP:
            logger.warning(f"Unknown method '{method_name}' — skipping.")
            continue

        # Resume: skip if checkpoint exists
        if args.resume and ckpt.exists(method_name):
            logger.info(f"[{now_str()}] SKIP {method_name} — checkpoint exists.")
            saved = ckpt.load_result(method_name)
            if saved:
                row = saved.get("metrics", {})
                row["method"] = method_name
                all_results.append(row)
                save_row(row, results_csv)
            continue

        try:
            row = run_one_method(method_name, config, args, logger, ckpt)
            all_results.append(row)
            save_row(row, results_csv)
            logger.info(f"[{now_str()}] Saved row to {results_csv}")

        except Exception as e:
            logger.error(f"[{now_str()}] {method_name} FAILED: {e}", exc_info=True)

    # Print summary table
    if all_results:
        logger.info(f"\n[{now_str()}] ===== RESULTS SUMMARY =====")
        header = f"{'Method':<14} {'FA↓':>6} {'RA↑':>6} {'MIA':>6} {'Q_INT4↓':>9} {'FT_500↓':>9} {'Time(min)':>10}"
        logger.info(header)
        logger.info("-" * len(header))
        for r in all_results:
            logger.info(
                f"{r.get('method','?'):<14} "
                f"{r.get('forget_acc',-1):>6.3f} "
                f"{r.get('retain_acc',-1):>6.3f} "
                f"{r.get('mia_auc',-1):>6.3f} "
                f"{r.get('quant_int4',-1):>9.3f} "
                f"{r.get('ft_500steps',-1):>9.3f} "
                f"{r.get('wall_time_min',-1):>10.1f}"
            )

        # Generate Figure 1
        plot_figure1(
            results_csv,
            os.path.join(fig_dir, "figure1_baseline_failure.png"),
        )

    logger.info(f"\n[{now_str()}] Phase 0 complete.")
    logger.info(f"Results CSV : {results_csv}")
    logger.info(f"Checkpoints : {ckpt_dir}/")


if __name__ == "__main__":
    main()
