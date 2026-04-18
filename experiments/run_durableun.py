"""
experiments/run_durableun.py — DurableUn Phase 1-3 runner
Fixed: resume_from_phase now correctly loads saved PEFT checkpoint.
"""

import argparse
import csv
import logging
import os
import sys
from datetime import datetime

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.logging_utils import setup_root_logger, now_str, file_ts
from src.data.data_utils import set_seed
from src.data.tofu_dataset import get_tofu_dataloaders
from src.models.model_utils import load_model_with_lora, load_tokenizer, _get_device
from src.durableun.durableun import DurableUn
from src.evaluation.evaluator import (
    compute_token_accuracy, compute_quantization_recovery,
    compute_finetuning_recovery, compute_mia_auc,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="configs/durableun_config.yaml")
    p.add_argument("--phase",   choices=["saf", "owd", "qrs", "all"], default="saf")
    p.add_argument("--resume_from_phase", default=None, choices=["saf", "owd"])
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


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
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_row(row, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading — FIXED to correctly restore PEFT checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def load_model_for_phase(config, resume_from_phase=None, logger=None):
    """
    Load model for a training phase.

    If resume_from_phase is set AND the checkpoint exists:
      → Load base model + apply saved LoRA adapter weights from checkpoint
    Otherwise:
      → Load fresh model with new LoRA adapters
    """
    model_name   = _get(config, "model.name",         default="meta-llama/Meta-Llama-3-8B-Instruct")
    dtype        = _get(config, "model.dtype",        default="bfloat16")
    device_map   = _get(config, "model.device_map",   default="cuda:0")
    load_in_4bit = _get(config, "model.load_in_4bit", default=True)
    lora_cfg     = config.get("lora")
    cache_dir    = _get(config, "paths.cache_dir",    default=None)
    ckpt_dir     = _get(config, "paths.checkpoints",  default="checkpoints")

    # Check if we have a phase checkpoint to resume from
    if resume_from_phase:
        ckpt_model_dir = os.path.join(ckpt_dir, f"durableun_{resume_from_phase}", "model")
        if os.path.exists(ckpt_model_dir):
            if logger:
                logger.info(
                    f"Loading from {resume_from_phase} checkpoint: {ckpt_model_dir}"
                )
            try:
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, BitsAndBytesConfig
                import torch as _torch

                tok = load_tokenizer(model_name, cache_dir)

                # Load base model (4-bit)
                torch_dtype = _torch.bfloat16
                bnb_config  = None
                if load_in_4bit:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch_dtype,
                    )

                base = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype if bnb_config is None else None,
                    device_map=device_map,
                    quantization_config=bnb_config,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                )
                base.config.use_cache = False

                from peft import prepare_model_for_kbit_training
                if load_in_4bit:
                    base = prepare_model_for_kbit_training(
                        base, use_gradient_checkpointing=True
                    )

                # Load saved LoRA adapter on top
                model = PeftModel.from_pretrained(
                    base, ckpt_model_dir, is_trainable=True
                )
                model.train()

                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total     = sum(p.numel() for p in model.parameters())
                if logger:
                    logger.info(
                        f"Loaded checkpoint. "
                        f"Trainable: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)"
                    )
                return model, tok

            except Exception as e:
                if logger:
                    logger.warning(
                        f"Failed to load checkpoint ({e}). Loading fresh model."
                    )
        else:
            if logger:
                logger.warning(
                    f"Checkpoint '{ckpt_model_dir}' not found. Loading fresh model."
                )

    # Fresh model
    if logger:
        logger.info(f"Loading fresh model: {model_name}")
    return load_model_with_lora(
        model_name, lora_config=lora_cfg, dtype=dtype,
        device_map=device_map, load_in_4bit=load_in_4bit, cache_dir=cache_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, tokenizer, forget_loader, retain_loader, config, phase_name, logger):
    dev        = str(_real_device(model))
    max_b      = _get(config, "eval.max_batches",      default=30)
    quant_prec = _get(config, "eval.quant_precisions", default=["bf16", "int8", "int4"])
    ft_steps   = _get(config, "eval.ft_attack_steps",  default=[50])
    skip_ft    = _get(config, "eval.skip_ft_attack",   default=False)
    max_length = _get(config, "dataset.max_length",    default=256)

    logger.info(f"[{now_str()}] Evaluating {phase_name}...")
    forget_acc = compute_token_accuracy(model, forget_loader, dev, max_b)
    retain_acc = compute_token_accuracy(model, retain_loader, dev, max_b)
    mia_auc    = compute_mia_auc(model, forget_loader, retain_loader, dev)

    logger.info(f"  Forget Acc : {forget_acc:.4f}  (target <0.05)")
    logger.info(f"  Retain Acc : {retain_acc:.4f}")
    logger.info(f"  MIA AUC    : {mia_auc:.4f}  (0.5 = perfect forget)")

    logger.info(f"[{now_str()}] Running quantization recovery attack...")
    quant_rec = compute_quantization_recovery(model, forget_loader, dev, quant_prec, max_b)

    ft_rec = {}
    if not skip_ft:
        try:
            from src.data.data_utils import get_downstream_dataloader
            alpaca = get_downstream_dataloader(
                tokenizer, datasets=["alpaca"], n_samples_per_dist=200,
                max_length=max_length, batch_size=4, num_workers=0,
            )
            ft_rec = compute_finetuning_recovery(
                model, tokenizer, forget_loader, alpaca, dev,
                steps_list=ft_steps, max_eval_batches=max_b,
            )
        except Exception as e:
            logger.warning(f"FT attack failed: {e}")

    row = {
        "phase": phase_name, "evaluated_at": now_str(),
        "forget_acc": round(forget_acc, 4),
        "retain_acc": round(retain_acc, 4),
        "mia_auc":    round(mia_auc,    4),
    }
    for p, v in quant_rec.items():
        row[f"quant_{p}"] = round(v, 4)
    for k, v in ft_rec.items():
        row[f"ft_{k}steps"] = round(v, 4)

    logger.info("\n  ── Comparison vs Phase 0 baselines ──")
    logger.info(f"  {'Method':<14} {'FA↓':>6} {'Q_INT4↓':>9}")
    logger.info(f"  {'─'*32}")
    for name, fa, qi4 in [("GA",0.028,0.262),("SCRUB",0.037,0.212),("SalUn",0.011,0.051)]:
        logger.info(f"  {name:<14} {fa:>6.3f} {qi4:>9.3f}")
    logger.info(f"  {phase_name:<14} {forget_acc:>6.3f} {quant_rec.get('int4',-1):>9.3f}  ← DurableUn")
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = load_config(args.config)

    log_dir  = _get(config, "paths.logs",        default="logs")
    res_dir  = _get(config, "paths.results",     default="results")
    ckpt_dir = _get(config, "paths.checkpoints", default="checkpoints")

    log_path = setup_root_logger(log_dir)
    logger   = logging.getLogger("durableun")
    os.makedirs(res_dir, exist_ok=True)
    results_csv = os.path.join(res_dir, f"durableun_{file_ts()}.csv")

    logger.info(f"\n{'='*60}")
    logger.info(f"  DurableUn | Phase: {args.phase.upper()}")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Time:   {now_str()}")
    logger.info(f"{'='*60}\n")

    set_seed(_get(config, "training.seed", default=42))

    if args.phase == "all":
        phases = [("saf", None), ("owd", "saf"), ("qrs", "owd")]
    else:
        phases = [(args.phase, args.resume_from_phase)]

    all_results = []

    for phase_name, resume_from in phases:
        logger.info(f"\n{'─'*60}")
        logger.info(f"  PHASE: {phase_name.upper()}")
        if resume_from:
            logger.info(f"  Resuming from: {resume_from} checkpoint")
        logger.info(f"{'─'*60}")

        model, tokenizer = load_model_for_phase(config, resume_from, logger)
        device = _real_device(model)

        forget_loader, retain_loader, _ = get_tofu_dataloaders(
            tokenizer,
            forget_split = _get(config, "dataset.forget_split", default="forget10"),
            retain_split = _get(config, "dataset.retain_split", default="retain90"),
            batch_size   = _get(config, "dataset.batch_size",   default=4),
            max_length   = _get(config, "dataset.max_length",   default=256),
            num_workers  = 0,
        )

        pipeline = DurableUn(
            model=model, tokenizer=tokenizer,
            forget_loader=forget_loader, retain_loader=retain_loader,
            config=config, device=device, ckpt_dir=ckpt_dir,
        )

        if   phase_name == "saf": pipeline.run_saf()
        elif phase_name == "owd": pipeline.run_owd()
        elif phase_name == "qrs": pipeline.run_qrs()

        row = evaluate_model(
            model, tokenizer, forget_loader, retain_loader,
            config, phase_name, logger,
        )
        all_results.append(row)
        save_row(row, results_csv)
        logger.info(f"[{now_str()}] Results saved to {results_csv}")

        del model
        torch.cuda.empty_cache()

    logger.info(f"\n{'='*60}")
    logger.info("DURABLEUN FINAL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Phase':<14} {'FA↓':>6} {'RA↑':>6} {'Q_INT4↓':>9}")
    logger.info("-" * 38)
    for r in all_results:
        logger.info(
            f"{r.get('phase','?'):<14} "
            f"{r.get('forget_acc',-1):>6.3f} "
            f"{r.get('retain_acc',-1):>6.3f} "
            f"{r.get('quant_int4',-1):>9.3f}"
        )
    logger.info(f"\nResults CSV: {results_csv}")


if __name__ == "__main__":
    main()
