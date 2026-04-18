"""
experiments/multi_dataset_audit.py
====================================
Runs the quantization recovery attack audit across multiple datasets and methods.
Replaces/extends phase0_baseline_audit.py with full dataset + method coverage.

Datasets:
  tofu        — TOFU forget10 (Maini et al. 2024) — default
  muse_news   — MUSE BBC News corpus (Shi et al. 2024)
  muse_books  — MUSE Harry Potter corpus
  wpu         — WikiBio Person Unlearning (factual knowledge)

Methods (any from baseline_registry.py):
  Phase 0 originals: ga, npo, scrub, salun, rmu, alpha_edit
  Modern:            graddiff, wga, wga_lp, tv, dare, noisy_ga, langevin
  DurableUn:         durableun_saf_v3, durableun_saf_alpha3

Usage:
  # Full audit — all Phase 0 baselines on TOFU (reproduces Table 1):
  python experiments/multi_dataset_audit.py \
      --config configs/base_config.yaml \
      --methods ga npo scrub salun rmu alpha_edit \
      --datasets tofu

  # Multi-dataset audit — GA + SalUn + DurableUn on all 3 datasets:
  python experiments/multi_dataset_audit.py \
      --config configs/durableun_config.yaml \
      --methods ga salun durableun_saf_v3 durableun_saf_alpha3 \
      --datasets tofu muse_news wpu

  # Just modern baselines on TOFU:
  python experiments/multi_dataset_audit.py \
      --config configs/base_config.yaml \
      --methods graddiff wga tv dare noisy_ga langevin \
      --datasets tofu

  # Resume after crash (skips existing checkpoints):
  python experiments/multi_dataset_audit.py ... --resume

Expected runtimes (RTX 4090):
  ga/scrub/npo/graddiff/wga/noisy_ga/langevin:  ~20-40 min each
  salun:                                          ~25 min
  rmu:                                            ~11 hours (skip unless needed)
  tv/dare:                                        ~1 min (training-free)
  durableun_saf_v3:                               ~25 min
  durableun_saf_alpha3:                           ~350 min
"""

import argparse, csv, json, logging, os, sys
from datetime import datetime
import torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.logging_utils import setup_root_logger, now_str, file_ts
from src.data.data_utils import set_seed
from src.data.dataset_registry import get_dataloaders, AVAILABLE_DATASETS
from src.models.model_utils import load_model_with_lora
from src.baselines.baseline_registry import (
    get_baseline, BASELINE_NAMES, DISPLAY_NAMES
)
from src.evaluation.evaluator import (
    compute_token_accuracy, compute_quantization_recovery, compute_mia_auc
)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-dataset unlearning audit")
    p.add_argument("--config",   default="configs/base_config.yaml")
    p.add_argument("--methods",  nargs="+", default=["ga", "salun"],
                   choices=BASELINE_NAMES)
    p.add_argument("--datasets", nargs="+", default=["tofu"],
                   choices=AVAILABLE_DATASETS)
    p.add_argument("--seeds",    nargs="+", type=int, default=[42])
    p.add_argument("--n_steps",  type=int, default=None,
                   help="Override n_steps from config")
    p.add_argument("--resume",   action="store_true",
                   help="Skip method+dataset combinations that already have checkpoints")
    p.add_argument("--skip_ft",  action="store_true",
                   help="Skip fine-tuning recovery attack (saves ~10 min per run)")
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


def checkpoint_exists(ckpt_dir, method, dataset, seed):
    path = os.path.join(
        ckpt_dir,
        f"{method}_{dataset}_seed{seed}",
        "result.json"
    )
    return os.path.exists(path)


def run_one(method_name, dataset_name, seed, config, args, logger):
    """Train one method on one dataset at one seed. Returns metrics dict."""
    set_seed(seed)
    logger.info(f"\n  [{method_name}] dataset={dataset_name} seed={seed}")

    model_name = _get(config, "model.name",
                      default="meta-llama/Meta-Llama-3-8B-Instruct")

    model, tokenizer = load_model_with_lora(
        model_name,
        lora_config=config.get("lora"),
        dtype=_get(config, "model.dtype", default="bfloat16"),
        device_map=_get(config, "model.device_map", default="cuda:0"),
        load_in_4bit=_get(config, "model.load_in_4bit", default=True),
        cache_dir=_get(config, "paths.cache_dir"),
    )
    device = _real_device(model)

    # Load dataset
    fl, rl, extra = get_dataloaders(
        tokenizer,
        dataset=dataset_name,
        forget_split=_get(config, "dataset.forget_split", default="forget10"),
        retain_split=_get(config, "dataset.retain_split", default="retain90"),
        max_length=_get(config, "dataset.max_length", default=256),
        batch_size=_get(config, "dataset.batch_size", default=4),
        num_workers=0,
    )

    n_steps = args.n_steps or _get(config, "training.n_steps", default=300)
    lr      = _get(config, "training.lr",              default=5e-5)
    clip    = _get(config, "training.gradient_clip",   default=1.0)
    log_e   = _get(config, "training.log_every",       default=50)

    # Get unlearner
    unlearner = get_baseline(
        method_name,
        model=model, forget_loader=fl, retain_loader=rl,
        device=device, n_steps=n_steps, lr=lr,
        gradient_clip=clip, log_every=log_e,
    )
    result = unlearner.unlearn()

    # Evaluate
    dev   = str(device)
    max_b = _get(config, "eval.max_batches", default=30)
    fa    = compute_token_accuracy(model, fl, dev, max_b)
    ra    = compute_token_accuracy(model, rl, dev, max_b)
    mia   = compute_mia_auc(model, fl, rl, dev)
    quant = compute_quantization_recovery(
        model, fl, dev, ["bf16", "int8", "int4"], max_b
    )

    # RA-INT4 (retain under INT4)
    try:
        from src.evaluation.evaluator_additions import compute_token_accuracy_quantized
        ra_int4 = compute_token_accuracy_quantized(model, rl, dev, "int4", max_b)
    except Exception:
        ra_int4 = -1.0

    # Fine-tuning recovery attack (optional)
    ft50 = -1.0
    if not args.skip_ft:
        try:
            from src.evaluation.evaluator import compute_finetuning_recovery
            from src.data.data_utils import get_downstream_dataloader
            alpaca = get_downstream_dataloader(
                tokenizer, datasets=["alpaca"], n_samples_per_dist=200,
                max_length=256, batch_size=4, num_workers=0,
            )
            ft_res = compute_finetuning_recovery(
                model, tokenizer, fl, alpaca, dev,
                steps_list=[50], max_eval_batches=max_b,
            )
            ft50 = ft_res.get(50, -1.0)
        except Exception as e:
            logger.warning(f"FT attack failed: {e}")

    row = {
        "method":        method_name,
        "method_display": DISPLAY_NAMES.get(method_name, method_name),
        "dataset":       dataset_name,
        "seed":          seed,
        "n_steps":       n_steps,
        "forget_acc":    round(fa, 4),
        "retain_acc":    round(ra, 4),
        "mia_auc":       round(mia, 4),
        "quant_bf16":    round(quant.get("bf16", -1), 4),
        "quant_int8":    round(quant.get("int8", -1), 4),
        "quant_int4":    round(quant.get("int4", -1), 4),
        "ra_int4":       round(ra_int4, 4),
        "ft_50steps":    round(ft50, 4),
        "wall_time_min": round(result.wall_time_seconds / 60, 1),
        "evaluated_at":  now_str(),
    }

    logger.info(
        f"  FA={fa:.4f}  RA={ra:.4f}  "
        f"Q-INT8={quant.get('int8',-1):.4f}  Q-INT4={quant.get('int4',-1):.4f}  "
        f"RA-INT4={ra_int4:.4f}"
    )

    # Save checkpoint
    ckpt_dir = _get(config, "paths.checkpoints", default="checkpoints")
    ckpt_path = os.path.join(ckpt_dir, f"{method_name}_{dataset_name}_seed{seed}")
    os.makedirs(ckpt_path, exist_ok=True)
    model.save_pretrained(os.path.join(ckpt_path, "model"))
    tokenizer.save_pretrained(os.path.join(ckpt_path, "model"))
    with open(os.path.join(ckpt_path, "result.json"), "w") as f:
        json.dump(row, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return row


def main():
    args   = parse_args()
    config = load_config(args.config)

    setup_root_logger(_get(config, "paths.logs", default="logs"))
    logger = logging.getLogger("multi_dataset_audit")

    res_dir  = _get(config, "paths.results",     default="results")
    ckpt_dir = _get(config, "paths.checkpoints", default="checkpoints")
    os.makedirs(res_dir, exist_ok=True)

    results_csv = os.path.join(res_dir, f"multi_dataset_audit_{file_ts()}.csv")

    logger.info(f"\n{'='*65}")
    logger.info(f"  Multi-Dataset Unlearning Audit")
    logger.info(f"  Methods:  {args.methods}")
    logger.info(f"  Datasets: {args.datasets}")
    logger.info(f"  Seeds:    {args.seeds}")
    logger.info(f"{'='*65}\n")

    all_rows = []

    for dataset in args.datasets:
        for method in args.methods:
            for seed in args.seeds:
                # Resume check
                if args.resume and checkpoint_exists(ckpt_dir, method, dataset, seed):
                    logger.info(f"  SKIP {method}/{dataset}/seed{seed} — checkpoint exists")
                    # Load existing result
                    result_path = os.path.join(
                        ckpt_dir, f"{method}_{dataset}_seed{seed}", "result.json"
                    )
                    with open(result_path) as f:
                        all_rows.append(json.load(f))
                    continue

                try:
                    row = run_one(method, dataset, seed, config, args, logger)
                    all_rows.append(row)

                    write_hdr = not os.path.exists(results_csv)
                    with open(results_csv, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
                        if write_hdr: w.writeheader()
                        w.writerow(row)

                except Exception as e:
                    logger.error(
                        f"FAILED: {method}/{dataset}/seed{seed}: {e}", exc_info=True
                    )
                    continue

    # ── Final summary table ───────────────────────────────────────────────────
    logger.info(f"\n{'='*80}")
    logger.info("MULTI-DATASET AUDIT RESULTS")
    logger.info(f"{'='*80}")

    for dataset in args.datasets:
        ds_rows = [r for r in all_rows if r.get("dataset") == dataset]
        if not ds_rows:
            continue

        logger.info(f"\n  Dataset: {dataset.upper()}")
        logger.info(
            f"  {'Method':<22} {'FA↓':>7} {'RA↑':>7} {'Q-INT8↓':>8} "
            f"{'Q-INT4↓':>8} {'RA-INT4↑':>9} {'FT@50↓':>8} {'Cert.':>6}"
        )
        logger.info("  " + "-"*75)

        for row in ds_rows:
            cert = "✓" if row.get("quant_int4", 1.0) <= 0.05 else "✗"
            logger.info(
                f"  {row.get('method_display','?'):<22} "
                f"{row.get('forget_acc',-1):>7.4f} "
                f"{row.get('retain_acc',-1):>7.4f} "
                f"{row.get('quant_int8',-1):>8.4f} "
                f"{row.get('quant_int4',-1):>8.4f} "
                f"{row.get('ra_int4',-1):>9.4f} "
                f"{row.get('ft_50steps',-1):>8.4f} "
                f"{cert:>6}"
            )

    logger.info(f"\nFull results CSV: {results_csv}")


if __name__ == "__main__":
    main()
