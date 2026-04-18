"""
experiments/priority_audit.py
================================
The focused experiment matrix for the revised paper.

Main table (run these first):
    Methods:  ga, salun, graddiff, durableun_saf_v3, durableun_saf_alpha3
    Datasets: tofu, muse_news, wpu

Appendix table (run after, time permitting):
    Methods:  wga, tv, dare
    Datasets: tofu, muse_news

Usage:

  # Step 1 — Main table (run first, ~6 hours total):
  python experiments/priority_audit.py --tier main --datasets tofu

  # Step 2 — Main table on MUSE-News (~6 hours):
  python experiments/priority_audit.py --tier main --datasets muse_news

  # Step 3 — Main table on WikiBio (~6 hours):
  python experiments/priority_audit.py --tier main --datasets wpu

  # Step 4 — Appendix baselines on TOFU (~2 hours, training-free methods are fast):
  python experiments/priority_audit.py --tier appendix --datasets tofu

  # Resume any interrupted run:
  python experiments/priority_audit.py --tier main --datasets tofu --resume

  # Single method/dataset for testing:
  python experiments/priority_audit.py --methods graddiff --datasets tofu

Runtimes per method (RTX 4090, 300 steps):
  ga:                    ~8  min
  salun:                 ~25 min  (saliency pre-compute + training)
  graddiff:              ~12 min
  wga:                   ~12 min
  tv / dare:             ~1  min  (training-free)
  durableun_saf_v3:      ~25 min
  durableun_saf_alpha3:  ~350 min
"""

import argparse, csv, json, logging, os, sys, time
from datetime import datetime
import torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.logging_utils import setup_root_logger, now_str, file_ts
from src.data.data_utils import set_seed
from src.data.dataset_registry import get_dataloaders
from src.models.model_utils import load_model_with_lora
from src.evaluation.evaluator import (
    compute_token_accuracy, compute_quantization_recovery, compute_mia_auc
)

logger = logging.getLogger(__name__)

# ── Priority tiers ────────────────────────────────────────────────────────────
TIERS = {
    "main": [
        "ga",
        "salun",
        "graddiff",
        "durableun_saf_v3",
        "durableun_saf_alpha3",
    ],
    "appendix": [
        "wga",
        "tv",
        "dare",
    ],
}

ALL_METHODS = TIERS["main"] + TIERS["appendix"]

DISPLAY = {
    "ga":                   "GA",
    "salun":                "SalUn",
    "graddiff":             "GradDiff",
    "wga":                  "WGA",
    "tv":                   "Task Vector",
    "dare":                 "DARE",
    "durableun_saf_v3":    "DurableUn-SAF v3",
    "durableun_saf_alpha3": "DurableUn-SAF α=3",
}


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="configs/base_config.yaml")
    p.add_argument("--tier",     choices=["main","appendix","all"],
                   default=None, help="Run a preset tier of methods")
    p.add_argument("--methods",  nargs="+", default=None,
                   choices=ALL_METHODS,
                   help="Override: specific methods to run")
    p.add_argument("--datasets", nargs="+",
                   default=["tofu"],
                   choices=["tofu","muse_news","muse_books","wpu"])
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--n_steps",  type=int, default=None)
    p.add_argument("--resume",   action="store_true",
                   help="Skip runs that already have a saved result.json")
    p.add_argument("--skip_ft",  action="store_true",
                   help="Skip fine-tuning attack (saves ~10 min per run)")
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


def _device(model):
    for p in model.parameters():
        if p.device.type != "meta": return p.device
    return torch.device("cuda")


def _ckpt_path(ckpt_dir, method, dataset, seed):
    return os.path.join(ckpt_dir, f"{method}_{dataset}_s{seed}")


def _result_exists(ckpt_dir, method, dataset, seed):
    return os.path.exists(
        os.path.join(_ckpt_path(ckpt_dir, method, dataset, seed), "result.json")
    )


# ── Per-method training ───────────────────────────────────────────────────────
def _train(method, model, fl, rl, device, n_steps, lr, clip, log_every, logger):
    """Dispatch to the right unlearner. Returns wall_time_seconds."""
    t0 = time.time()

    if method == "ga":
        from src.baselines.base import _clm_loss
        try:
            from src.baselines.ga import GA
            GA(model=model, forget_loader=fl, retain_loader=rl,
               device=device, n_steps=n_steps, lr=lr, retain_lambda=1.0,
               gradient_clip=clip, log_every=log_every).unlearn()
        except ImportError:
            _ga_inline(model, fl, rl, device, n_steps, lr, clip)

    elif method == "salun":
        from src.baselines.salun import SalUn
        SalUn(model=model, forget_loader=fl, retain_loader=rl,
              device=device, n_steps=n_steps, lr=lr, retain_lambda=1.0,
              gradient_clip=clip, log_every=log_every).unlearn()

    elif method == "graddiff":
        from src.baselines.gradient_difference import GradDiff
        GradDiff(model=model, forget_loader=fl, retain_loader=rl,
                 device=device, n_steps=n_steps, lr=lr, retain_lambda=1.0,
                 gradient_clip=clip, log_every=log_every).unlearn()

    elif method == "wga":
        from src.baselines.wga import WGA
        WGA(model=model, forget_loader=fl, retain_loader=rl,
            device=device, n_steps=n_steps, lr=lr, retain_lambda=1.0,
            gradient_clip=clip, log_every=log_every,
            variant="weighted", temperature=1.0).unlearn()

    elif method == "tv":
        from src.baselines.tv_distance import TaskVectorUnlearning
        TaskVectorUnlearning(model=model, forget_loader=fl, retain_loader=rl,
                             device=device, scale=1.0, method="negate").unlearn()

    elif method == "dare":
        from src.baselines.tv_distance import TaskVectorUnlearning
        TaskVectorUnlearning(model=model, forget_loader=fl, retain_loader=rl,
                             device=device, scale=1.0, method="dare",
                             dare_p=0.9).unlearn()

    elif method == "durableun_saf_v3":
        from src.durableun.saf import SAF
        SAF(model=model, forget_loader=fl, retain_loader=rl,
            device=device, n_steps=n_steps, lr=lr,
            retain_lambda=2.0, alpha_quant=1.0, warmup_steps=100,
            gradient_clip=clip, log_every=log_every).unlearn()

    elif method == "durableun_saf_alpha3":
        from src.durableun.saf import SAF
        SAF(model=model, forget_loader=fl, retain_loader=rl,
            device=device, n_steps=n_steps, lr=lr,
            retain_lambda=4.0, alpha_quant=3.0, warmup_steps=100,
            gradient_clip=clip, log_every=log_every).unlearn()

    else:
        raise ValueError(f"Unknown method: {method}")

    return time.time() - t0


def _ga_inline(model, fl, rl, device, n_steps, lr, clip):
    """Inline GA fallback if src.baselines.ga missing."""
    from src.baselines.base import _clm_loss
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from tqdm import tqdm

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sch = CosineAnnealingLR(opt, T_max=n_steps)

    def inf(loader):
        while True:
            for b in loader: yield b
    fi = inf(fl); ri = inf(rl)

    for _ in tqdm(range(n_steps), desc="GA"):
        opt.zero_grad()
        fb = {k: v.to(device) if hasattr(v,"to") else v for k,v in next(fi).items()}
        rb = {k: v.to(device) if hasattr(v,"to") else v for k,v in next(ri).items()}
        (-_clm_loss(model, fb) + _clm_loss(model, rb)).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], clip)
        opt.step(); sch.step()


# ── Evaluation ────────────────────────────────────────────────────────────────
def _evaluate(model, fl, rl, device, config, skip_ft, tokenizer):
    max_b  = _get(config, "eval.max_batches", default=30)
    dev    = str(device)
    precs  = ["bf16", "int8", "int4"]

    fa    = compute_token_accuracy(model, fl, dev, max_b)
    ra    = compute_token_accuracy(model, rl, dev, max_b)
    mia   = compute_mia_auc(model, fl, rl, dev)
    quant = compute_quantization_recovery(model, fl, dev, precs, max_b)

    # RA-INT4
    try:
        from src.evaluation.evaluator_additions import compute_token_accuracy_quantized
        ra_int4 = compute_token_accuracy_quantized(model, rl, dev, "int4", max_b)
    except Exception:
        ra_int4 = -1.0

    ft50 = -1.0
    if not skip_ft:
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

    return {
        "forget_acc": round(fa, 4),
        "retain_acc": round(ra, 4),
        "mia_auc":    round(mia, 4),
        "quant_bf16": round(quant.get("bf16", -1), 4),
        "quant_int8": round(quant.get("int8", -1), 4),
        "quant_int4": round(quant.get("int4", -1), 4),
        "ra_int4":    round(ra_int4, 4),
        "ft_50":      round(ft50, 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    config = load_config(args.config)

    setup_root_logger(_get(config, "paths.logs", default="logs"))
    logger = logging.getLogger("priority_audit")

    res_dir  = _get(config, "paths.results",     default="results")
    ckpt_dir = _get(config, "paths.checkpoints", default="checkpoints")
    os.makedirs(res_dir, exist_ok=True)

    # Determine methods
    if args.methods:
        methods = args.methods
    elif args.tier == "all":
        methods = ALL_METHODS
    elif args.tier:
        methods = TIERS[args.tier]
    else:
        methods = TIERS["main"]

    datasets = args.datasets
    seed     = args.seed
    n_steps  = args.n_steps or _get(config, "training.n_steps", default=300)
    lr       = _get(config, "training.lr",            default=5e-5)
    clip     = _get(config, "training.gradient_clip", default=1.0)
    log_e    = _get(config, "training.log_every",     default=50)

    results_csv = os.path.join(res_dir, f"priority_audit_{file_ts()}.csv")

    logger.info(f"\n{'='*60}")
    logger.info(f"  Priority Audit")
    logger.info(f"  Methods:  {methods}")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Seed:     {seed}  |  Steps: {n_steps}")
    logger.info(f"{'='*60}\n")

    model_name = _get(config, "model.name",
                      default="meta-llama/Meta-Llama-3-8B-Instruct")
    all_rows = []

    for dataset in datasets:
        for method in methods:
            run_id = f"{method}/{dataset}/s{seed}"

            # Resume check
            if args.resume and _result_exists(ckpt_dir, method, dataset, seed):
                logger.info(f"  SKIP {run_id} — result exists")
                with open(os.path.join(
                    _ckpt_path(ckpt_dir, method, dataset, seed), "result.json"
                )) as f:
                    all_rows.append(json.load(f))
                continue

            logger.info(f"\n{'─'*55}")
            logger.info(f"  RUN: {run_id}")
            logger.info(f"{'─'*55}")

            try:
                set_seed(seed)
                model, tokenizer = load_model_with_lora(
                    model_name,
                    lora_config=config.get("lora"),
                    dtype=_get(config, "model.dtype", default="bfloat16"),
                    device_map=_get(config, "model.device_map", default="cuda:0"),
                    load_in_4bit=_get(config, "model.load_in_4bit", default=True),
                    cache_dir=_get(config, "paths.cache_dir"),
                )
                device = _device(model)

                fl, rl, _ = get_dataloaders(
                    tokenizer, dataset=dataset,
                    forget_split=_get(config, "dataset.forget_split", default="forget10"),
                    retain_split=_get(config, "dataset.retain_split", default="retain90"),
                    max_length=_get(config, "dataset.max_length", default=256),
                    batch_size=_get(config, "dataset.batch_size", default=4),
                    num_workers=0,
                )

                wall_sec = _train(method, model, fl, rl, device,
                                  n_steps, lr, clip, log_e, logger)

                metrics = _evaluate(model, fl, rl, device, config,
                                    args.skip_ft, tokenizer)

                row = {
                    "method":        method,
                    "method_display": DISPLAY.get(method, method),
                    "dataset":       dataset,
                    "seed":          seed,
                    "n_steps":       n_steps,
                    "wall_min":      round(wall_sec / 60, 1),
                    **metrics,
                    "cert":          "Y" if metrics["quant_int4"] <= 0.05 else "N",
                    "evaluated_at":  now_str(),
                }
                all_rows.append(row)

                # Save result.json
                cp = _ckpt_path(ckpt_dir, method, dataset, seed)
                os.makedirs(cp, exist_ok=True)
                model.save_pretrained(os.path.join(cp, "model"))
                tokenizer.save_pretrained(os.path.join(cp, "model"))
                with open(os.path.join(cp, "result.json"), "w") as f:
                    json.dump(row, f, indent=2)

                # Append to CSV
                write_hdr = not os.path.exists(results_csv)
                with open(results_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
                    if write_hdr: w.writeheader()
                    w.writerow(row)

                logger.info(
                    f"\n  {run_id} done | "
                    f"FA={metrics['forget_acc']:.4f}  "
                    f"RA={metrics['retain_acc']:.4f}  "
                    f"Q-INT4={metrics['quant_int4']:.4f}  "
                    f"RA-INT4={metrics['ra_int4']:.4f}  "
                    f"cert={row['cert']}  "
                    f"time={row['wall_min']}min"
                )

            except Exception as e:
                logger.error(f"FAILED: {run_id}: {e}", exc_info=True)
                torch.cuda.empty_cache()
                continue

            del model
            torch.cuda.empty_cache()

    # ── Print final tables ────────────────────────────────────────────────────
    for dataset in datasets:
        rows = [r for r in all_rows if r.get("dataset") == dataset]
        if not rows: continue

        logger.info(f"\n{'='*80}")
        logger.info(f"  {dataset.upper()} RESULTS")
        logger.info(f"{'='*80}")
        logger.info(
            f"  {'Method':<24} {'FA↓':>6} {'RA↑':>6} "
            f"{'Q-INT8↓':>8} {'Q-INT4↓':>8} {'RA-INT4↑':>9} {'FT@50↓':>7} {'Cert':>5}"
        )
        logger.info("  " + "-"*75)
        for r in rows:
            logger.info(
                f"  {r.get('method_display','?'):<24} "
                f"{r.get('forget_acc',-1):>6.4f} "
                f"{r.get('retain_acc',-1):>6.4f} "
                f"{r.get('quant_int8',-1):>8.4f} "
                f"{r.get('quant_int4',-1):>8.4f} "
                f"{r.get('ra_int4',-1):>9.4f} "
                f"{r.get('ft_50',-1):>7.4f} "
                f"{r.get('cert','?'):>5}"
            )

    logger.info(f"\nCSV: {results_csv}")


if __name__ == "__main__":
    main()
