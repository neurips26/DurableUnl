"""
experiments/multi_seed_eval.py
================================
Multi-seed evaluation for statistical reliability.
Adds RA-INT4 column (addresses reviewer "missing metrics" concern).

Usage:
  python experiments/multi_seed_eval.py \
      --config configs/durableun_config.yaml \
      --methods ga salun durableun_saf_alpha1 durableun_saf_alpha3 \
      --seeds 42 123 5508

Expected runtime: ~40 min per method per seed on RTX 4090.
Results saved incrementally — safe to interrupt and resume.
"""

import argparse, csv, json, logging, os, sys, statistics
import torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.logging_utils import setup_root_logger, now_str, file_ts
from src.data.data_utils import set_seed
from src.data.tofu_dataset import get_tofu_dataloaders
from src.models.model_utils import load_model_with_lora


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="configs/durableun_config.yaml")
    p.add_argument("--methods", nargs="+",
                   choices=["ga","salun","scrub","npo","graddiff",
                             "durableun_saf_alpha1","durableun_saf_alpha3"],
                   default=["ga","salun"])
    p.add_argument("--seeds",   nargs="+", type=int, default=[42, 123, 5508])
    p.add_argument("--split",   default="forget10")
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


def run_one(method_name, seed, config, split, logger):
    """Train + evaluate one method at one seed. Returns metrics dict."""
    set_seed(seed)
    logger.info(f"\n  [{method_name}] seed={seed} split={split}")

    model, tokenizer = load_model_with_lora(
        _get(config, "model.name", default="meta-llama/Meta-Llama-3-8B-Instruct"),
        lora_config=config.get("lora"),
        dtype=_get(config, "model.dtype", default="bfloat16"),
        device_map=_get(config, "model.device_map", default="cuda:0"),
        load_in_4bit=_get(config, "model.load_in_4bit", default=True),
    )
    device = _real_device(model)

    retain_split = "retain" + str(100 - int(split.replace("forget", "")))
    fl, rl, _ = get_tofu_dataloaders(
        tokenizer,
        forget_split=split, retain_split=retain_split,
        batch_size=4, max_length=256, num_workers=0,
    )

    n_steps = _get(config, "training.n_steps", default=300)
    lr      = _get(config, "training.lr",      default=5e-5)

    # ── Train ────────────────────────────────────────────────────────────────
    if method_name == "ga":
        from src.baselines.base import _clm_loss
        try:
            from src.baselines.ga import GA
            GA(model=model, forget_loader=fl, retain_loader=rl,
               device=device, n_steps=n_steps, lr=lr, retain_lambda=1.0).unlearn()
        except ImportError:
            # Inline GA if module missing
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
                fb = {k: v.to(device) if hasattr(v, "to") else v for k, v in next(fi).items()}
                rb = {k: v.to(device) if hasattr(v, "to") else v for k, v in next(ri).items()}
                (-_clm_loss(model, fb) + _clm_loss(model, rb)).backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sch.step()

    elif method_name == "salun":
        from src.baselines.salun import SalUn
        SalUn(model=model, forget_loader=fl, retain_loader=rl,
              device=device, n_steps=n_steps, lr=lr, retain_lambda=1.0).unlearn()

    elif method_name == "scrub":
        from src.baselines.scrub import SCRUB
        SCRUB(model=model, forget_loader=fl, retain_loader=rl,
              device=device, n_steps=n_steps, lr=lr, retain_lambda=1.0).unlearn()

    elif method_name == "npo":
        from src.baselines.npo import NPO
        NPO(model=model, forget_loader=fl, retain_loader=rl,
            device=device, n_steps=n_steps, lr=lr, retain_lambda=1.0).unlearn()

    elif method_name == "graddiff":
        from src.baselines.gradient_difference import GradDiff
        GradDiff(model=model, forget_loader=fl, retain_loader=rl,
                 device=device, n_steps=n_steps, lr=lr, retain_lambda=1.0).unlearn()

    elif method_name in ["durableun_saf_alpha1", "durableun_saf_alpha3"]:
        from src.durableun.saf import SAF
        alpha = 1.0 if "alpha1" in method_name else 3.0
        SAF(model=model, forget_loader=fl, retain_loader=rl,
            device=device, n_steps=n_steps, lr=lr,
            retain_lambda=max(1.0, alpha + 1.0),
            alpha_quant=alpha, warmup_steps=100).unlearn()

    # ── Evaluate ─────────────────────────────────────────────────────────────
    from src.evaluation.evaluator import (
        compute_token_accuracy, compute_quantization_recovery, compute_mia_auc
    )
    from src.evaluation.evaluator_additions import compute_token_accuracy_quantized

    dev   = str(device)
    max_b = _get(config, "eval.max_batches", default=30)
    fa    = compute_token_accuracy(model, fl, dev, max_b)
    ra    = compute_token_accuracy(model, rl, dev, max_b)
    mia   = compute_mia_auc(model, fl, rl, dev)
    quant = compute_quantization_recovery(model, fl, dev, ["bf16","int8","int4"], max_b)

    # NEW: RA-INT4 (retain accuracy under INT4 quantization)
    ra_int4 = compute_token_accuracy_quantized(model, rl, dev, "int4", max_b)

    result = {
        "method":      method_name,
        "seed":        seed,
        "split":       split,
        "forget_acc":  round(fa,    4),
        "retain_acc":  round(ra,    4),
        "mia_auc":     round(mia,   4),
        "quant_bf16":  round(quant.get("bf16", -1), 4),
        "quant_int8":  round(quant.get("int8", -1), 4),
        "quant_int4":  round(quant.get("int4", -1), 4),
        "ra_int4":     round(ra_int4, 4),
        "evaluated_at": now_str(),
    }
    logger.info(
        f"  FA={fa:.4f}  RA={ra:.4f}  Q-INT4={quant.get('int4',-1):.4f}  RA-INT4={ra_int4:.4f}"
    )

    del model
    torch.cuda.empty_cache()
    return result


def main():
    args   = parse_args()
    config = load_config(args.config)

    setup_root_logger(_get(config, "paths.logs", default="logs"))
    logger = logging.getLogger("multi_seed_eval")
    os.makedirs(_get(config, "paths.results", default="results"), exist_ok=True)
    results_csv = os.path.join(
        _get(config, "paths.results", default="results"),
        f"multi_seed_{file_ts()}.csv"
    )
    logger.info(f"Multi-seed eval | methods={args.methods} | seeds={args.seeds} | split={args.split}")

    for method in args.methods:
        method_rows = []
        for seed in args.seeds:
            row = run_one(method, seed, config, args.split, logger)
            method_rows.append(row)

            write_hdr = not os.path.exists(results_csv)
            with open(results_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
                if write_hdr: w.writeheader()
                w.writerow(row)

        # Mean ± std summary
        logger.info(f"\n{'─'*55}")
        logger.info(f"  {method} | mean ± std over seeds {args.seeds}")
        for metric in ["forget_acc","retain_acc","quant_int4","ra_int4"]:
            vals = [r[metric] for r in method_rows if r[metric] >= 0]
            if len(vals) >= 2:
                logger.info(
                    f"    {metric:<14}: {statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"
                )

    # Final cross-method summary
    logger.info(f"\n{'='*65}")
    logger.info("FINAL MULTI-SEED SUMMARY")
    logger.info(f"{'='*65}")
    logger.info(f"{'Method':<25} {'FA (mean±std)':>16} {'Q-INT4 (mean±std)':>20} {'RA-INT4':>10}")
    logger.info("-"*65)

    logger.info(f"\nCSV saved: {results_csv}")


if __name__ == "__main__":
    main()
