"""
experiments/revision_multiseed.py
===================================
Reviewer ask: "multi-seed validation."
Runs SalUn and DurableUn-SAF alpha=3 across seeds {42, 123, 5508} on TOFU.

This is the most important revision experiment. It answers:
  - Does the certificate (Q-INT4 <= 0.05) hold consistently for SAF alpha=3?
  - How variable is SalUn's Q-INT4=0.051 (borderline)?

Usage:
  py -m experiments.revision_multiseed

Expected runtime: ~25 min per method per seed = ~2.5 hours total.
Results saved incrementally — safe to interrupt.
"""

import csv, json, logging, os, sys, statistics, time
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

CONFIG_PATH = os.path.join(ROOT, "configs", "base_config.yaml")
DURABLE_CFG = os.path.join(ROOT, "configs", "durableun_config.yaml")


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


def run_one(method, seed, logger):
    """Train one method at one seed. Return metrics dict."""
    cfg = load_config(DURABLE_CFG if "saf" in method else CONFIG_PATH)
    model_name = _get(cfg, "model.name",
                      default="meta-llama/Meta-Llama-3-8B-Instruct")

    # Check if result already saved
    ckpt_dir  = _get(cfg, "paths.checkpoints", default="checkpoints")
    result_f  = os.path.join(ckpt_dir, f"{method}_tofu_s{seed}", "result.json")
    if os.path.exists(result_f):
        logger.info(f"  SKIP {method}/s{seed} — result exists")
        with open(result_f) as f:
            return json.load(f)

    logger.info(f"\n{'─'*50}\n  {method} | seed={seed}\n{'─'*50}")
    set_seed(seed)

    model, tokenizer = load_model_with_lora(
        model_name,
        lora_config=cfg.get("lora"),
        dtype=_get(cfg, "model.dtype", default="bfloat16"),
        device_map=_get(cfg, "model.device_map", default="cuda:0"),
        load_in_4bit=_get(cfg, "model.load_in_4bit", default=True),
        cache_dir=_get(cfg, "paths.cache_dir"),
    )
    device  = _device(model)
    n_steps = _get(cfg, "training.n_steps", default=300)
    lr      = _get(cfg, "training.lr",      default=5e-5)

    fl, rl, _ = get_tofu_dataloaders(
        tokenizer,
        forget_split=_get(cfg, "dataset.forget_split", default="forget10"),
        retain_split=_get(cfg, "dataset.retain_split", default="retain90"),
        batch_size=4, max_length=256, num_workers=0,
    )

    t0 = time.time()

    if method == "salun":
        from src.baselines.salun import SalUn
        SalUn(model=model, forget_loader=fl, retain_loader=rl,
              device=device, n_steps=n_steps, lr=lr,
              retain_lambda=1.0, gradient_clip=1.0, log_every=50).unlearn()

    elif method == "saf_alpha3":
        from src.durableun.saf import SAF
        SAF(model=model, forget_loader=fl, retain_loader=rl,
            device=device, n_steps=n_steps, lr=lr,
            retain_lambda=4.0, alpha_quant=3.0, warmup_steps=100,
            gradient_clip=1.0, log_every=50).unlearn()

    wall_min = (time.time() - t0) / 60

    # Evaluate
    dev   = str(device)
    max_b = _get(cfg, "eval.max_batches", default=30)
    fa    = compute_token_accuracy(model, fl, dev, max_b)
    ra    = compute_token_accuracy(model, rl, dev, max_b)
    mia   = compute_mia_auc(model, fl, rl, dev)
    quant = compute_quantization_recovery(model, fl, dev,
                                          ["bf16","int8","int4"], max_b)

    # RA-INT4
    try:
        from src.evaluation.evaluator_additions import compute_token_accuracy_quantized
        ra_int4 = compute_token_accuracy_quantized(model, rl, dev, "int4", max_b)
    except Exception:
        ra_int4 = -1.0

    row = {
        "method":     method,
        "dataset":    "tofu",
        "seed":       seed,
        "forget_acc": round(fa,    4),
        "retain_acc": round(ra,    4),
        "mia_auc":    round(mia,   4),
        "quant_int8": round(quant.get("int8", -1), 4),
        "quant_int4": round(quant.get("int4", -1), 4),
        "ra_int4":    round(ra_int4, 4),
        "cert":       "Y" if quant.get("int4", 1.0) <= 0.05 else "N",
        "wall_min":   round(wall_min, 1),
        "timestamp":  now_str(),
    }

    logger.info(
        f"  FA={fa:.4f}  RA={ra:.4f}  "
        f"Q-INT4={quant.get('int4',-1):.4f}  RA-INT4={ra_int4:.4f}  "
        f"cert={row['cert']}"
    )

    # Save checkpoint
    cp = os.path.join(ckpt_dir, f"{method}_tofu_s{seed}")
    os.makedirs(cp, exist_ok=True)
    model.save_pretrained(os.path.join(cp, "model"))
    tokenizer.save_pretrained(os.path.join(cp, "model"))
    with open(result_f, "w") as f:
        json.dump(row, f, indent=2)

    del model; torch.cuda.empty_cache()
    return row


def main():
    setup_root_logger("logs")
    logger  = logging.getLogger("revision_multiseed")
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", f"revision_multiseed_{file_ts()}.csv")

    methods = ["salun", "saf_alpha3"]
    seeds   = [42, 123, 5508]

    logger.info("=== Revision: Multi-Seed Validation ===")
    logger.info(f"Methods: {methods}")
    logger.info(f"Seeds:   {seeds}")

    all_rows = []

    for method in methods:
        method_rows = []
        for seed in seeds:
            row = run_one(method, seed, logger)
            method_rows.append(row)
            all_rows.append(row)

            write_hdr = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
                if write_hdr: w.writeheader()
                w.writerow(row)

        # Print mean ± std
        logger.info(f"\n{'='*55}")
        logger.info(f"  {method} | mean ± std over seeds {seeds}")
        for metric in ["forget_acc", "retain_acc", "quant_int4", "ra_int4"]:
            vals = [r[metric] for r in method_rows if r.get(metric, -1) >= 0]
            if len(vals) >= 2:
                m, s = statistics.mean(vals), statistics.stdev(vals)
                cert_rate = sum(1 for r in method_rows if r["cert"] == "Y")
                logger.info(f"    {metric:<14}: {m:.4f} ± {s:.4f}")
        logger.info(
            f"    cert rate     : "
            f"{sum(1 for r in method_rows if r['cert']=='Y')}/{len(seeds)}"
        )

    # Final summary table
    logger.info(f"\n{'='*65}")
    logger.info("MULTI-SEED SUMMARY (for paper Table)")
    logger.info(f"{'='*65}")
    logger.info(f"{'Method':<14} {'FA':>16} {'RA':>16} {'Q-INT4':>16} {'Cert'}")
    logger.info("-"*65)
    for method in methods:
        rows = [r for r in all_rows if r["method"] == method]
        for metric_name, metric_key in [("FA","forget_acc"),("RA","retain_acc"),("Q-INT4","quant_int4")]:
            vals = [r[metric_key] for r in rows if r.get(metric_key,-1) >= 0]
        fa_v   = [r["forget_acc"] for r in rows]
        ra_v   = [r["retain_acc"] for r in rows]
        qi4_v  = [r["quant_int4"] for r in rows]
        cr     = sum(1 for r in rows if r["cert"]=="Y")
        logger.info(
            f"  {method:<12} "
            f"  {statistics.mean(fa_v):.3f}±{statistics.stdev(fa_v):.3f}"
            f"  {statistics.mean(ra_v):.3f}±{statistics.stdev(ra_v):.3f}"
            f"  {statistics.mean(qi4_v):.3f}±{statistics.stdev(qi4_v):.3f}"
            f"  {cr}/{len(seeds)}"
        )

    logger.info(f"\nCSV: {csv_path}")
    logger.info("\nLaTeX snippet for paper:")
    for method in methods:
        rows = [r for r in all_rows if r["method"] == method]
        fa_v   = [r["forget_acc"] for r in rows]
        ra_v   = [r["retain_acc"] for r in rows]
        qi4_v  = [r["quant_int4"] for r in rows]
        cr     = sum(1 for r in rows if r["cert"]=="Y")
        name   = "SalUn" if method == "salun" else r"\ourmethod{} $\alpha$=3"
        logger.info(
            f"  {name} & "
            f"${statistics.mean(fa_v):.3f}\\pm{statistics.stdev(fa_v):.3f}$ & "
            f"${statistics.mean(ra_v):.3f}\\pm{statistics.stdev(ra_v):.3f}$ & "
            f"${statistics.mean(qi4_v):.3f}\\pm{statistics.stdev(qi4_v):.3f}$ & "
            f"{cr}/{len(seeds)} \\\\"
        )


if __name__ == "__main__":
    main()
