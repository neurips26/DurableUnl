"""
experiments/revision_alpha_sweep.py
=====================================
Reviewer ask: "demonstrate a point on the frontier with materially better RA"
              "limited baseline tuning"

This script runs:
  1. Dense SAF alpha sweep: alpha in {1.5, 2.0, 2.0+lambda5} to find a
     better RA/Q-INT4 operating point than alpha=3 (RA=0.045)
  2. SalUn at original paper hyperparameters (lr=1e-4, 500 steps)
     to answer "under-tuned baselines" concern

Expected outcomes:
  - alpha=1.5 or 2.0 may give RA~0.15-0.30 with Q-INT4~0.05-0.08
    This is a materially better point than alpha=3 (RA=0.045)
  - SalUn at original HPs: if it still fails cert, baseline tuning concern answered
  - If SalUn at original HPs passes cert, we report it honestly

Usage:
  py -m experiments.revision_alpha_sweep

  # Run only SAF sweep (skip SalUn rerun):
  py -m experiments.revision_alpha_sweep --skip_salun

  # Run only SalUn rerun:
  py -m experiments.revision_alpha_sweep --skip_saf

Runtime: ~45 min per SAF run, ~25 min SalUn. Total ~3.5 hours.
Results saved incrementally.
"""

import argparse, csv, json, logging, os, sys, time
import torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.logging_utils import setup_root_logger, now_str, file_ts
from src.data.data_utils import set_seed
from src.data.tofu_dataset import get_tofu_dataloaders
from src.models.model_utils import load_model_with_lora
from src.evaluation.evaluator import (
    compute_token_accuracy, compute_quantization_recovery
)

logger = logging.getLogger("revision_alpha_sweep")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip_salun", action="store_true")
    p.add_argument("--skip_saf",   action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_cfg():
    with open(os.path.join(ROOT, "configs", "base_config.yaml")) as f:
        return yaml.safe_load(f)

def load_dcfg():
    with open(os.path.join(ROOT, "configs", "durableun_config.yaml")) as f:
        return yaml.safe_load(f)

def _get(cfg, *keys, default=None):
    for k in keys:
        v = cfg
        try:
            for part in k.split("."): v = v[part]
            return v
        except: pass
    return default

def _device(model):
    for p in model.parameters():
        if p.device.type != "meta": return p.device
    return torch.device("cuda")


# ── SAF runs ──────────────────────────────────────────────────────────────────

# Dense sweep configurations
# (alpha, lambda, label)
# λ = max(1, α+1) is the default formula, but we also test higher λ at α=2
SAF_CONFIGS = [
    (1.5, 2.5, "saf_a1p5_l2p5"),   # α=1.5, default λ
    (2.0, 3.0, "saf_a2p0_l3p0"),   # α=2.0, default λ
    (2.0, 5.0, "saf_a2p0_l5p0"),   # α=2.0, higher λ — may preserve RA better
    (2.5, 3.5, "saf_a2p5_l3p5"),   # α=2.5, default λ
]


def run_saf(alpha, lam, label, seed, cfg, dcfg, fl, rl, device, ckpt_base, res_dir):
    result_f = os.path.join(ckpt_base, f"{label}_tofu_s{seed}", "result.json")
    if os.path.exists(result_f):
        logger.info(f"  SKIP {label} — result exists")
        with open(result_f) as f: return json.load(f)

    logger.info(f"\n{'─'*55}")
    logger.info(f"  SAF α={alpha} λ={lam} | seed={seed}")
    logger.info(f"{'─'*55}")

    set_seed(seed)
    model_name = _get(dcfg, "model.name",
                      default="meta-llama/Meta-Llama-3-8B-Instruct")

    model, tokenizer = load_model_with_lora(
        model_name,
        lora_config=dcfg.get("lora"),
        dtype=_get(dcfg, "model.dtype", default="bfloat16"),
        device_map=_get(dcfg, "model.device_map", default="cuda:0"),
        load_in_4bit=_get(dcfg, "model.load_in_4bit", default=True),
        cache_dir=_get(dcfg, "paths.cache_dir"),
    )

    t0 = time.time()
    from src.durableun.saf import SAF
    SAF(
        model=model, forget_loader=fl, retain_loader=rl,
        device=str(device), n_steps=300, lr=5e-5,
        retain_lambda=lam, alpha_quant=alpha,
        warmup_steps=100, gradient_clip=1.0, log_every=50,
    ).unlearn()
    wall_min = (time.time() - t0) / 60

    dev = str(_device(model))
    fa  = compute_token_accuracy(model, fl, dev, 30)
    ra  = compute_token_accuracy(model, rl, dev, 30)
    q   = compute_quantization_recovery(model, fl, dev, ["bf16","int8","int4"], 30)

    # RA-INT4
    try:
        from src.evaluation.evaluator_additions import compute_token_accuracy_quantized
        ra_int4 = compute_token_accuracy_quantized(model, rl, dev, "int4", 30)
    except Exception:
        ra_int4 = -1.0

    row = {
        "method":     label,
        "alpha":      alpha,
        "lambda":     lam,
        "seed":       seed,
        "forget_acc": round(fa, 4),
        "retain_acc": round(ra, 4),
        "quant_int8": round(q.get("int8", -1), 4),
        "quant_int4": round(q.get("int4", -1), 4),
        "ra_int4":    round(ra_int4, 4),
        "cert":       "Y" if q.get("int4", 1.0) <= 0.05 else "N",
        "wall_min":   round(wall_min, 1),
        "timestamp":  now_str(),
    }

    logger.info(
        f"  FA={fa:.4f}  RA={ra:.4f}  "
        f"Q-INT8={q.get('int8',-1):.4f}  Q-INT4={q.get('int4',-1):.4f}  "
        f"RA-INT4={ra_int4:.4f}  cert={row['cert']}"
    )
    logger.info(f"  Wall time: {wall_min:.1f} min")

    cp = os.path.join(ckpt_base, f"{label}_tofu_s{seed}")
    os.makedirs(cp, exist_ok=True)
    model.save_pretrained(os.path.join(cp, "model"))
    tokenizer.save_pretrained(os.path.join(cp, "model"))
    with open(result_f, "w") as f:
        json.dump(row, f, indent=2)

    del model; torch.cuda.empty_cache()
    return row


# ── SalUn at original hyperparameters ─────────────────────────────────────────

def run_salun_original(seed, cfg, fl, rl, ckpt_base):
    """
    SalUn at original paper hyperparameters from Foster et al. 2024 (ICLR).
    Original paper uses: lr=1e-4, 500 steps, top-5% saliency mask.
    Our baseline used: lr=5e-5, 300 steps (same as all other baselines for
    fairness). This run tests whether the cert result changes with original HPs.
    """
    label   = "salun_orig_hp"
    result_f = os.path.join(ckpt_base, f"{label}_tofu_s{seed}", "result.json")
    if os.path.exists(result_f):
        logger.info(f"  SKIP {label} — result exists")
        with open(result_f) as f: return json.load(f)

    logger.info(f"\n{'─'*55}")
    logger.info(f"  SalUn ORIGINAL HPs (lr=1e-4, 500 steps) | seed={seed}")
    logger.info(f"  (Our baseline used lr=5e-5, 300 steps for fair comparison)")
    logger.info(f"{'─'*55}")

    set_seed(seed)
    model_name = _get(cfg, "model.name",
                      default="meta-llama/Meta-Llama-3-8B-Instruct")

    from src.models.model_utils import load_model_with_lora
    model, tokenizer = load_model_with_lora(
        model_name,
        lora_config=cfg.get("lora"),
        dtype=_get(cfg, "model.dtype", default="bfloat16"),
        device_map=_get(cfg, "model.device_map", default="cuda:0"),
        load_in_4bit=_get(cfg, "model.load_in_4bit", default=True),
        cache_dir=_get(cfg, "paths.cache_dir"),
    )
    device = str(_device(model))

    t0 = time.time()
    from src.baselines.salun import SalUn
    SalUn(
        model=model,
        forget_loader=fl,
        retain_loader=rl,
        device=device,
        n_steps=500,          # original paper: 500 steps
        lr=1e-4,              # original paper: lr=1e-4
        retain_lambda=1.0,
        gradient_clip=1.0,
        log_every=50,
        # saliency_ratio=0.05  # top-5% — use default if not configurable
    ).unlearn()
    wall_min = (time.time() - t0) / 60

    fa  = compute_token_accuracy(model, fl, device, 30)
    ra  = compute_token_accuracy(model, rl, device, 30)
    q   = compute_quantization_recovery(model, fl, device,
                                        ["bf16","int8","int4"], 30)

    try:
        from src.evaluation.evaluator_additions import compute_token_accuracy_quantized
        ra_int4 = compute_token_accuracy_quantized(model, rl, device, "int4", 30)
    except Exception:
        ra_int4 = -1.0

    row = {
        "method":     label,
        "alpha":      None,
        "lambda":     1.0,
        "seed":       seed,
        "forget_acc": round(fa, 4),
        "retain_acc": round(ra, 4),
        "quant_int8": round(q.get("int8", -1), 4),
        "quant_int4": round(q.get("int4", -1), 4),
        "ra_int4":    round(ra_int4, 4),
        "cert":       "Y" if q.get("int4", 1.0) <= 0.05 else "N",
        "wall_min":   round(wall_min, 1),
        "timestamp":  now_str(),
        "note":       "SalUn at original Foster et al. HPs: lr=1e-4, 500 steps",
    }

    logger.info(
        f"  FA={fa:.4f}  RA={ra:.4f}  "
        f"Q-INT8={q.get('int8',-1):.4f}  Q-INT4={q.get('int4',-1):.4f}  "
        f"cert={row['cert']}"
    )
    logger.info(
        f"  Comparison: our baseline (lr=5e-5, 300 steps) gave "
        f"FA=0.011, Q-INT4=0.051, cert=N"
    )

    cp = os.path.join(ckpt_base, f"{label}_tofu_s{seed}")
    os.makedirs(cp, exist_ok=True)
    model.save_pretrained(os.path.join(cp, "model"))
    tokenizer.save_pretrained(os.path.join(cp, "model"))
    with open(result_f, "w") as f:
        json.dump(row, f, indent=2)

    del model; torch.cuda.empty_cache()
    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_root_logger("logs")

    cfg  = load_cfg()
    dcfg = load_dcfg()

    model_name = _get(dcfg, "model.name",
                      default="meta-llama/Meta-Llama-3-8B-Instruct")
    ckpt_base  = _get(cfg, "paths.checkpoints", default="checkpoints")
    res_dir    = _get(cfg, "paths.results",     default="results")
    cache_dir  = _get(cfg, "paths.cache_dir")
    os.makedirs(res_dir,   exist_ok=True)
    os.makedirs(ckpt_base, exist_ok=True)

    # Dataset (shared across all runs)
    from src.models.model_utils import load_tokenizer
    tok = load_tokenizer(model_name, cache_dir)
    fl, rl, _ = get_tofu_dataloaders(
        tok, forget_split="forget10", retain_split="retain90",
        batch_size=4, max_length=256, num_workers=0,
    )
    device = torch.device("cuda:0")

    csv_path  = os.path.join(res_dir, f"revision_alpha_sweep_{file_ts()}.csv")
    all_rows  = []

    # ── SAF dense sweep ───────────────────────────────────────────────────────
    if not args.skip_saf:
        logger.info("\n" + "="*60)
        logger.info("SAF DENSE ALPHA SWEEP")
        logger.info("Goal: find operating point with RA > 0.15 and Q-INT4 < 0.08")
        logger.info("="*60)

        for alpha, lam, label in SAF_CONFIGS:
            row = run_saf(alpha, lam, label, args.seed,
                          cfg, dcfg, fl, rl, device, ckpt_base, res_dir)
            all_rows.append(row)

            write_hdr = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
                if write_hdr: w.writeheader()
                w.writerow(row)

    # ── SalUn at original HPs ─────────────────────────────────────────────────
    if not args.skip_salun:
        logger.info("\n" + "="*60)
        logger.info("SalUn at ORIGINAL paper HPs (lr=1e-4, 500 steps)")
        logger.info("Answers reviewer: 'limited baseline tuning'")
        logger.info("="*60)

        row = run_salun_original(args.seed, cfg, fl, rl, ckpt_base)
        all_rows.append(row)

        write_hdr = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
            if write_hdr: w.writeheader()
            w.writerow(row)

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("COMPLETE RESULTS SUMMARY")
    logger.info(f"{'='*70}")

    # Reference points
    logger.info("\nReference (existing):")
    logger.info(f"  {'Method':<22} {'FA':>6} {'RA':>6} {'Q-INT4':>7} {'Cert':>5}")
    logger.info(f"  {'─'*50}")
    refs = [
        ("SalUn (our HPs)",     0.011, 0.541, 0.051, "N"),
        ("SAF α=1.0 λ=2.0",    0.275, 0.317, 0.060, "N"),
        ("SAF α=3.0 λ=4.0",    0.040, 0.045, 0.044, "Y"),
    ]
    for name, fa, ra, qi4, cert in refs:
        logger.info(f"  {name:<22} {fa:>6.3f} {ra:>6.3f} {qi4:>7.3f} {cert:>5}")

    logger.info("\nNew results:")
    logger.info(f"  {'Method':<22} {'FA':>6} {'RA':>6} {'Q-INT4':>7} {'Cert':>5}")
    logger.info(f"  {'─'*50}")
    for r in all_rows:
        logger.info(
            f"  {r['method']:<22} "
            f"{r.get('forget_acc', -1):>6.3f} "
            f"{r.get('retain_acc', -1):>6.3f} "
            f"{r.get('quant_int4', -1):>7.3f} "
            f"{r.get('cert', '?'):>5}"
        )

    logger.info(f"\nCSV saved: {csv_path}")

    # Paper text snippet
    logger.info("\n" + "="*70)
    logger.info("PAPER IMPLICATIONS")
    logger.info("="*70)

    saf_rows = [r for r in all_rows if "saf" in r.get("method","")]
    if saf_rows:
        # Find best RA with cert
        cert_rows = [r for r in saf_rows if r.get("cert") == "Y"]
        nocert_rows = sorted(saf_rows, key=lambda r: r.get("retain_acc", 0), reverse=True)

        if cert_rows:
            best = max(cert_rows, key=lambda r: r.get("retain_acc", 0))
            logger.info(
                f"\n  Best certified point (new): "
                f"α={best['alpha']} λ={best['lambda']}  "
                f"FA={best['forget_acc']}  RA={best['retain_acc']}  "
                f"Q-INT4={best['quant_int4']}"
            )
            if best['retain_acc'] > 0.10:
                logger.info(
                    f"  -> RA={best['retain_acc']} >> 0.045 (old cert point)")
                logger.info(
                    f"  -> Write: 'At α={best['alpha']}, DurableUn-SAF achieves "
                    f"RA={best['retain_acc']}, FA={best['forget_acc']}, "
                    f"Q-INT4={best['quant_int4']} (certificate granted), "
                    f"demonstrating a materially better FA-RA-Q-INT4 operating point "
                    f"than α=3.'")

        if nocert_rows:
            best_ra = nocert_rows[0]
            logger.info(
                f"\n  Best RA without cert: "
                f"α={best_ra['alpha']}  "
                f"RA={best_ra['retain_acc']}  Q-INT4={best_ra['quant_int4']}"
            )

    salun_orig = next((r for r in all_rows if "orig" in r.get("method","")), None)
    if salun_orig:
        logger.info(f"\n  SalUn original HPs: "
                    f"FA={salun_orig.get('forget_acc')}  "
                    f"RA={salun_orig.get('retain_acc')}  "
                    f"Q-INT4={salun_orig.get('quant_int4')}  "
                    f"cert={salun_orig.get('cert')}")
        qi4 = salun_orig.get('quant_int4', 1.0)
        if qi4 > 0.05:
            logger.info(
                f"  -> SalUn still fails cert at original HPs (Q-INT4={qi4}>0.05)")
            logger.info(
                f"  -> Write: 'SalUn at original lr=1e-4, 500 steps achieves "
                f"Q-INT4={qi4}, confirming the baseline tuning does not explain "
                f"the INT4 vulnerability.'")
        else:
            logger.info(
                f"  -> SalUn PASSES cert at original HPs. Report honestly.")
            logger.info(
                f"  -> Update Table 1 with this result.")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
