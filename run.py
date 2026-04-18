"""
run.py  —  DurableUn Master Script
====================================
One script to run everything. Uses 'py' not 'python'.

USAGE:
  py run.py <command> [options]

COMMANDS:
  preflight              Check GPU, HF token, datasets load correctly
  baseline               Run Phase 0 baselines on any dataset
  saf                    Train DurableUn-SAF (any alpha)
  pareto                 Pareto sweep: alpha 0, 1, 3
  certificate            Compute durability certificate from checkpoint
  figures                Generate all paper figures
  multi_dataset          Run main paper experiment matrix
  seeds                  Multi-seed reliability eval (3 seeds)
  ste_baselines          STE-augmented SalUn/GA comparison
  full                   Run EVERYTHING in order (overnight)

EXAMPLES:

  # 1. Check setup first (always run this first):
  py run.py preflight

  # 2. Phase 0 baselines on TOFU (Table 1 of paper):
  py run.py baseline --datasets tofu

  # 3. Train DurableUn-SAF v3 (best FA, ~25 min):
  py run.py saf --alpha 1.0

  # 4. Pareto sweep for the paper (~3.5 hours):
  py run.py pareto

  # 5. Get durability certificate:
  py run.py certificate --checkpoint checkpoints/saf_alpha_3p0

  # 6. Multi-dataset experiment (TOFU + MUSE + WikiBio):
  py run.py multi_dataset --datasets tofu muse_news wpu

  # 7. Generate all figures:
  py run.py figures

  # 8. Run everything overnight:
  py run.py full
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

import torch
import yaml

# ── Root setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.utils.logging_utils import setup_root_logger, now_str, file_ts
from src.data.data_utils import set_seed

logger = logging.getLogger("run")

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG     = os.path.join(ROOT, "configs", "base_config.yaml")
DURABLEUN_CONFIG   = os.path.join(ROOT, "configs", "durableun_config.yaml")


def load_config(path=DEFAULT_CONFIG):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}. Did you run from durableun_v2/ ?")
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
        if p.device.type != "meta": return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════════════
# COMMAND: preflight
# ═════════════════════════════════════════════════════════════════════════════

def cmd_preflight(args):
    """Check everything is set up correctly before running experiments."""
    print("\n" + "="*55)
    print("  DurableUn Preflight Check")
    print("="*55)
    ok = True

    # GPU
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [OK]  GPU: {name} ({mem:.0f} GB VRAM)")
        if mem < 20:
            print(f"  [WARN] Less than 20 GB VRAM — some runs may OOM")
    else:
        print("  [FAIL] No CUDA GPU found")
        ok = False

    # HF token
    try:
        from hf_token import HF_TOKEN
        if HF_TOKEN and len(HF_TOKEN) > 10:
            print(f"  [OK]  HF token found: {HF_TOKEN[:8]}...")
        else:
            print("  [FAIL] HF_TOKEN in hf_token.py is empty")
            ok = False
    except ImportError:
        print("  [FAIL] hf_token.py not found — create it with HF_TOKEN = 'hf_...'")
        ok = False

    # Config files
    for cfg_path in [DEFAULT_CONFIG, DURABLEUN_CONFIG]:
        if os.path.exists(cfg_path):
            print(f"  [OK]  Config: {os.path.basename(cfg_path)}")
        else:
            print(f"  [FAIL] Config missing: {cfg_path}")
            ok = False

    # Python packages
    pkgs = ["torch", "transformers", "peft", "bitsandbytes", "datasets", "yaml", "tqdm"]
    for pkg in pkgs:
        try:
            __import__(pkg)
            print(f"  [OK]  {pkg}")
        except ImportError:
            print(f"  [FAIL] {pkg} not installed — run: pip install {pkg}")
            ok = False

    # Directories
    for d in ["checkpoints", "results", "logs", "figures"]:
        path = os.path.join(ROOT, d)
        os.makedirs(path, exist_ok=True)
        print(f"  [OK]  Directory: {d}/")

    # Quick dataset load test
    print("\n  Testing TOFU dataset load...")
    try:
        from datasets import load_dataset
        ds = load_dataset("locuslab/TOFU", "forget10", split="train")
        print(f"  [OK]  TOFU forget10: {len(ds)} samples")
    except Exception as e:
        print(f"  [WARN] TOFU load failed: {e}")

    print("\n" + "="*55)
    if ok:
        print("  ALL CHECKS PASSED — ready to run experiments")
    else:
        print("  SOME CHECKS FAILED — fix above issues first")
    print("="*55 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# COMMAND: baseline
# ═════════════════════════════════════════════════════════════════════════════

def cmd_baseline(args):
    """Run Phase 0 baselines. Reproduces Table 1 of the paper."""
    config   = load_config(DEFAULT_CONFIG)
    datasets = args.datasets or ["tofu"]
    methods  = args.methods  or ["ga", "npo", "scrub", "salun", "alpha_edit"]
    seed     = args.seed

    setup_root_logger(_get(config, "paths.logs", default="logs"))

    from src.models.model_utils import load_model_with_lora
    from src.data.dataset_registry import get_dataloaders
    from src.evaluation.evaluator import (
        compute_token_accuracy, compute_quantization_recovery, compute_mia_auc
    )

    res_dir = _get(config, "paths.results", default="results")
    os.makedirs(res_dir, exist_ok=True)
    csv_path = os.path.join(res_dir, f"baseline_{file_ts()}.csv")
    all_rows = []

    model_name = _get(config, "model.name",
                      default="meta-llama/Meta-Llama-3-8B-Instruct")
    n_steps = args.n_steps or _get(config, "training.n_steps", default=300)
    lr      = _get(config, "training.lr", default=5e-5)

    for dataset in datasets:
        for method in methods:
            run_id = f"{method}/{dataset}"
            logger.info(f"\n{'='*50}\n  {run_id}\n{'='*50}")

            # Check if already done
            ckpt_dir = _get(config, "paths.checkpoints", default="checkpoints")
            result_f = os.path.join(ckpt_dir, f"{method}_{dataset}_s{seed}", "result.json")
            if args.resume and os.path.exists(result_f):
                logger.info(f"  SKIP — result exists")
                with open(result_f) as f:
                    all_rows.append(json.load(f))
                continue

            set_seed(seed)
            model, tokenizer = load_model_with_lora(
                model_name,
                lora_config=config.get("lora"),
                dtype=_get(config, "model.dtype", default="bfloat16"),
                device_map=_get(config, "model.device_map", default="cuda:0"),
                load_in_4bit=_get(config, "model.load_in_4bit", default=True),
                cache_dir=_get(config, "paths.cache_dir"),
            )
            device = _real_device(model)

            fl, rl, _ = get_dataloaders(
                tokenizer, dataset=dataset,
                forget_split=_get(config, "dataset.forget_split", default="forget10"),
                retain_split=_get(config, "dataset.retain_split", default="retain90"),
                max_length=_get(config, "dataset.max_length", default=256),
                batch_size=_get(config, "dataset.batch_size", default=4),
                num_workers=0,
            )

            # Train
            t0 = time.time()
            _run_method(method, model, fl, rl, device, n_steps, lr, logger)
            wall_min = (time.time() - t0) / 60

            # Evaluate
            row = _eval_full(
                model, tokenizer, fl, rl, device, config,
                method, dataset, seed, wall_min
            )
            all_rows.append(row)
            _save_row(row, csv_path)
            _save_checkpoint(model, tokenizer, row, ckpt_dir, method, dataset, seed)

            del model
            torch.cuda.empty_cache()

    _print_table(all_rows, datasets)
    logger.info(f"\nResults: {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
# COMMAND: saf
# ═════════════════════════════════════════════════════════════════════════════

def cmd_saf(args):
    """Train DurableUn-SAF at a given alpha."""
    config  = load_config(DURABLEUN_CONFIG)
    alpha   = args.alpha
    lam     = max(1.0, alpha + 1.0)
    seed    = args.seed
    dataset = (args.datasets or ["tofu"])[0]
    n_steps = args.n_steps or 300

    setup_root_logger(_get(config, "paths.logs", default="logs"))
    logger.info(f"DurableUn-SAF | alpha={alpha} lambda={lam} seed={seed}")

    from src.models.model_utils import load_model_with_lora
    from src.data.dataset_registry import get_dataloaders
    from src.durableun.saf import SAF

    set_seed(seed)
    model, tokenizer = load_model_with_lora(
        _get(config, "model.name", default="meta-llama/Meta-Llama-3-8B-Instruct"),
        lora_config=config.get("lora"),
        dtype=_get(config, "model.dtype", default="bfloat16"),
        device_map=_get(config, "model.device_map", default="cuda:0"),
        load_in_4bit=_get(config, "model.load_in_4bit", default=True),
        cache_dir=_get(config, "paths.cache_dir"),
    )
    device = _real_device(model)

    fl, rl, _ = get_dataloaders(
        tokenizer, dataset=dataset,
        forget_split=_get(config, "dataset.forget_split", default="forget10"),
        retain_split=_get(config, "dataset.retain_split", default="retain90"),
        max_length=_get(config, "dataset.max_length", default=256),
        batch_size=_get(config, "dataset.batch_size", default=4),
        num_workers=0,
    )

    t0 = time.time()
    SAF(
        model=model, forget_loader=fl, retain_loader=rl,
        device=device, n_steps=n_steps,
        lr=_get(config, "training.lr", default=5e-5),
        retain_lambda=lam,
        gradient_clip=_get(config, "training.gradient_clip", default=1.0),
        log_every=_get(config, "training.log_every", default=50),
        alpha_quant=alpha, warmup_steps=100,
    ).unlearn()
    wall_min = (time.time() - t0) / 60

    res  = _get(config, "paths.results",     default="results")
    ckpt = _get(config, "paths.checkpoints", default="checkpoints")
    os.makedirs(res, exist_ok=True)

    row = _eval_full(model, tokenizer, fl, rl, device, config,
                     f"saf_alpha{alpha}", dataset, seed, wall_min)
    _save_row(row, os.path.join(res, f"saf_alpha{alpha}_{file_ts()}.csv"))
    _save_checkpoint(model, tokenizer, row, ckpt,
                     f"saf_alpha{str(alpha).replace('.','p')}", dataset, seed)

    logger.info(f"\nDone | FA={row['forget_acc']}  RA={row['retain_acc']}  "
                f"Q-INT4={row['quant_int4']}  cert={row['cert']}")
    del model; torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════
# COMMAND: pareto
# ═════════════════════════════════════════════════════════════════════════════

def cmd_pareto(args):
    """Pareto sweep: alpha in {0.0, 1.0, 3.0}. Reproduces Table 2."""
    alphas = args.alphas or [0.0, 1.0, 3.0]
    for alpha in sorted(alphas):
        args2      = argparse.Namespace(**vars(args))
        args2.alpha = alpha
        args2.datasets = args.datasets or ["tofu"]
        cmd_saf(args2)


# ═════════════════════════════════════════════════════════════════════════════
# COMMAND: certificate
# ═════════════════════════════════════════════════════════════════════════════

def cmd_certificate(args):
    """Compute the empirical durability certificate from a checkpoint."""
    config    = load_config(DURABLEUN_CONFIG)
    ckpt_dir  = os.path.abspath(args.checkpoint)
    ckpt_model = os.path.join(ckpt_dir, "model")
    epsilon   = args.epsilon
    model_name = _get(config, "model.name",
                      default="meta-llama/Meta-Llama-3-8B-Instruct")

    print(f"\nCheckpoint: {ckpt_dir}")
    print(f"Target epsilon: {epsilon}")

    if not os.path.exists(ckpt_model):
        print(f"ERROR: {ckpt_model} not found.")
        print("Available checkpoints:")
        ckpt_base = _get(config, "paths.checkpoints", default="checkpoints")
        for d in sorted(os.listdir(ckpt_base)):
            if os.path.isdir(os.path.join(ckpt_base, d)):
                print(f"  checkpoints/{d}")
        return

    # Create adapter_config.json if missing
    _ensure_adapter_config(ckpt_model, model_name, config.get("lora", {}))

    from src.models.model_utils import load_tokenizer
    from src.data.tofu_dataset import get_tofu_dataloaders
    from src.theory.certificate import compute_certificate
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    tok  = load_tokenizer(model_name, _get(config, "paths.cache_dir"))
    bnb  = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                               bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cuda:0", quantization_config=bnb,
        cache_dir=_get(config, "paths.cache_dir"), trust_remote_code=True)
    base.config.use_cache = False
    base  = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, ckpt_model, is_trainable=False)
    model.eval()

    fl, _, _ = get_tofu_dataloaders(
        tok,
        forget_split=_get(config, "dataset.forget_split", default="forget10"),
        retain_split=_get(config, "dataset.retain_split", default="retain90"),
        batch_size=4, max_length=256, num_workers=0,
    )

    cert = compute_certificate(
        model=model, forget_loader=fl,
        method_name=os.path.basename(ckpt_dir),
        epsilon_target=epsilon,
        save_path=os.path.join(ckpt_dir, "certificate.json"),
    )

    print(f"\nLaTeX table row:")
    p = cert.fa_per_precision
    granted = r'\checkmark' if cert.is_durable else r'\times'
    print(f"  & {p.get('bf16','?')} & {p.get('int8','?')} & "
          f"\\textbf{{{p.get('int4','?')}}} & {granted} \\\\")


# ═════════════════════════════════════════════════════════════════════════════
# COMMAND: figures
# ═════════════════════════════════════════════════════════════════════════════

def cmd_figures(args):
    """Generate all paper figures from experimental results."""
    import importlib.util
    fig_script = os.path.join(ROOT, "experiments", "generate_figures.py")
    spec = importlib.util.spec_from_file_location("gen_figs", fig_script)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("Figures saved to figures/")


# ═════════════════════════════════════════════════════════════════════════════
# COMMAND: multi_dataset
# ═════════════════════════════════════════════════════════════════════════════

def cmd_multi_dataset(args):
    """
    Run the main paper experiment matrix.
    Methods: ga, salun, graddiff, durableun_saf_v3, durableun_saf_alpha3
    Datasets: tofu, muse_news, wpu
    """
    datasets = args.datasets or ["tofu", "muse_news", "wpu"]
    methods  = args.methods  or ["ga", "salun", "graddiff",
                                  "durableun_saf_v3", "durableun_saf_alpha3"]
    seed     = args.seed
    n_steps  = args.n_steps or 300

    # Route to priority_audit.py logic
    config   = load_config(DEFAULT_CONFIG)
    setup_root_logger(_get(config, "paths.logs", default="logs"))

    from src.models.model_utils import load_model_with_lora
    from src.data.dataset_registry import get_dataloaders

    res_dir  = _get(config, "paths.results",     default="results")
    ckpt_dir = _get(config, "paths.checkpoints", default="checkpoints")
    os.makedirs(res_dir, exist_ok=True)
    csv_path = os.path.join(res_dir, f"multi_dataset_{file_ts()}.csv")
    all_rows = []

    model_name = _get(config, "model.name",
                      default="meta-llama/Meta-Llama-3-8B-Instruct")
    lr   = _get(config, "training.lr",            default=5e-5)
    clip = _get(config, "training.gradient_clip", default=1.0)
    loge = _get(config, "training.log_every",     default=50)

    for dataset in datasets:
        for method in methods:
            run_id = f"{method}/{dataset}/s{seed}"
            result_f = os.path.join(ckpt_dir, f"{method}_{dataset}_s{seed}", "result.json")

            if args.resume and os.path.exists(result_f):
                logger.info(f"  SKIP {run_id}")
                with open(result_f) as f:
                    all_rows.append(json.load(f))
                continue

            logger.info(f"\n{'='*55}\n  {run_id}\n{'='*55}")

            # Use durableun config for durableun methods
            cfg = load_config(DURABLEUN_CONFIG if "durableun" in method else DEFAULT_CONFIG)

            try:
                set_seed(seed)
                model, tokenizer = load_model_with_lora(
                    model_name,
                    lora_config=cfg.get("lora"),
                    dtype=_get(cfg, "model.dtype", default="bfloat16"),
                    device_map=_get(cfg, "model.device_map", default="cuda:0"),
                    load_in_4bit=_get(cfg, "model.load_in_4bit", default=True),
                    cache_dir=_get(cfg, "paths.cache_dir"),
                )
                device = _real_device(model)

                fl, rl, _ = get_dataloaders(
                    tokenizer, dataset=dataset,
                    forget_split=_get(cfg, "dataset.forget_split", default="forget10"),
                    retain_split=_get(cfg, "dataset.retain_split", default="retain90"),
                    max_length=_get(cfg, "dataset.max_length", default=256),
                    batch_size=_get(cfg, "dataset.batch_size", default=4),
                    num_workers=0,
                )

                t0 = time.time()
                _run_method(method, model, fl, rl, device, n_steps, lr, logger)
                wall_min = (time.time() - t0) / 60

                row = _eval_full(model, tokenizer, fl, rl, device, cfg,
                                 method, dataset, seed, wall_min)
                all_rows.append(row)
                _save_row(row, csv_path)
                _save_checkpoint(model, tokenizer, row, ckpt_dir, method, dataset, seed)

            except Exception as e:
                logger.error(f"FAILED {run_id}: {e}", exc_info=True)
                torch.cuda.empty_cache()
                continue

            del model
            torch.cuda.empty_cache()

    _print_table(all_rows, datasets)
    logger.info(f"\nCSV: {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
# COMMAND: seeds  (multi-seed reliability)
# ═════════════════════════════════════════════════════════════════════════════

def cmd_seeds(args):
    """Run GA, SalUn, DurableUn-SAF across 3 seeds for mean±std."""
    import statistics
    methods = args.methods or ["ga", "salun", "durableun_saf_v3", "durableun_saf_alpha3"]
    seeds   = args.seeds   or [42, 123, 5508]
    dataset = (args.datasets or ["tofu"])[0]

    config   = load_config(DEFAULT_CONFIG)
    setup_root_logger(_get(config, "paths.logs", default="logs"))

    from src.models.model_utils import load_model_with_lora
    from src.data.dataset_registry import get_dataloaders

    res_dir = _get(config, "paths.results", default="results")
    os.makedirs(res_dir, exist_ok=True)
    csv_path = os.path.join(res_dir, f"multi_seed_{file_ts()}.csv")

    model_name = _get(config, "model.name",
                      default="meta-llama/Meta-Llama-3-8B-Instruct")
    n_steps = args.n_steps or 300
    lr      = _get(config, "training.lr", default=5e-5)

    per_method = {}

    for method in methods:
        rows = []
        for seed in seeds:
            cfg      = load_config(DURABLEUN_CONFIG if "durableun" in method else DEFAULT_CONFIG)
            run_id   = f"{method}/s{seed}"
            ckpt_dir = _get(cfg, "paths.checkpoints", default="checkpoints")
            result_f = os.path.join(ckpt_dir, f"{method}_{dataset}_s{seed}", "result.json")

            if args.resume and os.path.exists(result_f):
                logger.info(f"  SKIP {run_id}")
                with open(result_f) as f:
                    rows.append(json.load(f))
                continue

            logger.info(f"\n  [{run_id}]")
            set_seed(seed)

            model, tokenizer = load_model_with_lora(
                model_name,
                lora_config=cfg.get("lora"),
                dtype=_get(cfg, "model.dtype", default="bfloat16"),
                device_map="cuda:0",
                load_in_4bit=_get(cfg, "model.load_in_4bit", default=True),
            )
            device = _real_device(model)

            fl, rl, _ = get_dataloaders(
                tokenizer, dataset=dataset,
                forget_split="forget10", retain_split="retain90",
                max_length=256, batch_size=4, num_workers=0,
            )

            t0 = time.time()
            _run_method(method, model, fl, rl, device, n_steps, lr, logger)
            wall_min = (time.time() - t0) / 60

            row = _eval_full(model, tokenizer, fl, rl, device, cfg,
                             method, dataset, seed, wall_min)
            rows.append(row)
            _save_row(row, csv_path)
            _save_checkpoint(model, tokenizer, row, ckpt_dir, method, dataset, seed)

            del model; torch.cuda.empty_cache()

        per_method[method] = rows

        # Print mean±std
        if len(rows) >= 2:
            logger.info(f"\n  {method} | mean ± std over {seeds}")
            for metric in ["forget_acc", "retain_acc", "quant_int4", "ra_int4"]:
                vals = [r[metric] for r in rows if r.get(metric, -1) >= 0]
                if len(vals) >= 2:
                    logger.info(
                        f"    {metric:<14}: "
                        f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"
                    )

    logger.info(f"\nCSV: {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
# COMMAND: ste_baselines
# ═════════════════════════════════════════════════════════════════════════════

def cmd_ste_baselines(args):
    """Run STE-augmented SalUn and GA baselines."""
    config  = load_config(DEFAULT_CONFIG)
    setup_root_logger(_get(config, "paths.logs", default="logs"))

    from src.models.model_utils import load_model_with_lora
    from src.data.dataset_registry import get_dataloaders
    from src.baselines.ste_augmented_baselines import GA_STE, SalUn_STE

    dataset = (args.datasets or ["tofu"])[0]
    seed    = args.seed
    n_steps = args.n_steps or 300
    alpha   = args.alpha if hasattr(args, "alpha") else 1.0

    res_dir  = _get(config, "paths.results",     default="results")
    ckpt_dir = _get(config, "paths.checkpoints", default="checkpoints")
    os.makedirs(res_dir, exist_ok=True)
    csv_path = os.path.join(res_dir, f"ste_baselines_{file_ts()}.csv")

    model_name = _get(config, "model.name",
                      default="meta-llama/Meta-Llama-3-8B-Instruct")

    for method_cls, method_name in [(GA_STE, "ga_ste"), (SalUn_STE, "salun_ste")]:
        set_seed(seed)
        model, tokenizer = load_model_with_lora(
            model_name, lora_config=config.get("lora"),
            dtype="bfloat16", device_map="cuda:0", load_in_4bit=True,
        )
        device = _real_device(model)
        fl, rl, _ = get_dataloaders(tokenizer, dataset=dataset,
                                     max_length=256, batch_size=4, num_workers=0)

        retain_lambda = max(2.0, alpha + 1.0)
        obj = method_cls(model=model, forget_loader=fl, retain_loader=rl,
                         device=device, n_steps=n_steps,
                         lr=_get(config, "training.lr", default=5e-5),
                         retain_lambda=retain_lambda, alpha_ste=alpha)
        t0 = time.time()
        obj.run()
        wall_min = (time.time() - t0) / 60

        row = _eval_full(model, tokenizer, fl, rl, device, config,
                         method_name, dataset, seed, wall_min)
        _save_row(row, csv_path)
        _save_checkpoint(model, tokenizer, row, ckpt_dir, method_name, dataset, seed)
        logger.info(f"  {method_name}: FA={row['forget_acc']}  Q-INT4={row['quant_int4']}")
        del model; torch.cuda.empty_cache()

    logger.info(f"\nCSV: {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
# COMMAND: full  (run everything in priority order)
# ═════════════════════════════════════════════════════════════════════════════

def cmd_full(args):
    """
    Run everything in the recommended order. Use for overnight runs.
    Skips completed steps automatically (--resume behavior).
    """
    args.resume = True

    print("\n" + "="*55)
    print("  DurableUn FULL RUN")
    print("  This will take ~12-15 hours total.")
    print("  Results save after each method — safe to interrupt.")
    print("="*55 + "\n")

    steps = [
        # (description, datasets, methods, use_durableun_config)
        ("Step 1: Training-free baselines on TOFU",
            ["tofu"], ["tv", "dare"], False),
        ("Step 2: Phase 0 baselines on TOFU",
            ["tofu"], ["ga", "salun", "graddiff"], False),
        ("Step 3: DurableUn v3 on TOFU",
            ["tofu"], ["durableun_saf_v3"], True),
        ("Step 4: Phase 0 baselines on MUSE-News",
            ["muse_news"], ["ga", "salun", "graddiff"], False),
        ("Step 5: DurableUn v3 on MUSE-News",
            ["muse_news"], ["durableun_saf_v3"], True),
        ("Step 6: Phase 0 baselines on WikiBio",
            ["wpu"], ["ga", "salun", "graddiff"], False),
        ("Step 7: DurableUn v3 on WikiBio",
            ["wpu"], ["durableun_saf_v3"], True),
        ("Step 8: Appendix baselines on TOFU",
            ["tofu"], ["wga"], False),
        ("Step 9: DurableUn alpha=3 on TOFU (certificate, long)",
            ["tofu"], ["durableun_saf_alpha3"], True),
    ]

    for desc, datasets, methods, _ in steps:
        print(f"\n{'─'*55}")
        print(f"  {desc}")
        print(f"{'─'*55}")
        sub_args           = argparse.Namespace(**vars(args))
        sub_args.datasets  = datasets
        sub_args.methods   = methods
        sub_args.resume    = True
        sub_args.skip_ft   = True
        sub_args.n_steps   = args.n_steps or 300
        sub_args.seed      = args.seed
        sub_args.alphas    = None
        cmd_multi_dataset(sub_args)

    # Generate figures at the end
    print("\n  Generating figures...")
    cmd_figures(args)

    print("\n" + "="*55)
    print("  FULL RUN COMPLETE")
    print("="*55)


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

def _run_method(method, model, fl, rl, device, n_steps, lr, logger):
    """Dispatch training to the right method."""
    from src.baselines.base import _clm_loss

    def _inf(loader):
        while True:
            for b in loader: yield b

    if method == "ga":
        try:
            from src.baselines.ga import GA
            GA(model=model, forget_loader=fl, retain_loader=rl, device=device,
               n_steps=n_steps, lr=lr, retain_lambda=1.0).unlearn()
        except ImportError:
            # Inline GA
            from torch.optim import AdamW
            from torch.optim.lr_scheduler import CosineAnnealingLR
            from tqdm import tqdm
            opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
            sch = CosineAnnealingLR(opt, T_max=n_steps)
            fi  = _inf(fl); ri = _inf(rl)
            for _ in tqdm(range(n_steps), desc="GA", file=sys.stdout):
                opt.zero_grad()
                dev = device if isinstance(device, str) else str(device)
                fb  = {k: v.to(dev) if hasattr(v,"to") else v for k,v in next(fi).items()}
                rb  = {k: v.to(dev) if hasattr(v,"to") else v for k,v in next(ri).items()}
                (-_clm_loss(model,fb) + _clm_loss(model,rb)).backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sch.step()

    elif method in ["npo", "scrub", "salun", "alpha_edit", "rmu"]:
        from src.baselines.baseline_registry import get_baseline
        get_baseline(method, model=model, forget_loader=fl, retain_loader=rl,
                     device=device, n_steps=n_steps, lr=lr).unlearn()

    elif method == "graddiff":
        from src.baselines.gradient_difference import GradDiff
        GradDiff(model=model, forget_loader=fl, retain_loader=rl,
                 device=device, n_steps=n_steps, lr=lr, retain_lambda=1.0).unlearn()

    elif method == "wga":
        from src.baselines.wga import WGA
        WGA(model=model, forget_loader=fl, retain_loader=rl,
            device=device, n_steps=n_steps, lr=lr,
            retain_lambda=1.0, variant="weighted").unlearn()

    elif method == "tv":
        from src.baselines.tv_distance import TaskVectorUnlearning
        TaskVectorUnlearning(model=model, forget_loader=fl, retain_loader=rl,
                             device=device, scale=1.0, method="negate").unlearn()

    elif method == "dare":
        from src.baselines.tv_distance import TaskVectorUnlearning
        TaskVectorUnlearning(model=model, forget_loader=fl, retain_loader=rl,
                             device=device, scale=1.0, method="dare").unlearn()

    elif method == "durableun_saf_v3":
        from src.durableun.saf import SAF
        SAF(model=model, forget_loader=fl, retain_loader=rl,
            device=device, n_steps=n_steps, lr=lr,
            retain_lambda=2.0, alpha_quant=1.0, warmup_steps=100).unlearn()

    elif method == "durableun_saf_alpha3":
        from src.durableun.saf import SAF
        SAF(model=model, forget_loader=fl, retain_loader=rl,
            device=device, n_steps=n_steps, lr=lr,
            retain_lambda=4.0, alpha_quant=3.0, warmup_steps=100).unlearn()

    else:
        raise ValueError(f"Unknown method: {method}. "
                         f"Choose from: ga, npo, scrub, salun, alpha_edit, rmu, "
                         f"graddiff, wga, tv, dare, durableun_saf_v3, durableun_saf_alpha3")


def _eval_full(model, tokenizer, fl, rl, device, config, method, dataset, seed, wall_min):
    """Run full evaluation and return a results dict."""
    from src.evaluation.evaluator import (
        compute_token_accuracy, compute_quantization_recovery, compute_mia_auc
    )

    dev   = str(device)
    max_b = _get(config, "eval.max_batches", default=30)

    fa    = compute_token_accuracy(model, fl, dev, max_b)
    ra    = compute_token_accuracy(model, rl, dev, max_b)
    mia   = compute_mia_auc(model, fl, rl, dev)
    quant = compute_quantization_recovery(model, fl, dev, ["bf16","int8","int4"], max_b)

    try:
        from src.evaluation.evaluator_additions import compute_token_accuracy_quantized
        ra_int4 = compute_token_accuracy_quantized(model, rl, dev, "int4", max_b)
    except Exception:
        ra_int4 = -1.0

    DISPLAY = {
        "ga":"GA","npo":"NPO","scrub":"SCRUB","salun":"SalUn","rmu":"RMU",
        "alpha_edit":"AlphaEdit","graddiff":"GradDiff","wga":"WGA",
        "tv":"Task Vector","dare":"DARE","noisy_ga":"NoisyGA",
        "durableun_saf_v3":"DurableUn-SAF v3",
        "durableun_saf_alpha3":"DurableUn-SAF α=3",
    }

    row = {
        "method":          method,
        "method_display":  DISPLAY.get(method, method),
        "dataset":         dataset,
        "seed":            seed,
        "forget_acc":      round(fa,    4),
        "retain_acc":      round(ra,    4),
        "mia_auc":         round(mia,   4),
        "quant_bf16":      round(quant.get("bf16",-1), 4),
        "quant_int8":      round(quant.get("int8",-1), 4),
        "quant_int4":      round(quant.get("int4",-1), 4),
        "ra_int4":         round(ra_int4, 4),
        "ft_50":           -1.0,
        "wall_min":        round(wall_min, 1),
        "cert":            "Y" if quant.get("int4",1.0) <= 0.05 else "N",
        "evaluated_at":    now_str(),
    }
    logger.info(
        f"  FA={fa:.4f}  RA={ra:.4f}  "
        f"Q-INT8={quant.get('int8',-1):.4f}  Q-INT4={quant.get('int4',-1):.4f}  "
        f"RA-INT4={ra_int4:.4f}  cert={row['cert']}"
    )
    return row


def _save_row(row, csv_path):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    write_hdr = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
        if write_hdr: w.writeheader()
        w.writerow(row)


def _save_checkpoint(model, tokenizer, row, ckpt_dir, method, dataset, seed):
    path = os.path.join(ckpt_dir, f"{method}_{dataset}_s{seed}")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(os.path.join(path, "model"))
    tokenizer.save_pretrained(os.path.join(path, "model"))
    with open(os.path.join(path, "result.json"), "w") as f:
        json.dump(row, f, indent=2)
    logger.info(f"  Saved checkpoint: {path}")


def _print_table(rows, datasets):
    for ds in datasets:
        ds_rows = [r for r in rows if r.get("dataset") == ds]
        if not ds_rows: continue
        logger.info(f"\n{'='*75}")
        logger.info(f"  {ds.upper()}")
        logger.info(f"  {'Method':<24} {'FA↓':>6} {'RA↑':>6} {'Q-INT8':>7} {'Q-INT4':>7} {'RA-INT4':>8} {'Cert':>5}")
        logger.info("  " + "-"*65)
        for r in ds_rows:
            logger.info(
                f"  {r.get('method_display','?'):<24} "
                f"{r.get('forget_acc',-1):>6.4f} "
                f"{r.get('retain_acc',-1):>6.4f} "
                f"{r.get('quant_int8',-1):>7.4f} "
                f"{r.get('quant_int4',-1):>7.4f} "
                f"{r.get('ra_int4',-1):>8.4f} "
                f"{r.get('cert','?'):>5}"
            )


def _ensure_adapter_config(ckpt_model_dir, model_name, lora_cfg):
    adapter_cfg = os.path.join(ckpt_model_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg): return
    cfg = {
        "peft_type":"LORA","task_type":"CAUSAL_LM",
        "r": lora_cfg.get("r",16),
        "lora_alpha": lora_cfg.get("lora_alpha",32),
        "lora_dropout": lora_cfg.get("lora_dropout",0.05),
        "bias": lora_cfg.get("bias","none"),
        "target_modules": lora_cfg.get("target_modules",["q_proj","v_proj","k_proj","o_proj"]),
        "fan_in_fan_out": False, "inference_mode": True,
        "base_model_name_or_path": model_name,
    }
    with open(adapter_cfg, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Created adapter_config.json")


# ═════════════════════════════════════════════════════════════════════════════
# CLI parser
# ═════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        prog="run.py",
        description="DurableUn master script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # Shared arguments
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--seed",     type=int, default=42)
    shared.add_argument("--n_steps",  type=int, default=None)
    shared.add_argument("--datasets", nargs="+", default=None)
    shared.add_argument("--methods",  nargs="+", default=None)
    shared.add_argument("--resume",   action="store_true")
    shared.add_argument("--skip_ft",  action="store_true")

    sub.add_parser("preflight",    parents=[shared], help="Check setup")
    sub.add_parser("baseline",     parents=[shared], help="Phase 0 baselines")

    saf_p = sub.add_parser("saf", parents=[shared], help="Train DurableUn-SAF")
    saf_p.add_argument("--alpha", type=float, default=1.0)

    par_p = sub.add_parser("pareto", parents=[shared], help="Pareto sweep alpha={0,1,3}")
    par_p.add_argument("--alphas", nargs="+", type=float, default=None)

    cert_p = sub.add_parser("certificate", parents=[shared], help="Compute certificate")
    cert_p.add_argument("--checkpoint", required=True)
    cert_p.add_argument("--epsilon",    type=float, default=0.05)

    sub.add_parser("figures",      parents=[shared], help="Generate figures")
    sub.add_parser("multi_dataset",parents=[shared], help="Main paper matrix")

    seed_p = sub.add_parser("seeds", parents=[shared], help="Multi-seed eval")
    seed_p.add_argument("--seeds", nargs="+", type=int, default=None)

    sub.add_parser("ste_baselines",parents=[shared], help="STE-augmented baselines")
    sub.add_parser("full",         parents=[shared], help="Run everything overnight")

    return p


def main():
    setup_root_logger("logs")
    p    = build_parser()
    args = p.parse_args()

    dispatch = {
        "preflight":    cmd_preflight,
        "baseline":     cmd_baseline,
        "saf":          cmd_saf,
        "pareto":       cmd_pareto,
        "certificate":  cmd_certificate,
        "figures":      cmd_figures,
        "multi_dataset":cmd_multi_dataset,
        "seeds":        cmd_seeds,
        "ste_baselines":cmd_ste_baselines,
        "full":         cmd_full,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
