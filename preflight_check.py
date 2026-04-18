"""
preflight_check.py
==================
Runs a comprehensive check of EVERY component before full training.
Takes ~15 minutes (including one training step per method).

Usage:
  python preflight_check.py --config configs/base_config.yaml

  # Fast mode - skip model load / training steps (~2 min):
  python preflight_check.py --config configs/base_config.yaml --skip_model_load
"""

import argparse
import importlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from typing import Callable, List, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import yaml


def _ts():
    return datetime.now().strftime("%H:%M:%S")


class CheckResult:
    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []

    def record(self, name: str, passed: bool, message: str = ""):
        icon = "✅ PASS" if passed else "❌ FAIL"
        print(f"  [{_ts()}] {icon}  {name}")
        if message:
            prefix = "         " + ("" if passed else "⚠ FIX: ")
            print(f"{prefix}{message}")
        self.results.append((name, passed, message))

    def summary(self):
        passed = sum(1 for _, p, _ in self.results if p)
        total  = len(self.results)
        print(f"\n{'='*60}")
        print(f"PRE-FLIGHT SUMMARY: {passed}/{total} checks passed")
        print(f"{'='*60}")
        for name, p, _ in self.results:
            print(f"  {'✅' if p else '❌'}  {name}")
        print()
        if passed == total:
            print("🟢 ALL CHECKS PASSED — Safe to run full training.")
            print()
            print("Next step:")
            print("  python cleanup_checkpoints.py --methods ga   # delete old GA checkpoint")
            print("  python experiments/phase0_baseline_audit.py --config configs/base_config.yaml")
        else:
            failed = [n for n, p, _ in self.results if not p]
            print(f"🔴 {len(failed)} check(s) failed: {failed}")
            print("Fix the issues above before running full training.")
        print()
        return passed == total


cr = CheckResult()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get(cfg, *keys, default=None):
    for k in keys:
        v = cfg
        try:
            for part in k.split("."): v = v[part]
            return v
        except (KeyError, TypeError):
            pass
    return default


# ─────────────────────────────────────────────────────────────────────────────
# Check 1: Python + packages
# ─────────────────────────────────────────────────────────────────────────────

def check_packages():
    print(f"\n[1/10] Python environment")
    py = sys.version_info
    cr.record("Python >= 3.9", py >= (3, 9),
              f"Current: {py.major}.{py.minor}." if py < (3,9) else "")

    for pkg in ["torch", "transformers", "peft", "bitsandbytes",
                "datasets", "sklearn", "tqdm", "yaml"]:
        mod = "sklearn" if pkg == "sklearn" else pkg
        try:
            m   = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            cr.record(f"Package: {pkg}", True, f"v{ver}")
        except ImportError:
            cr.record(f"Package: {pkg}", False, f"pip install {pkg}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 2: CUDA
# ─────────────────────────────────────────────────────────────────────────────

def check_cuda():
    print(f"\n[2/10] CUDA and GPU")
    import torch
    cr.record("CUDA available", torch.cuda.is_available(),
              "No GPU found. Training will be extremely slow on CPU.")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        name = torch.cuda.get_device_name(0)
        ok   = vram >= 20
        cr.record(f"GPU VRAM >= 20GB", ok,
                  f"{name}: {vram:.1f}GB. Need ≥20GB for 4-bit LLaMA-3-8B."
                  if not ok else f"{name}: {vram:.1f}GB ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Check 3: HF token
# ─────────────────────────────────────────────────────────────────────────────

def check_token_file():
    print(f"\n[3/10] HuggingFace token")
    token_file = os.path.join(ROOT, "hf_token.py")
    cr.record("hf_token.py exists", os.path.exists(token_file),
              "Create hf_token.py with: HF_TOKEN = 'hf_your_token'")
    if not os.path.exists(token_file):
        return None
    try:
        ns = {}
        with open(token_file, encoding="utf-8") as f:
            exec(f.read(), ns)
        token = ns.get("HF_TOKEN", "")
        valid = bool(token) and "PASTE" not in token.upper() and token.startswith("hf_")
        cr.record("HF_TOKEN valid", valid,
                  "Edit hf_token.py. Token must start with 'hf_'." if not valid else "")
        return token if valid else None
    except Exception as e:
        cr.record("hf_token.py readable", False, str(e))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Check 4: HF login
# ─────────────────────────────────────────────────────────────────────────────

def check_hf_login(token):
    print(f"\n[4/10] HuggingFace login")
    if not token:
        cr.record("HF login", False, "No valid token. Fix Check 3 first.")
        return
    try:
        os.environ["HF_TOKEN"]               = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        from huggingface_hub import login, whoami
        login(token=token, add_to_git_credential=False)
        info = whoami()
        cr.record("HF login succeeds", True, f"Logged in as: {info.get('name','?')}")
    except Exception as e:
        cr.record("HF login succeeds", False,
                  f"Login failed: {e}. Check huggingface.co/settings/tokens")


# ─────────────────────────────────────────────────────────────────────────────
# Check 5: TOFU dataset
# ─────────────────────────────────────────────────────────────────────────────

def check_tofu_dataset(config):
    print(f"\n[5/10] TOFU dataset")
    forget_split = _get(config, "dataset.forget_split", default="forget10")
    retain_split = _get(config, "dataset.retain_split", default="retain90")
    cache_dir    = _get(config, "paths.cache_dir",      default=None)
    try:
        from src.data.tofu_dataset import _load_tofu_split
        n = len(_load_tofu_split(forget_split, cache_dir))
        cr.record(f"TOFU {forget_split}: {n} samples", n >= 100,
                  f"Got {n}. Expected 400. Token problem if <10." if n < 100 else "")
    except Exception as e:
        cr.record(f"TOFU {forget_split}", False, str(e))
    try:
        from src.data.tofu_dataset import _load_tofu_split
        n = len(_load_tofu_split(retain_split, cache_dir))
        cr.record(f"TOFU {retain_split}: {n} samples", n >= 1000,
                  f"Got {n}. Expected 3600." if n < 1000 else "")
    except Exception as e:
        cr.record(f"TOFU {retain_split}", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Check 6: Model loads on GPU
# ─────────────────────────────────────────────────────────────────────────────

def check_model_load(config):
    print(f"\n[6/10] Model loading onto GPU")
    try:
        from src.models.model_utils import load_model_with_lora, _get_device
        t0 = time.time()
        model, tok = load_model_with_lora(
            _get(config, "model.name",        default="meta-llama/Meta-Llama-3-8B-Instruct"),
            lora_config=config.get("lora"),
            dtype=_get(config, "model.dtype",       default="bfloat16"),
            device_map=_get(config, "model.device_map",  default="cuda:0"),
            load_in_4bit=_get(config, "model.load_in_4bit", default=True),
            cache_dir=_get(config, "paths.cache_dir", default=None),
        )
        t     = time.time() - t0
        dev   = _get_device(model)
        on_gpu = dev.type == "cuda"
        cr.record(f"Model on GPU in {t:.0f}s", on_gpu,
                  f"Device={dev}. Set device_map: cuda:0 in config." if not on_gpu else "")
        return model, tok
    except Exception as e:
        cr.record("Model loads on GPU", False,
                  f"{e}\nIf OOM: ensure load_in_4bit: true in base_config.yaml")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Check 7: Forward pass (no OOM)
# ─────────────────────────────────────────────────────────────────────────────

def check_forward_pass(model, tokenizer, config):
    print(f"\n[7/10] Forward pass (OOM check)")
    if model is None:
        cr.record("Forward pass", False, "Model not loaded. Fix Check 6.")
        return

    import torch
    max_length = _get(config, "dataset.max_length", default=256)
    device     = next(p for p in model.parameters() if p.device.type != "meta").device

    try:
        enc  = tokenizer(
            "Question: What is the capital of France?\nAnswer: Paris",
            return_tensors="pt", max_length=max_length,
            truncation=True, padding="max_length",
        )
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        lbl  = ids.clone()

        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, labels=lbl)

        cr.record("Forward pass (no OOM)", True, f"Loss: {out.loss.item():.4f}")

        # NOTE: We do NOT test .backward() here because 4-bit quantized models
        # loaded with bitsandbytes require gradient_checkpointing_enable() before
        # calling backward. This is done inside BaseUnlearner automatically.
        # Check 8 (training steps) verifies backward works correctly.
        cr.record(
            "Backward pass handled by training step check",
            True,
            "Backward is verified in Check 8 (training steps). "
            "4-bit models need prepare_model_for_kbit_training() first."
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            cr.record("Forward pass (no OOM)", False,
                      "GPU OOM! Set load_in_4bit: true and max_length: 256 in config.")
        else:
            cr.record("Forward pass (no OOM)", False, str(e))
    except Exception as e:
        cr.record("Forward pass (no OOM)", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Check 8: One training step for each method
# ─────────────────────────────────────────────────────────────────────────────

def check_one_training_step(config):
    print(f"\n[8/10] One training step per baseline method")
    import torch
    from src.data.tofu_dataset import get_tofu_dataloaders
    from src.models.model_utils import load_model_with_lora, load_tokenizer, _get_device
    from src.baselines import get_baseline

    model_name   = _get(config, "model.name",         default="meta-llama/Meta-Llama-3-8B-Instruct")
    dtype        = _get(config, "model.dtype",        default="bfloat16")
    device_map   = _get(config, "model.device_map",   default="cuda:0")
    load_in_4bit = _get(config, "model.load_in_4bit", default=True)
    lora_cfg     = config.get("lora")
    cache_dir    = _get(config, "paths.cache_dir",    default=None)
    max_length   = _get(config, "dataset.max_length", default=256)

    try:
        tok     = load_tokenizer(model_name, cache_dir)
        fl, rl, _ = get_tofu_dataloaders(
            tok, forget_split="forget10", retain_split="retain90",
            batch_size=1, max_length=max_length, num_workers=0,
        )
    except Exception as e:
        cr.record("Load data for step test", False, str(e))
        return

    # Expected timings per step (for reference):
    # GA: ~4s, NPO: ~6s, SCRUB: ~5s (capped KL), SalUn: ~3s (mask computed once)
    # RMU: ~5s, AlphaEdit: ~30s (SVD computed once)
    methods = ["ga", "npo", "scrub", "salun", "rmu", "alpha_edit"]

    for method_name in methods:
        try:
            model, _ = load_model_with_lora(
                model_name, lora_config=lora_cfg, dtype=dtype,
                device_map=device_map, load_in_4bit=load_in_4bit,
                cache_dir=cache_dir,
            )
            device = _get_device(model)

            unlearner = get_baseline(
                method_name, model=model,
                forget_loader=fl, retain_loader=rl,
                device=device, n_steps=1, log_every=999,
            )
            t0 = time.time()
            unlearner.unlearn(fl, rl)
            elapsed = time.time() - t0

            # Estimate full-run time
            est_min = elapsed * _get(config, "training.n_steps", default=300) / 60
            cr.record(
                f"1 step: {method_name}",
                True,
                f"{elapsed:.1f}s/step → ~{est_min:.0f} min for 300 steps"
            )
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            cr.record(f"1 step: {method_name}", False, str(e))
            try: del model; torch.cuda.empty_cache()
            except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# Check 9: Quantization simulation
# ─────────────────────────────────────────────────────────────────────────────

def check_quantization(model, tokenizer, config):
    print(f"\n[9/10] Quantization simulation")
    if model is None:
        cr.record("Quantization", False, "Model not loaded."); return

    import torch
    from src.evaluation.evaluator import _simulate_quantize, compute_token_accuracy
    from src.data.tofu_dataset import get_tofu_dataloaders

    max_length = _get(config, "dataset.max_length", default=256)
    device     = next(p for p in model.parameters() if p.device.type != "meta").device

    try:
        fl, _, _ = get_tofu_dataloaders(
            tokenizer, batch_size=1, max_length=max_length, num_workers=0,
        )
        for prec in ["bf16", "int8", "int4"]:
            try:
                q   = _simulate_quantize(model, prec)
                q.to(device)
                acc = compute_token_accuracy(q, fl, str(device), max_batches=2)
                cr.record(f"Quant simulate {prec}", True, f"acc={acc:.3f}")
                del q; torch.cuda.empty_cache()
            except Exception as e:
                cr.record(f"Quant simulate {prec}", False, str(e))
    except Exception as e:
        cr.record("Quantization setup", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Check 10: Checkpoint + downstream loader
# ─────────────────────────────────────────────────────────────────────────────

def check_checkpoint_and_downstream(model, tokenizer, config):
    print(f"\n[10/10] Checkpoint save/load + downstream dataloader")
    import torch

    # Checkpoint
    if model is not None:
        from src.utils.checkpoint import CheckpointManager
        ckpt_dir = _get(config, "paths.checkpoints", default="checkpoints")
        test_dir = os.path.join(ckpt_dir, "_preflight_test_")
        try:
            cm = CheckpointManager(ckpt_dir)
            cm.save("_preflight_test_", model, tokenizer, {"test": 1.0}, config)
            data = cm.load_result("_preflight_test_")
            ok   = data is not None and "metrics" in data
            cr.record("Checkpoint save+load", ok)
        except Exception as e:
            cr.record("Checkpoint save+load", False, str(e))
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
    else:
        cr.record("Checkpoint save+load", False, "Model not loaded.")

    # Downstream dataloader
    if tokenizer is not None:
        from src.data.data_utils import get_downstream_dataloader
        max_length = _get(config, "dataset.max_length", default=256)
        for ds_name in ["alpaca", "c4", "gsm8k"]:
            try:
                loader = get_downstream_dataloader(
                    tokenizer, datasets=[ds_name],
                    n_samples_per_dist=10, max_length=max_length,
                    batch_size=2, num_workers=0,
                )
                n = len(loader.dataset)
                cr.record(f"Downstream loader: {ds_name}", n >= 5,
                          f"Only {n} samples" if n < 5 else f"{n} samples ✓")
            except Exception as e:
                cr.record(f"Downstream loader: {ds_name}", False, str(e))
    else:
        cr.record("Downstream loaders", False, "Tokenizer not loaded.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--skip_model_load", action="store_true",
                        help="Skip checks 6-10 (fast, ~2 min)")
    args   = parser.parse_args()
    config = load_config(args.config)

    print(f"\n{'='*60}")
    print(f"  DurableUn Pre-Flight Check")
    print(f"  Config: {args.config}")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    check_packages()
    check_cuda()
    token = check_token_file()
    check_hf_login(token)
    check_tofu_dataset(config)

    model, tokenizer = None, None
    if not args.skip_model_load:
        model, tokenizer = check_model_load(config)
        check_forward_pass(model, tokenizer, config)
        check_one_training_step(config)
        check_quantization(model, tokenizer, config)
        check_checkpoint_and_downstream(model, tokenizer, config)
    else:
        print("\n[6-10] Skipped (--skip_model_load)")

    all_passed = cr.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
