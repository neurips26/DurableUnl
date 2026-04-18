"""
experiments/revision_realquant_eval.py
=========================================
Reviewer ask: "one rigorous evaluation with a standard PTQ pipeline."

This script answers the reviewer's core objection by using REAL bitsandbytes
INT4/INT8 quantization on MERGED models, not our custom simulator.

Pipeline for each checkpoint:
  1. Load LoRA checkpoint (base NF4 + LoRA adapters)
  2. Merge LoRA into base: model.merge_and_unload()
  3. Save merged model to disk in bfloat16 (full precision)
  4. Reload saved model with bitsandbytes NF4 (real production INT4)
  5. Reload saved model with bitsandbytes INT8 (real production INT8)
  6. Evaluate FA on TOFU forget10 for each precision
  7. Compare against our symmetric simulator results

This directly validates whether the INT4 attack holds under the actual
quantization stack used in production LLM deployment.

Disk space: ~16 GB per merged model (bfloat16 safetensors)
Total disk needed: ~4 models x 16 GB = ~64 GB
Make sure you have space on your Windows C: or D: drive.

Usage:
  py -m experiments.revision_realquant_eval

  # Just one checkpoint (faster, ~30 min):
  py -m experiments.revision_realquant_eval --checkpoint ga_tofu_s42

  # All key checkpoints (~2 hours):
  py -m experiments.revision_realquant_eval

Expected results:
  - Real NF4 Q should be similar to our simulator Q-INT4
  - INT8 should remain near 0 (confirms simulator)
  - This validates the attack under production quantization

Note on disk: merged models are deleted after evaluation unless --keep_merged.
"""

import argparse, csv, json, logging, os, sys, shutil, time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.logging_utils import setup_root_logger, now_str, file_ts

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default=None,
                   help="Single checkpoint name (e.g. ga_tofu_s42). "
                        "If not set, evaluates all key checkpoints.")
    p.add_argument("--keep_merged", action="store_true",
                   help="Keep merged model on disk (default: delete after eval)")
    p.add_argument("--merge_dir",   default=None,
                   help="Where to save merged models (default: checkpoints/merged/)")
    p.add_argument("--max_batches", type=int, default=30)
    return p.parse_args()


# Key checkpoints to evaluate (adjust if yours have different names)
DEFAULT_CHECKPOINTS = [
    "ga_tofu_s42",
    "salun_tofu_s42",
    "graddiff_tofu_s42",
    "saf_alpha3p0_tofu_s42",
]


# ── Step 1: Merge LoRA and save to disk ──────────────────────────────────────

def merge_and_save(ckpt_dir: str, model_name: str, save_dir: str,
                   cache_dir=None) -> str:
    """
    Load LoRA checkpoint, merge adapters into base, save merged model.
    Returns path to saved merged model directory.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training

    ckpt_model = os.path.join(ckpt_dir, "model")
    _ensure_adapter_config(ckpt_model, model_name)

    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"  Loading base model for merge...")

    # Load in full bfloat16 (NOT 4-bit) for clean merge
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    base.config.use_cache = False

    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token    = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    vram_base = _vram()
    logger.info(f"  Base (bfloat16) VRAM: {vram_base:.2f} GB")

    logger.info(f"  Loading LoRA adapters from {ckpt_model}")
    model = PeftModel.from_pretrained(base, ckpt_model, is_trainable=False)

    logger.info(f"  Merging LoRA into base...")
    t0 = time.time()
    merged = model.merge_and_unload()
    logger.info(f"  Merge took {time.time()-t0:.1f}s | VRAM: {_vram():.2f} GB")

    logger.info(f"  Saving merged model to {save_dir} ...")
    merged.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)

    del merged, model, base
    torch.cuda.empty_cache()
    logger.info(f"  Saved. Disk: {_dir_size_gb(save_dir):.1f} GB")
    return save_dir


# ── Step 2: Reload with bitsandbytes and evaluate ────────────────────────────

def eval_with_bnb_quant(
    merged_model_dir: str,
    forget_loader,
    device: str,
    quant_type: str,   # "bf16", "int8", "nf4_int4"
    max_batches: int = 30,
    cache_dir=None,
) -> float:
    """
    Load merged model with bitsandbytes quantization and evaluate FA.

    quant_type options:
      "bf16"     — no quantization, full precision baseline
      "int8"     — bitsandbytes INT8 (load_in_8bit=True)
      "nf4_int4" — bitsandbytes NF4 4-bit (production INT4 equivalent)
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    logger.info(f"  Loading merged model with quant={quant_type}...")
    t0 = time.time()

    if quant_type == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            merged_model_dir,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    elif quant_type == "int8":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            merged_model_dir,
            device_map="cuda:0",
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
    elif quant_type == "nf4_int4":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            merged_model_dir,
            device_map="cuda:0",
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}")

    model.eval()
    model.config.use_cache = False
    load_time = time.time() - t0
    logger.info(f"  Load time: {load_time:.1f}s | VRAM: {_vram():.2f} GB")

    # Evaluate FA
    correct = total = 0
    with torch.no_grad():
        for i, batch in enumerate(forget_loader):
            if i >= max_batches: break
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch.get("labels", ids).to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            valid  = (shift_labels != -100) & mask[:, 1:].bool()
            preds  = shift_logits.argmax(dim=-1)
            correct += (preds[valid] == shift_labels[valid]).sum().item()
            total   += valid.sum().item()

    fa = correct / max(total, 1)
    logger.info(f"  FA@{quant_type}: {fa:.4f}")

    del model
    torch.cuda.empty_cache()
    return fa


# ── Main ─────────────────────────────────────────────────────────────────────

def _vram():
    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def _dir_size_gb(path):
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try: total += os.path.getsize(os.path.join(dirpath, f))
            except: pass
    return total / 1e9


def _ensure_adapter_config(ckpt_model_dir, model_name):
    cfg_path = os.path.join(ckpt_model_dir, "adapter_config.json")
    if os.path.exists(cfg_path): return
    with open(cfg_path, "w") as f:
        json.dump({
            "peft_type": "LORA", "task_type": "CAUSAL_LM",
            "r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none",
            "target_modules": ["q_proj","v_proj","k_proj","o_proj"],
            "fan_in_fan_out": False, "inference_mode": True,
            "base_model_name_or_path": model_name,
        }, f)


def main():
    args = parse_args()
    setup_root_logger("logs")
    logger = logging.getLogger("revision_realquant_eval")

    import yaml
    cfg_path = os.path.join(ROOT, "configs", "base_config.yaml")
    with open(cfg_path) as f: cfg = yaml.safe_load(f)

    def _get(cfg, *keys, default=None):
        for k in keys:
            v = cfg
            try:
                for part in k.split("."): v = v[part]
                return v
            except: pass
        return default

    model_name = _get(cfg, "model.name",
                      default="meta-llama/Meta-Llama-3-8B-Instruct")
    ckpt_base  = _get(cfg, "paths.checkpoints", default="checkpoints")
    cache_dir  = _get(cfg, "paths.cache_dir")
    res_dir    = _get(cfg, "paths.results",     default="results")
    merge_base = args.merge_dir or os.path.join(ckpt_base, "merged")
    os.makedirs(res_dir,    exist_ok=True)
    os.makedirs(merge_base, exist_ok=True)

    # Dataset
    from src.models.model_utils import load_tokenizer
    from src.data.tofu_dataset import get_tofu_dataloaders
    tok = load_tokenizer(model_name, cache_dir)
    fl, _, _ = get_tofu_dataloaders(
        tok, forget_split="forget10", retain_split="retain90",
        batch_size=4, max_length=256, num_workers=0,
    )
    device = "cuda:0"

    # Which checkpoints
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = DEFAULT_CHECKPOINTS

    # Our simulator results (from existing results) for comparison
    simulator_results = {
        "ga_tofu_s42":          {"sim_int8": 0.000, "sim_int4": 0.262},
        "salun_tofu_s42":       {"sim_int8": 0.000, "sim_int4": 0.051},
        "graddiff_tofu_s42":    {"sim_int8": 0.000, "sim_int4": 0.151},
        "saf_alpha3p0_tofu_s42":{"sim_int8": 0.000, "sim_int4": 0.044},
    }

    csv_path = os.path.join(res_dir, f"revision_realquant_{file_ts()}.csv")
    all_rows = []

    for ckpt_name in checkpoints:
        ckpt_dir = os.path.join(ckpt_base, ckpt_name)
        if not os.path.exists(os.path.join(ckpt_dir, "model")):
            logger.warning(f"SKIP {ckpt_name}: checkpoint not found")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"  {ckpt_name}")
        logger.info(f"{'='*60}")

        merge_dir = os.path.join(merge_base, ckpt_name)
        row = {"checkpoint": ckpt_name}

        try:
            # ── Merge ────────────────────────────────────────────────────────
            if os.path.exists(os.path.join(merge_dir, "config.json")):
                logger.info(f"  Merged model already exists at {merge_dir}")
            else:
                logger.info(f"  Merging LoRA adapters...")
                logger.info(f"  NOTE: Loading in bfloat16 requires ~15-16 GB VRAM.")
                logger.info(f"  If OOM: your GPU has 24 GB, this should fit.")
                merge_and_save(ckpt_dir, model_name, merge_dir, cache_dir)

            # ── Evaluate at each precision ────────────────────────────────────
            for qtype, key in [
                ("bf16",     "fa_bf16"),
                ("int8",     "fa_bnb_int8"),
                ("nf4_int4", "fa_bnb_nf4"),
            ]:
                try:
                    fa = eval_with_bnb_quant(
                        merge_dir, fl, device, qtype, args.max_batches, cache_dir
                    )
                    row[key] = round(fa, 4)
                except torch.cuda.OutOfMemoryError:
                    logger.error(f"  OOM loading merged model at {qtype}")
                    row[key] = -1.0
                except Exception as e:
                    logger.error(f"  Failed at {qtype}: {e}")
                    row[key] = -1.0

            # ── Simulator comparison ──────────────────────────────────────────
            sim = simulator_results.get(ckpt_name, {})
            row["sim_int4"] = sim.get("sim_int4", -1)
            row["sim_int8"] = sim.get("sim_int8", -1)

            # Is the attack confirmed?
            bnb_nf4 = row.get("fa_bnb_nf4", -1)
            bf16    = row.get("fa_bf16",    -1)
            if bnb_nf4 >= 0 and bf16 >= 0:
                recovery = bnb_nf4 / max(bf16, 1e-6)
                row["bnb_recovery_ratio"] = round(recovery, 2)
                row["attack_confirmed"] = "Y" if bnb_nf4 > bf16 * 1.5 else "N"
            else:
                row["bnb_recovery_ratio"] = -1
                row["attack_confirmed"] = "?"

            all_rows.append(row)

            write_hdr = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
                if write_hdr: w.writeheader()
                w.writerow(row)

            logger.info(f"\n  {ckpt_name} summary:")
            logger.info(f"    FA@BF16:      {row.get('fa_bf16',    '?')}")
            logger.info(f"    FA@BnB-INT8:  {row.get('fa_bnb_int8','?')} "
                        f"(sim: {row.get('sim_int8','?')})")
            logger.info(f"    FA@BnB-NF4:   {row.get('fa_bnb_nf4', '?')} "
                        f"(sim: {row.get('sim_int4','?')})")
            logger.info(f"    Recovery:     {row.get('bnb_recovery_ratio','?')}x  "
                        f"Attack confirmed: {row.get('attack_confirmed','?')}")

        except Exception as e:
            logger.error(f"FAILED {ckpt_name}: {e}", exc_info=True)
            torch.cuda.empty_cache()
            continue
        finally:
            # Clean up merged model unless --keep_merged
            if not args.keep_merged and os.path.exists(merge_dir):
                logger.info(f"  Removing merged model ({_dir_size_gb(merge_dir):.1f} GB)...")
                shutil.rmtree(merge_dir)

    # ── Final summary table ───────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("REAL PTQ EVALUATION RESULTS (for paper)")
    logger.info("Answers reviewer: 'standard PTQ pipeline'")
    logger.info(f"{'='*70}")
    logger.info(
        f"{'Checkpoint':<28} {'BF16':>6} {'BnB-INT8':>9} {'BnB-NF4':>8} "
        f"{'Sim-INT4':>9} {'Ratio':>6} {'Confirmed':>10}"
    )
    logger.info("-"*80)
    for r in all_rows:
        logger.info(
            f"  {r['checkpoint']:<26} "
            f"{r.get('fa_bf16',    '?'):>6} "
            f"{r.get('fa_bnb_int8','?'):>9} "
            f"{r.get('fa_bnb_nf4', '?'):>8} "
            f"{r.get('sim_int4',   '?'):>9} "
            f"{r.get('bnb_recovery_ratio','?'):>6} "
            f"{r.get('attack_confirmed','?'):>10}"
        )

    logger.info(f"\nCSV: {csv_path}")
    logger.info("\nLaTeX snippet (Table for paper):")
    logger.info(r"\begin{tabular}{lcccc}")
    logger.info(r"\toprule")
    logger.info(r"Method & FA@BF16 & FA@BnB-INT8 & FA@BnB-NF4 & Attack \\ \midrule")
    for r in all_rows:
        name = r["checkpoint"].replace("_tofu_s42","").replace("saf_alpha3p0","SAF α=3").upper()
        logger.info(
            f"{name} & {r.get('fa_bf16','?')} & "
            f"{r.get('fa_bnb_int8','?')} & "
            f"{r.get('fa_bnb_nf4','?')} & "
            f"{r.get('attack_confirmed','?')} \\\\"
        )
    logger.info(r"\bottomrule\end{tabular}")

    logger.info("\nKey message:")
    confirmed = sum(1 for r in all_rows if r.get("attack_confirmed") == "Y")
    logger.info(f"  Attack confirmed under real BnB PTQ: {confirmed}/{len(all_rows)} checkpoints")
    if confirmed == len(all_rows):
        logger.info("  -> Full validation: INT4 attack holds under production quantization stack")
    elif confirmed > 0:
        logger.info("  -> Partial validation: INT4 attack holds for some methods")


if __name__ == "__main__":
    main()
