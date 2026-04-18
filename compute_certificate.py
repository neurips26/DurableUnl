"""
compute_certificate.py — Compute empirical durability certificate.

Usage:
  python compute_certificate.py --checkpoint checkpoints/saf_alpha_3p0
  python compute_certificate.py --checkpoint checkpoints/saf_alpha_3p0_lambda_4p0
"""

import argparse, os, sys, json
import torch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Checkpoint folder, e.g. checkpoints/saf_alpha_3p0")
    p.add_argument("--config",  default="configs/durableun_config.yaml")
    p.add_argument("--epsilon", type=float, default=0.05)
    return p.parse_args()


def _get(cfg, *keys, default=None):
    for k in keys:
        v = cfg
        try:
            for part in k.split("."): v = v[part]
            return v
        except (KeyError, TypeError): pass
    return default


def _ensure_adapter_config(ckpt_model_dir: str, lora_cfg: dict):
    """
    If adapter_config.json is missing (save_pretrained didn't write it),
    create it from the training config so PeftModel.from_pretrained works.
    """
    adapter_cfg_path = os.path.join(ckpt_model_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        return   # already there

    print(f"  adapter_config.json missing — creating from config...")
    adapter_cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r":               lora_cfg.get("r", 16),
        "lora_alpha":      lora_cfg.get("lora_alpha", 32),
        "lora_dropout":    lora_cfg.get("lora_dropout", 0.05),
        "bias":            lora_cfg.get("bias", "none"),
        "target_modules":  lora_cfg.get("target_modules",
                           ["q_proj","v_proj","k_proj","o_proj"]),
        "fan_in_fan_out":  False,
        "modules_to_save": None,
        "inference_mode":  True,
        "base_model_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct",
    }
    with open(adapter_cfg_path, "w") as f:
        json.dump(adapter_cfg, f, indent=2)
    print(f"  Created: {adapter_cfg_path}")


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Use absolute path to avoid Windows backslash issues with PEFT
    ckpt_dir   = os.path.abspath(args.checkpoint)
    ckpt_model = os.path.join(ckpt_dir, "model")
    method_name = os.path.basename(ckpt_dir)
    model_name  = _get(config, "model.name",
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    cache_dir   = _get(config, "paths.cache_dir", default=None)
    lora_cfg    = config.get("lora", {})

    print(f"\nCheckpoint : {ckpt_dir}")
    print(f"Model dir  : {ckpt_model}")

    if not os.path.exists(ckpt_model):
        print(f"ERROR: {ckpt_model} does not exist.")
        print(f"Available checkpoints:")
        ckpt_base = os.path.join(os.path.dirname(ckpt_dir), "")
        for d in sorted(os.listdir(os.path.dirname(ckpt_dir))):
            print(f"  {d}")
        sys.exit(1)

    # Fix missing adapter_config.json
    _ensure_adapter_config(ckpt_model, lora_cfg)

    from src.models.model_utils import load_tokenizer
    from src.data.tofu_dataset import get_tofu_dataloaders
    from src.theory.certificate import compute_certificate
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    print("Loading tokenizer...")
    tok = load_tokenizer(model_name, cache_dir)

    print("Loading base model (4-bit)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cuda:0",
        quantization_config=bnb, cache_dir=cache_dir, trust_remote_code=True,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    print("Loading LoRA adapter from checkpoint...")
    model = PeftModel.from_pretrained(base, ckpt_model, is_trainable=False)
    model.eval()
    print("Model loaded.\n")

    print("Loading forget set...")
    fl, _, _ = get_tofu_dataloaders(
        tok,
        forget_split = _get(config, "dataset.forget_split", default="forget10"),
        retain_split = _get(config, "dataset.retain_split", default="retain90"),
        batch_size   = _get(config, "dataset.batch_size",   default=4),
        max_length   = _get(config, "dataset.max_length",   default=256),
        num_workers  = 0,
    )

    cert = compute_certificate(
        model         = model,
        forget_loader = fl,
        method_name   = method_name,
        epsilon_target= args.epsilon,
        save_path     = os.path.join(ckpt_dir, "certificate.json"),
    )

    print("\nLaTeX table row:")
    print(
        f"\\textbf{{{method_name}}} & "
        f"{cert.fa_per_precision.get('bf16','?')} & "
        f"{cert.fa_per_precision.get('int8','?')} & "
        f"\\textbf{{{cert.fa_per_precision.get('int4','?')}}} & "
        f"{'\\checkmark' if cert.is_durable else '\\times'} \\\\"
    )


if __name__ == "__main__":
    main()
