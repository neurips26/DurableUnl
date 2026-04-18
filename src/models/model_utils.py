"""
Model loading utilities.
Token is read from hf_token.py at the project root.
"""

import os
import logging
from typing import Optional, Dict, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)

_TOKEN_LOADED = False


def _login_hf():
    """
    Read HF_TOKEN from hf_token.py (project root) and:
      1. Log in via huggingface_hub
      2. Set HF_TOKEN env variable (so datasets library uses it too)

    Uses explicit UTF-8 encoding to avoid Windows cp1252 codec errors.
    Called before every model/tokenizer load.
    """
    global _TOKEN_LOADED
    if _TOKEN_LOADED:
        return   # only log in once per process

    try:
        here       = os.path.dirname(os.path.abspath(__file__))
        root       = os.path.dirname(os.path.dirname(here))
        token_file = os.path.join(root, "hf_token.py")

        if not os.path.exists(token_file):
            logger.warning(
                "hf_token.py not found. "
                "Gated models (LLaMA-3) will return 401."
            )
            return

        # MUST use utf-8 — Windows default (cp1252) crashes on some byte sequences
        ns: Dict[str, Any] = {}
        with open(token_file, encoding="utf-8") as f:
            exec(f.read(), ns)

        token = ns.get("HF_TOKEN", "")

        if not token or "PASTE" in token.upper():
            logger.warning(
                "hf_token.py found but token is still the placeholder. "
                "Edit hf_token.py and paste your real HuggingFace token."
            )
            return

        # Set env variable — datasets library reads this automatically
        os.environ["HF_TOKEN"]          = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token   # legacy env var

        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        logger.info("HuggingFace login successful.")
        _TOKEN_LOADED = True

    except Exception as e:
        logger.warning(f"HuggingFace login failed: {e}")


def _get_device(model) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(model_name: str, cache_dir: Optional[str] = None):
    _login_hf()
    tok = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token    = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    return tok


def load_base_model(
    model_name: str,
    dtype: str = "bfloat16",
    device_map: str = "cuda:0",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    cache_dir: Optional[str] = None,
):
    _login_hf()

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }.get(dtype, torch.bfloat16)

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    logger.info(
        f"Loading {model_name} | dtype={dtype} | "
        f"4bit={load_in_4bit} | device_map={device_map}"
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype if bnb_config is None else None,
            device_map=device_map,
            quantization_config=bnb_config,
            attn_implementation="sdpa",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.warning(f"sdpa failed ({e}), trying eager")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype if bnb_config is None else None,
            device_map=device_map,
            quantization_config=bnb_config,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

    model.config.use_cache = False
    total = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"Model loaded: {total:.2f}B params on {_get_device(model)}")
    return model


def load_model_with_lora(
    model_name: str,
    lora_config: Optional[Dict] = None,
    dtype: str = "bfloat16",
    device_map: str = "cuda:0",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    cache_dir: Optional[str] = None,
) -> Tuple[Any, Any]:
    tokenizer = load_tokenizer(model_name, cache_dir)
    model     = load_base_model(
        model_name, dtype=dtype, device_map=device_map,
        load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit,
        cache_dir=cache_dir,
    )

    if load_in_4bit or load_in_8bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    lc       = lora_config or {}
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lc.get("r", 64),
        lora_alpha=lc.get("lora_alpha", 128),
        target_modules=lc.get(
            "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
        ),
        lora_dropout=lc.get("lora_dropout", 0.05),
        bias=lc.get("bias", "none"),
        inference_mode=False,
    )

    model     = get_peft_model(model, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA applied: {trainable:,} / {total:,} params trainable "
        f"({100*trainable/total:.2f}%)"
    )
    return model, tokenizer
