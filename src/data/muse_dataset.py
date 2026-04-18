"""
src/data/muse_dataset.py
=========================
MUSE (Machine Unlearning Six-way Evaluation) Dataset Loader.
Shi et al. (2024) — https://arxiv.org/abs/2407.06460

MUSE-News structure (configs: knowmem, privleak, raw, scal, sust, train, verbmem):
  - 'raw'    → has forget/retain text splits (what we want for unlearning)
  - 'verbmem'→ verbatim memorization eval

Usage:
    from src.data.muse_dataset import get_muse_dataloaders
    forget_loader, retain_loader = get_muse_dataloaders(tokenizer, corpus="news")
"""

import logging
from typing import Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# MUSE uses config names, not just dataset splits
MUSE_CONFIGS = {
    "news": {
        "hf_id":        "muse-bench/MUSE-News",
        "config":       "raw",          # 'raw' has the actual article text
        "forget_split": "forget",
        "retain_split": "retain",
        "text_col":     "text",
        "description":  "BBC News articles",
    },
    "books": {
        "hf_id":        "muse-bench/MUSE-Books",
        "config":       "raw",
        "forget_split": "forget",
        "retain_split": "retain",
        "text_col":     "text",
        "description":  "Harry Potter excerpts",
    },
}


class MUSEDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        corpus: str = "news",
        split: str = "forget",
        max_length: int = 256,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        assert corpus in MUSE_CONFIGS, f"corpus must be one of {list(MUSE_CONFIGS)}"
        cfg = MUSE_CONFIGS[corpus]

        if tokenizer.pad_token is None:
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.text_col   = cfg["text_col"]

        logger.info(f"Loading MUSE-{corpus} config='{cfg['config']}' split='{split}' | {cfg['description']}")

        try:
            # Try with config name first (correct for MUSE)
            raw = load_dataset(
                cfg["hf_id"],
                cfg["config"],
                split=split,
                cache_dir=cache_dir,
            )
        except Exception as e1:
            logger.warning(f"Config '{cfg['config']}' failed: {e1}")
            try:
                # Fallback: try without config
                raw = load_dataset(cfg["hf_id"], split=split, cache_dir=cache_dir)
            except Exception as e2:
                logger.warning(f"Direct split failed: {e2}")
                # Final fallback: load full dataset and filter
                full = load_dataset(cfg["hf_id"], cfg["config"], cache_dir=cache_dir)
                if split in full:
                    raw = full[split]
                else:
                    # Use train split as retain, take first 10% as forget proxy
                    available = list(full.keys())
                    logger.warning(f"Split '{split}' not found. Available: {available}")
                    raw = full[available[0]]

        if hasattr(raw, "keys"):
            # Got a DatasetDict — take first available split
            keys = list(raw.keys())
            raw  = raw[split] if split in raw else raw[keys[0]]

        if max_samples and len(raw) > max_samples:
            raw = raw.select(range(max_samples))

        self.data = raw
        logger.info(f"MUSE-{corpus} '{split}': {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Handle different column names across MUSE configs
        text = None
        for col in [self.text_col, "text", "article", "content", "passage", "document"]:
            if col in item:
                text = item[col]
                break
        if text is None:
            # Last resort: concatenate all string values
            text = " ".join(str(v) for v in item.values() if isinstance(v, str))

        if not isinstance(text, str):
            text = str(text)
        text = f"Text: {text.strip()[:2000]}"  # cap at 2000 chars before tokenizing

        enc  = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        ids  = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        lbl  = ids.clone()
        lbl[mask == 0] = -100

        return {"input_ids": ids, "attention_mask": mask, "labels": lbl}


def get_muse_dataloaders(
    tokenizer,
    corpus: str = "news",
    max_length: int = 256,
    batch_size: int = 4,
    num_workers: int = 0,
    max_forget_samples: Optional[int] = None,
    max_retain_samples: Optional[int] = 500,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Returns (forget_loader, retain_loader) for a MUSE corpus."""
    forget_ds = MUSEDataset(
        tokenizer, corpus=corpus, split="forget",
        max_length=max_length, max_samples=max_forget_samples, cache_dir=cache_dir,
    )
    retain_ds = MUSEDataset(
        tokenizer, corpus=corpus, split="retain",
        max_length=max_length, max_samples=max_retain_samples, cache_dir=cache_dir,
    )
    forget_loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=False)
    retain_loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=False)
    logger.info(f"MUSE-{corpus}: forget={len(forget_ds)}, retain={len(retain_ds)} | batch={batch_size}")
    return forget_loader, retain_loader
