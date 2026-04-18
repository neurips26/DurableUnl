"""TOFU Dataset loader (locuslab/TOFU on HuggingFace)."""
import logging
from typing import Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

VALID_SPLITS = [
    "forget01", "forget05", "forget10",
    "retain90", "retain95", "retain99",
    "world_facts", "full",
]


def _load_tofu_split(split_name: str, cache_dir: Optional[str] = None):
    raw = load_dataset("locuslab/TOFU", split_name, cache_dir=cache_dir)
    if hasattr(raw, "keys"):
        keys = list(raw.keys())
        if "train" in raw:
            return raw["train"]
        return raw[keys[0]]
    return raw


class TOFUDataset(Dataset):
    def __init__(self, tokenizer, split="forget10", max_length=512, cache_dir=None):
        assert split in VALID_SPLITS
        self.tokenizer  = tokenizer
        self.max_length = max_length
        if tokenizer.pad_token is None:
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Loading TOFU split: {split}")
        self.data = _load_tofu_split(split, cache_dir)
        logger.info(f"TOFU '{split}': {len(self.data)} samples")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item   = self.data[idx]
        text   = f"Question: {item['question']}\nAnswer: {item['answer']}"
        q_text = f"Question: {item['question']}\nAnswer: "
        enc    = self.tokenizer(text, max_length=self.max_length, truncation=True,
                                padding="max_length", return_tensors="pt")
        ids    = enc["input_ids"].squeeze(0)
        mask   = enc["attention_mask"].squeeze(0)
        lbl    = ids.clone()
        lbl[mask == 0] = -100
        q_len  = self.tokenizer(q_text, max_length=self.max_length, truncation=True,
                                 return_tensors="pt")["input_ids"].shape[1]
        lbl[:q_len] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": lbl,
                "question": item["question"], "answer": item["answer"]}


def _collate(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]     for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
        "question":       [b["question"] for b in batch],
        "answer":         [b["answer"]   for b in batch],
    }


def get_tofu_dataloaders(tokenizer, forget_split="forget10", retain_split="retain90",
                         batch_size=4, eval_batch_size=8, max_length=512,
                         cache_dir=None, num_workers=0):
    forget_ds = TOFUDataset(tokenizer, forget_split, max_length, cache_dir)
    retain_ds = TOFUDataset(tokenizer, retain_split, max_length, cache_dir)
    eval_ds   = TOFUDataset(tokenizer, "world_facts", max_length, cache_dir)
    for name, ds in [("forget", forget_ds), ("retain", retain_ds)]:
        if len(ds) == 0:
            raise RuntimeError(f"TOFU '{name}' returned 0 samples. Check hf_token.py.")
    eff_bs = min(batch_size, len(forget_ds), len(retain_ds))
    if eff_bs < batch_size:
        logger.warning(f"Reducing batch_size {batch_size} -> {eff_bs}")
    logger.info(f"TOFU loaders: forget={len(forget_ds)}, retain={len(retain_ds)}, "
                f"world_facts={len(eval_ds)} | batch_size={eff_bs}")
    kw = dict(collate_fn=_collate, pin_memory=False, num_workers=num_workers)
    return (
        DataLoader(forget_ds, batch_size=eff_bs, shuffle=True,  drop_last=False, **kw),
        DataLoader(retain_ds, batch_size=eff_bs, shuffle=True,  drop_last=False, **kw),
        DataLoader(eval_ds,   batch_size=min(eval_batch_size, max(1,len(eval_ds))),
                   shuffle=False, drop_last=False, **kw),
    )
