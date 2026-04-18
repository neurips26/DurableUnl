"""
src/data/dataset_registry.py
==============================
Unified dataset loader. One function to get any forget/retain loaders.

Usage:
    from src.data.dataset_registry import get_dataloaders

    # TOFU (existing)
    fl, rl, extra = get_dataloaders(tokenizer, dataset="tofu",
                                     forget_split="forget10")

    # MUSE News
    fl, rl, extra = get_dataloaders(tokenizer, dataset="muse_news")

    # MUSE Books
    fl, rl, extra = get_dataloaders(tokenizer, dataset="muse_books")

    # WikiBio Person Unlearning
    fl, rl, extra = get_dataloaders(tokenizer, dataset="wpu")

All loaders return the same {input_ids, attention_mask, labels} format.
"""

import logging
from typing import Optional, Tuple, Dict
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

AVAILABLE_DATASETS = ["tofu", "muse_news", "muse_books", "wpu"]


def get_dataloaders(
    tokenizer,
    dataset: str = "tofu",
    # TOFU-specific
    forget_split: str = "forget10",
    retain_split: str = "retain90",
    # MUSE-specific
    max_retain_samples: int = 500,
    # WPU-specific
    n_forget_persons: int = 50,
    max_persons: int = 5000,
    # Shared
    max_length: int = 256,
    batch_size: int = 4,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Unified dataset loader returning (forget_loader, retain_loader, extra_info).

    extra_info: dict with dataset metadata (name, sizes, etc.)
    """
    assert dataset in AVAILABLE_DATASETS, \
        f"dataset must be one of {AVAILABLE_DATASETS}. Got: {dataset}"

    if dataset == "tofu":
        from src.data.tofu_dataset import get_tofu_dataloaders
        fl, rl, wf = get_tofu_dataloaders(
            tokenizer,
            forget_split=forget_split,
            retain_split=retain_split,
            max_length=max_length,
            batch_size=batch_size,
            num_workers=num_workers,
            cache_dir=cache_dir,
        )
        extra = {
            "dataset": "tofu",
            "forget_split": forget_split,
            "retain_split": retain_split,
            "world_facts_loader": wf,
        }
        return fl, rl, extra

    elif dataset in ["muse_news", "muse_books"]:
        from src.data.muse_dataset import get_muse_dataloaders
        corpus = "news" if dataset == "muse_news" else "books"
        fl, rl = get_muse_dataloaders(
            tokenizer,
            corpus=corpus,
            max_length=max_length,
            batch_size=batch_size,
            num_workers=num_workers,
            max_retain_samples=max_retain_samples,
            cache_dir=cache_dir,
        )
        extra = {
            "dataset": dataset,
            "corpus": corpus,
            "world_facts_loader": None,
        }
        return fl, rl, extra

    elif dataset == "wpu":
        from src.data.wpu_dataset import get_wpu_dataloaders
        fl, rl = get_wpu_dataloaders(
            tokenizer,
            n_forget_persons=n_forget_persons,
            max_persons=max_persons,
            max_length=max_length,
            batch_size=batch_size,
            num_workers=num_workers,
            cache_dir=cache_dir,
            seed=seed,
        )
        extra = {
            "dataset": "wpu",
            "n_forget_persons": n_forget_persons,
            "world_facts_loader": None,
        }
        return fl, rl, extra

    raise ValueError(f"Unknown dataset: {dataset}")
