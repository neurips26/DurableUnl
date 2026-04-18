"""Data utilities — downstream samplers for fine-tuning attack."""
import logging
import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"Seed set to {seed}")


class SimpleTextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):
        self.texts   = texts
        self.tok     = tokenizer
        self.max_len = max_length
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc  = self.tok(self.texts[idx], max_length=self.max_len, truncation=True,
                        padding="max_length", return_tensors="pt")
        ids  = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        lbl  = ids.clone(); lbl[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": lbl}


_DOWNSTREAM_DATASETS = {
    "alpaca": ("yahma/alpaca-cleaned", None,   "train",
               lambda i: f"Instruction: {i.get('instruction','')}\nResponse: {i.get('output','')}"),
    "c4":     ("allenai/c4",           "en",   "train",
               lambda i: i.get("text","")[:400]),
    "gsm8k":  ("openai/gsm8k",         "main", "train",
               lambda i: f"Problem: {i.get('question','')}\nSolution: {i.get('answer','')}"),
}

_FALLBACK_TEXTS = [
    "The capital of France is Paris. It is known for the Eiffel Tower.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language for data science.",
    "The sun is approximately 93 million miles from Earth.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Neural networks are inspired by the human brain's structure.",
    "Gradient descent is the core optimisation algorithm in deep learning.",
    "The transformer architecture introduced self-attention mechanisms.",
    "BERT and GPT are both based on transformer architecture.",
    "Backpropagation computes gradients through the computation graph.",
] * 50


def _load_single_dataset(name: str, n_samples: int, seed: int) -> List[str]:
    cfg = _DOWNSTREAM_DATASETS.get(name)
    if cfg is None:
        return []
    hf_name, subset, split, extract = cfg
    try:
        logger.info(f"Sampling {n_samples} from '{name}'")
        ds     = load_dataset(hf_name, subset, split=split, streaming=True)
        ds     = ds.shuffle(seed=seed, buffer_size=5000)
        texts  = []
        for item in ds:
            t = extract(item)
            if t and t.strip():
                texts.append(t.strip()[:400])
            if len(texts) >= n_samples:
                break
        logger.info(f"  -> {len(texts)} samples from '{name}'")
        return texts
    except Exception as e:
        logger.warning(f"Failed to load '{name}': {e}")
        return []


def get_downstream_dataloader(
    tokenizer,
    datasets: Optional[List[str]] = None,
    n_samples_per_dist: int = 200,
    max_length: int = 256,
    batch_size: int = 4,
    cache_dir: Optional[str] = None,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    if datasets is None:
        datasets = ["alpaca", "c4", "gsm8k"]

    all_texts = []
    for name in datasets:
        all_texts.extend(_load_single_dataset(name, n_samples_per_dist, seed))
        if len(all_texts) >= n_samples_per_dist:
            break

    if not all_texts:
        logger.warning("All downloads failed — using built-in fallback texts.")
        all_texts = _FALLBACK_TEXTS

    all_texts = all_texts[:n_samples_per_dist * len(datasets)]
    logger.info(f"Downstream loader: {len(all_texts)} total samples")
    dataset = SimpleTextDataset(all_texts, tokenizer, max_length)
    eff_bs  = min(batch_size, len(dataset))
    return DataLoader(dataset, batch_size=eff_bs, shuffle=True,
                      num_workers=num_workers, drop_last=False)
