"""
Checkpoint manager for DurableUn experiments.
Saves model + metadata after each completed stage.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import torch


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class CheckpointManager:
    """
    Saves and loads experiment checkpoints.

    Directory layout:
        checkpoints/
            ga/
                model/          ← HuggingFace save_pretrained() output
                result.json     ← metrics + timestamp
            npo/
                model/
                result.json
            ...
    """

    def __init__(self, base_dir: str = "checkpoints"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save(
        self,
        method_name: str,
        model,
        tokenizer,
        metrics: Dict[str, Any],
        config: Optional[Dict] = None,
    ) -> str:
        """
        Save model + metadata. Returns the checkpoint directory path.
        """
        ckpt_dir = os.path.join(self.base_dir, method_name)
        model_dir = os.path.join(ckpt_dir, "model")
        os.makedirs(model_dir, exist_ok=True)

        # Save model weights
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        # Save metadata
        meta = {
            "method": method_name,
            "saved_at": _ts(),
            "metrics": metrics,
        }
        if config:
            meta["config_snapshot"] = {
                "n_steps": config.get("training", {}).get("n_steps"),
                "model_name": config.get("model", {}).get("name"),
            }
        with open(os.path.join(ckpt_dir, "result.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        print(f"[{_ts()}] Checkpoint saved: {ckpt_dir}")
        return ckpt_dir

    def load_result(self, method_name: str) -> Optional[Dict]:
        """Load saved metrics for a method (or None if not found)."""
        path = os.path.join(self.base_dir, method_name, "result.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def exists(self, method_name: str) -> bool:
        """True if a checkpoint already exists for this method."""
        return os.path.exists(
            os.path.join(self.base_dir, method_name, "result.json")
        )

    def list_completed(self):
        """Return list of methods that already have checkpoints."""
        completed = []
        for name in os.listdir(self.base_dir):
            if self.exists(name):
                completed.append(name)
        return sorted(completed)
