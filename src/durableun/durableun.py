"""
DurableUn: Full three-mechanism pipeline SAF → OWD → QRS
"""
import logging
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader

from .saf import SAF
from .owd import OWD
from .qrs import QRS

logger = logging.getLogger(__name__)


class DurableUn:
    """
    Full DurableUn pipeline: SAF → OWD → QRS

    Can be run all at once or phase by phase.
    Each phase saves a checkpoint so you can resume.
    """

    def __init__(self, model, tokenizer, forget_loader, retain_loader,
                 config: Dict[str, Any], device=None, ckpt_dir: str = "checkpoints"):
        self.model         = model
        self.tokenizer     = tokenizer
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self.config        = config
        self.device        = device
        self.ckpt_dir      = ckpt_dir

    def _save_phase_checkpoint(self, phase: str, metrics: dict):
        path = os.path.join(self.ckpt_dir, f"durableun_{phase}")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(os.path.join(path, "model"))
        self.tokenizer.save_pretrained(os.path.join(path, "model"))
        with open(os.path.join(path, "result.json"), "w") as f:
            json.dump({"phase": phase, "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "metrics": metrics}, f, indent=2)
        logger.info(f"Phase {phase} checkpoint saved: {path}")

    def run_saf(self) -> dict:
        cfg = self.config.get("saf", {})
        saf = SAF(
            model          = self.model,
            forget_loader  = self.forget_loader,
            retain_loader  = self.retain_loader,
            device         = self.device,
            n_steps        = cfg.get("n_steps", 300),
            lr             = self.config.get("training", {}).get("lr", 5e-5),
            retain_lambda  = self.config.get("training", {}).get("retain_lambda", 1.0),
            gradient_clip  = self.config.get("training", {}).get("gradient_clip", 1.0),
            log_every      = self.config.get("training", {}).get("log_every", 50),
            rho            = cfg.get("rho", 0.05),
            mu_hessian     = cfg.get("mu_hessian", 0.01),
            n_hutchinson   = cfg.get("n_hutchinson", 4),
        )
        result = saf.unlearn()
        metrics = result.to_dict()
        self._save_phase_checkpoint("saf", metrics)
        return metrics

    def run_owd(self) -> dict:
        cfg = self.config.get("owd", {})
        owd = OWD(
            model                 = self.model,
            forget_loader         = self.forget_loader,
            retain_loader         = self.retain_loader,
            tokenizer             = self.tokenizer,
            device                = self.device,
            n_steps               = cfg.get("n_steps", 300),
            lr                    = self.config.get("training", {}).get("lr", 5e-5),
            retain_lambda         = self.config.get("training", {}).get("retain_lambda", 1.0),
            gradient_clip         = self.config.get("training", {}).get("gradient_clip", 1.0),
            log_every             = self.config.get("training", {}).get("log_every", 50),
            svd_rank              = cfg.get("svd_rank", 64),
            downstream_datasets   = cfg.get("downstream_datasets", ["alpaca", "c4"]),
            n_downstream_samples  = cfg.get("n_downstream_samples", 200),
            max_length            = self.config.get("dataset", {}).get("max_length", 256),
            cache_dir             = self.config.get("paths", {}).get("cache_dir"),
        )
        result = owd.unlearn()
        metrics = result.to_dict()
        self._save_phase_checkpoint("owd", metrics)
        return metrics

    def run_qrs(self) -> dict:
        cfg = self.config.get("qrs", {})
        qrs = QRS(
            model                = self.model,
            forget_loader        = self.forget_loader,
            retain_loader        = self.retain_loader,
            device               = self.device,
            n_steps              = cfg.get("n_steps", 300),
            lr                   = cfg.get("lr", 2e-5),
            retain_lambda        = self.config.get("training", {}).get("retain_lambda", 1.0),
            gradient_clip        = self.config.get("training", {}).get("gradient_clip", 1.0),
            log_every            = self.config.get("training", {}).get("log_every", 50),
            precisions           = cfg.get("precisions", ["int4", "int8", "bf16"]),
            forget_acc_threshold = cfg.get("forget_acc_threshold", 0.05),
            max_outer_iters      = cfg.get("max_outer_iters", 5),
            inner_steps          = cfg.get("inner_steps", 60),
        )
        result = qrs.unlearn()
        metrics = result.to_dict()
        self._save_phase_checkpoint("qrs", metrics)
        return metrics

    def run_all(self) -> dict:
        logger.info("DurableUn: Running full pipeline SAF → OWD → QRS")
        saf_m = self.run_saf()
        owd_m = self.run_owd()
        qrs_m = self.run_qrs()
        return {"saf": saf_m, "owd": owd_m, "qrs": qrs_m}
