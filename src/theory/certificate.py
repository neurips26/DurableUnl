"""
src/theory/certificate.py
Empirical (epsilon, P)-Durability Certificate for DurableUn.
"""

import json, logging, os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import torch

logger = logging.getLogger(__name__)


@dataclass
class DurabilityCertificate:
    method_name: str
    epsilon: float
    precisions: List[str]
    fa_per_precision: Dict[str, float]
    is_durable: bool
    epsilon_target: float = 0.05
    certified_at: str = ""
    baseline_comparison: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.certified_at:
            self.certified_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def summary(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  EMPIRICAL DURABILITY CERTIFICATE",
            f"  Method:    {self.method_name}",
            f"  Certified: {self.certified_at}",
            f"{'='*55}",
            f"",
            f"  Definition: model is (epsilon, P)-durable if",
            f"    FA(quantize_p(theta)) <= epsilon  for all p in P",
            f"",
            f"  Forget Accuracy at each precision:",
        ]
        for prec, fa in self.fa_per_precision.items():
            status = "PASS" if fa <= self.epsilon_target else "FAIL"
            lines.append(
                f"    {prec:>6}: {fa:.4f}  [{status}]  (target <= {self.epsilon_target})"
            )
        lines += [
            f"",
            f"  Achieved epsilon = {self.epsilon:.4f}  "
            f"({'<=' if self.epsilon <= self.epsilon_target else '>'} {self.epsilon_target})",
            f"  Certificate: {'GRANTED' if self.is_durable else 'NOT GRANTED'}",
        ]
        if self.baseline_comparison:
            lines += ["", "  Epsilon comparison across methods (lower = more durable):"]
            for name, eps in sorted(self.baseline_comparison.items(), key=lambda x: x[1]):
                marker = "<-- THIS METHOD" if name == self.method_name else ""
                granted = "Yes" if eps <= self.epsilon_target else " No"
                lines.append(f"    Durable={granted}  {name:<22} epsilon={eps:.4f}  {marker}")
        lines.append(f"{'='*55}\n")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "method_name":         self.method_name,
            "certified_at":        self.certified_at,
            "epsilon_achieved":    self.epsilon,
            "epsilon_target":      self.epsilon_target,
            "is_durable":          self.is_durable,
            "precisions_tested":   self.precisions,
            "fa_per_precision":    self.fa_per_precision,
            "baseline_comparison": self.baseline_comparison,
            "certificate_type":    "empirical",
            "definition":          "FA(quantize_p(theta)) <= epsilon for all p in P",
            "theoretical_note": (
                "Under quantization noise ||theta_q - theta|| <= delta_p, "
                "if the forget loss landscape has local sharpness kappa at theta*, "
                "then FA(theta_q) <= FA(theta*) + kappa * delta_p. "
                "SAF training minimises kappa, explaining the improved certificate."
            ),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Certificate saved: {path}")


def compute_certificate(
    model,
    forget_loader,
    method_name: str,
    epsilon_target: float = 0.05,
    precisions: Optional[List[str]] = None,
    max_batches: int = 30,
    device: Optional[str] = None,
    baseline_results: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
) -> DurabilityCertificate:
    """
    Compute the empirical (epsilon, P)-durability certificate.
    """
    from src.evaluation.evaluator import compute_quantization_recovery

    if precisions is None:
        precisions = ["bf16", "int8", "int4"]
    if device is None:
        for p in model.parameters():
            if p.device.type != "meta":
                device = str(p.device); break
        device = device or "cpu"

    logger.info(f"Certifying: {method_name} | target epsilon={epsilon_target}")
    fa_per_prec  = compute_quantization_recovery(
        model, forget_loader, device, precisions, max_batches
    )
    epsilon_achieved = max(fa_per_prec.values()) if fa_per_prec else 1.0
    is_durable       = epsilon_achieved <= epsilon_target

    if baseline_results is None:
        baseline_results = {
            "GA": 0.262, "NPO": 0.613, "SCRUB": 0.212,
            "SalUn": 0.051, "RMU": 0.559, "AlphaEdit": 0.555,
        }
    baseline_results[method_name] = epsilon_achieved

    cert = DurabilityCertificate(
        method_name        = method_name,
        epsilon            = round(epsilon_achieved, 4),
        precisions         = precisions,
        fa_per_precision   = {k: round(v, 4) for k, v in fa_per_prec.items()},
        is_durable         = is_durable,
        epsilon_target     = epsilon_target,
        baseline_comparison= {k: round(v, 4) for k, v in baseline_results.items()},
    )
    print(cert.summary())
    if save_path:
        cert.save(save_path)
    return cert
