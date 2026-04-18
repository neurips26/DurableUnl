"""Baseline unlearning methods registry and factory."""

from .gradient_ascent import GradientAscent
from .npo import NPO
from .scrub import SCRUB
from .salun import SalUn
from .rmu import RMU
from .alpha_edit import AlphaEdit
from .base import BaseUnlearner, UnlearningResult, _clm_loss

BASELINE_MAP = BASELINE_REGISTRY = {
    "ga":         GradientAscent,
    "nga":        GradientAscent,
    "npo":        NPO,
    "scrub":      SCRUB,
    "salun":      SalUn,
    "rmu":        RMU,
    "alpha_edit": AlphaEdit,
}


def get_baseline(
    name: str,
    model,
    forget_loader=None,
    retain_loader=None,
    device=None,
    n_steps: int = 300,
    retain_lambda: float = 1.0,
    lr: float = 5e-5,
    lr_forget: float = None,
    gradient_clip: float = 1.0,
    log_every: int = 50,
    **extra,
) -> BaseUnlearner:
    """
    Instantiate a baseline unlearner with the correct CLM loss function.

    Extra keyword args (beta, gamma, saliency_threshold, layer_id, svd_rank, etc.)
    are forwarded to the subclass.
    """
    key = name.lower()
    if key not in BASELINE_MAP:
        raise ValueError(f"Unknown baseline '{name}'. Available: {sorted(BASELINE_MAP.keys())}")

    return BASELINE_MAP[key](
        model=model,
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        loss_fn=_clm_loss,           # always correct — never nn.CrossEntropyLoss
        device=device,
        n_steps=n_steps,
        retain_lambda=retain_lambda,
        lr=lr_forget or lr,
        lr_forget=lr_forget,
        gradient_clip=gradient_clip,
        log_every=log_every,
        **extra,
    )


# alias
get_unlearner = get_baseline

__all__ = [
    "GradientAscent", "NPO", "SCRUB", "SalUn", "RMU", "AlphaEdit",
    "BaseUnlearner", "UnlearningResult",
    "BASELINE_MAP", "BASELINE_REGISTRY",
    "get_baseline", "get_unlearner",
]
