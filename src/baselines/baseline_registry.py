"""
src/baselines/baseline_registry.py
=====================================
Single registry mapping method names → unlearner classes.

Usage:
    from src.baselines.baseline_registry import get_baseline, BASELINE_NAMES

    unlearner = get_baseline("salun", model=model, forget_loader=fl, ...)
    unlearner.unlearn()

Available methods:
    Original baselines (Phase 0):
        ga, npo, scrub, salun, rmu, alpha_edit

    Modern baselines (added):
        graddiff        — Gradient Difference (Maini et al. TOFU 2024)
        wga             — Weighted Gradient Ascent (Jia et al. 2024)
        wga_lp          — WGA with label perturbation
        tv              — Task Vector Negation (Ilharco et al. 2023)
        dare            — DARE task vector (Yu et al. 2023)
        noisy_ga        — Noisy Gradient Ascent (Neel et al. 2021)
        langevin        — SGLD Langevin Unlearning (Chien et al. 2024)

    DurableUn methods:
        durableun_saf_v3    — SAF v3 (best FA, α=1)
        durableun_saf_alpha3 — SAF α=3 (certificate)
"""

from typing import Type
import torch.nn as nn

# ── Lazy import mapping ───────────────────────────────────────────────────────
# Each entry: name → (module_path, class_name, default_kwargs)

_REGISTRY = {
    # ── Original baselines ────────────────────────────────────────────────────
    "ga": (
        "src.baselines.base",       # fallback: inline in phase0
        "GradientAscent",
        {"retain_lambda": 1.0},
    ),
    "npo": (
        "src.baselines.npo",
        "NPO",
        {"retain_lambda": 1.0, "beta": 0.1},
    ),
    "scrub": (
        "src.baselines.scrub",
        "SCRUB",
        {"retain_lambda": 1.0},
    ),
    "salun": (
        "src.baselines.salun",
        "SalUn",
        {"retain_lambda": 1.0},
    ),
    "rmu": (
        "src.baselines.rmu",
        "RMU",
        {"retain_lambda": 1.0},
    ),
    "alpha_edit": (
        "src.baselines.alpha_edit",
        "AlphaEdit",
        {"retain_lambda": 1.0},
    ),

    # ── Modern baselines ──────────────────────────────────────────────────────
    "graddiff": (
        "src.baselines.gradient_difference",
        "GradDiff",
        {"retain_lambda": 1.0, "grad_diff_coeff": 1.0},
    ),
    "wga": (
        "src.baselines.wga",
        "WGA",
        {"retain_lambda": 1.0, "variant": "weighted", "temperature": 1.0},
    ),
    "wga_lp": (
        "src.baselines.wga",
        "WGA",
        {"retain_lambda": 1.0, "variant": "label_perturb"},
    ),
    "tv": (
        "src.baselines.tv_distance",
        "TaskVectorUnlearning",
        {"scale": 1.0, "method": "negate"},
    ),
    "dare": (
        "src.baselines.tv_distance",
        "TaskVectorUnlearning",
        {"scale": 1.0, "method": "dare", "dare_p": 0.9},
    ),
    "noisy_ga": (
        "src.baselines.langevin_unlearn",
        "NoisyGradientUnlearning",
        {"retain_lambda": 1.0, "noise_std": 0.01, "variant": "ng"},
    ),
    "langevin": (
        "src.baselines.langevin_unlearn",
        "NoisyGradientUnlearning",
        {"retain_lambda": 1.0, "variant": "langevin"},
    ),

    # ── DurableUn variants ────────────────────────────────────────────────────
    "durableun_saf_v3": (
        "src.durableun.saf",
        "SAF",
        {"alpha_quant": 1.0, "warmup_steps": 100, "retain_lambda": 2.0},
    ),
    "durableun_saf_alpha3": (
        "src.durableun.saf",
        "SAF",
        {"alpha_quant": 3.0, "warmup_steps": 100, "retain_lambda": 4.0},
    ),
}

BASELINE_NAMES = list(_REGISTRY.keys())

# Human-readable names for paper tables
DISPLAY_NAMES = {
    "ga":                   "GA",
    "npo":                  "NPO",
    "scrub":                "SCRUB",
    "salun":                "SalUn",
    "rmu":                  "RMU",
    "alpha_edit":           "AlphaEdit",
    "graddiff":             "GradDiff",
    "wga":                  "WGA",
    "wga_lp":               "WGA-LP",
    "tv":                   "TaskVec",
    "dare":                 "DARE",
    "noisy_ga":             "NoisyGA",
    "langevin":             "Langevin",
    "durableun_saf_v3":    "DurableUn-SAF v3",
    "durableun_saf_alpha3": "DurableUn-SAF α=3",
}


def get_baseline(
    name: str,
    model: nn.Module,
    forget_loader=None,
    retain_loader=None,
    device=None,
    n_steps: int = 300,
    lr: float = 5e-5,
    gradient_clip: float = 1.0,
    log_every: int = 50,
    **override_kwargs,
):
    """
    Instantiate a baseline unlearner by name.

    Args:
        name:           Method name from BASELINE_NAMES.
        model:          Model to unlearn.
        forget_loader:  Forget set DataLoader.
        retain_loader:  Retain set DataLoader.
        **override_kwargs: Override any default method kwargs.

    Returns:
        Instantiated unlearner (call .unlearn() to run).

    Example:
        unlearner = get_baseline("salun", model=model, forget_loader=fl,
                                  retain_loader=rl, device=device)
        result = unlearner.unlearn()
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown baseline: '{name}'. Available: {BASELINE_NAMES}"
        )

    module_path, class_name, defaults = _REGISTRY[name]

    # Dynamic import
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not load {class_name} from {module_path}: {e}\n"
            f"Make sure all baseline files are in src/baselines/"
        )

    # Merge defaults + overrides
    kwargs = {**defaults, **override_kwargs}

    return cls(
        model         = model,
        forget_loader = forget_loader,
        retain_loader = retain_loader,
        device        = device,
        n_steps       = n_steps,
        lr            = lr,
        gradient_clip = gradient_clip,
        log_every     = log_every,
        **kwargs,
    )


def list_baselines() -> None:
    """Print all available baselines."""
    print("\nAvailable baselines:")
    print(f"  {'Name':<25} {'Display':<22} {'Module'}")
    print("  " + "-" * 70)
    for name, (module, cls, _) in _REGISTRY.items():
        display = DISPLAY_NAMES.get(name, name)
        print(f"  {name:<25} {display:<22} {module}.{cls}")
    print()
