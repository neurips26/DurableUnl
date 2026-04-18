# DurableUn: INT4 Quantization as a Recovery Attack on Machine Unlearning

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.9.1-ee4c2c" alt="PyTorch">
  <img src="https://img.shields.io/badge/NeurIPS-2026-purple" alt="NeurIPS 2026">
  <img src="https://img.shields.io/badge/Model-LLaMA--3--8B-green" alt="LLaMA-3">
</p>

<p align="center">
  <b>NeurIPS 2026 Submission</b> &nbsp;|&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;|&nbsp;
  <a href="#results">Results</a> &nbsp;|&nbsp;
  <a href="#citation">Citation</a>
</p>

---

## TL;DR

> **Every existing machine unlearning method is evaluated at BF16. Every production LLM is deployed at INT4.**
> We show INT4 quantization silently restores forgotten content by **5–22×**.
> We introduce `DurableUn-SAF`, the first method with a stable INT4 durability certificate.

---

## The Problem

<p align="center">
  <img src="figures/fig1_overview.png" width="850" alt="System Overview">
</p>

**Standard pipeline (top):** A model is unlearned at BF16 (FA≈0 ✓), then deployed at INT4. Forgotten content is restored — the privacy guarantee is silently broken.

**DurableUn pipeline (bottom):** Our STE-based quantization-aware training produces a model certified at BF16, INT8, *and* INT4 simultaneously.

---

## Key Findings

<p align="center">
  <img src="figures/fig2_attack.png" width="820" alt="INT4 Attack Results">
</p>

| Finding | Result |
|---------|--------|
| **INT8 is harmless** | Q-INT8 = 0.000 across **all 7 methods**, every seed |
| **INT4 is a systematic attack** | Forget accuracy restored 5–22× in every method that successfully unlearned |
| **Best forgetting ≠ best robustness** | GradDiff (FA=0.008, best forgetter) has the **worst** INT4 recovery ratio (18.9×) |
| **RA-INT4 ≈ RA** | INT4 selectively re-exposes forgotten content — it does *not* simply undo all fine-tuning |

---

## The FA–RA–Q-INT4 Trilemma

<p align="center">
  <img src="figures/fig3_trilemma.png" width="820" alt="Trilemma and Pareto Frontier">
</p>

Our dense Pareto sweep (α ∈ {0, 1, 1.5, 2, 2.5, 3}) reveals a **sharp structural phase transition** between α=1 and α=1.5: once quantization pressure is sufficient for robustness, Q-INT4 drops below 0.05 — but RA collapses to ≈0.045 and **stays there regardless of further tuning**. No configuration simultaneously satisfies FA≤0.05, RA≥0.50, Q-INT4≤0.05.

---

## Multi-Seed Certificate Stability

<p align="center">
  <img src="figures/fig4_multiseed.png" width="820" alt="Multi-Seed Results">
</p>

| Method | Q-INT4 (mean±std) | Cert rate |
|--------|-------------------|-----------|
| SalUn (uniform HPs) | 0.100 ± 0.049 | **0/3** |
| SalUn (original HPs, lr=1e-4, 500 steps) | 0.052 ± 0.007 | **1/3** |
| **DurableUn-SAF α=3 (ours)** | **0.043 ± 0.002** | **3/3** |

SalUn's seed-42 result (Q-INT4=0.051) was an **optimistic outlier** — it fails the certificate 2 out of 3 seeds at its own published hyperparameters. DurableUn-SAF achieves cert 3/3 with **25× lower variance** than SalUn at uniform HPs.

---

## Main Results Table

| Method | FA↓ | RA↑ | Q-INT8↓ | Q-INT4↓ | RA-INT4↑ | Cert. |
|--------|-----|-----|---------|---------|----------|-------|
| GA | 0.028 | 0.521 | **0.000** | 0.262 | 0.540 | ✗ |
| NPO | 0.636† | 0.624 | **0.000** | 0.613† | 0.622 | ✗ |
| SCRUB | 0.037 | 0.526 | **0.000** | 0.212 | 0.524 | ✗ |
| SalUn | **0.011** | **0.541** | **0.000** | 0.051 | **0.521** | ✗ |
| RMU | 0.580† | 0.565 | **0.000** | 0.559† | 0.564 | ✗ |
| AlphaEdit | 0.575† | 0.558 | **0.000** | 0.555† | 0.558 | ✗ |
| GradDiff | **0.008** | 0.510 | **0.000** | 0.151 | 0.538 | ✗ |
| **DurableUn-SAF α=3 (ours)** | 0.040 | 0.045 | **0.000** | **0.044** | 0.047 | **✓** |

†: Method never achieved meaningful unlearning (FA >> 0.05); Q-INT4 reflects pre-unlearning distribution.

---

## Method: DurableUn-SAF

DurableUn-SAF extends gradient ascent with a **Straight-Through Estimator (STE)** quantization-aware loss:

```
L_SAF(θ) = -L_forget(θ)                          # Standard GA
           - α(t) · L_forget(Q_STE(θ))            # Quantization-aware term
           + λ · L_retain(θ)                       # Retain preservation
```

**Key design choices:**
- **Full-model STE** — applied to all 4.5B linear layer parameters, not just LoRA adapters (14M). LoRA-only STE gives Q-INT4=0.169, insufficient for certification.
- **Warmup schedule** — steps 1–100 are pure GA; STE term ramps in at step 101. Without warmup, FA stays at 0.290 at step 50 vs 0.120 with warmup.
- **Retain balance** — λ = max(1, α+1) scales with α to compensate for increased forget pressure.

---

## Quick Start

### Prerequisites

```bash
git clone https://github.com/[anonymous]/DurableUn.git
cd DurableUn
pip install -r requirements.txt
```

Set your HuggingFace token (required for LLaMA-3 access):
```python
# hf_token.py — replace with your token
HF_TOKEN = "hf_PASTE_YOUR_TOKEN_HERE"
```

Get your token at: https://huggingface.co/settings/tokens

### Sanity check (~20 min)

```bash
python experiments/phase0_baseline_audit.py --config configs/quick_config.yaml
```

### Reproduce Table 1 (all 7 baselines, ~4–6 hours on RTX 4090)

```bash
py run.py baseline --datasets tofu --methods ga npo scrub salun rmu alpha_edit graddiff
```

### Run DurableUn-SAF

```bash
# α=3 — grants the certificate (Q-INT4=0.044, cert=Y)
py run.py saf --alpha 3.0 --seed 42

# α=1 — better retain utility trade-off (RA=0.317, Q-INT4=0.060)
py run.py saf --alpha 1.0 --seed 42
```

### Dense Pareto sweep (reproduces Table 2)

```bash
py -m experiments.revision_alpha_sweep --skip_salun
```

### Multi-seed validation (reproduces Table 3)

```bash
py -m experiments.revision_multiseed
```

### Real PTQ validation (Appendix D)

```bash
py -m experiments.revision_realquant_eval
```

### Compute durability certificate

```bash
py run.py certificate --checkpoint checkpoints/saf_alpha3p0_tofu_s42
```

---

## Installation

**Pinned versions for exact reproduction:**

```
torch==2.9.1+cu126
transformers==5.2.0
peft==0.18.1
bitsandbytes==0.49.2
datasets>=2.20.0
accelerate>=0.31.0
```

Full `requirements.txt` included. Hardware requirement: NVIDIA GPU with ≥24 GB VRAM (tested on RTX 4090).

---

## Project Structure

```
DurableUn/
├── run.py                          ← Master script (baseline + SAF + certificate)
├── STEPS.md                        ← Step-by-step reproduction guide
├── compute_certificate.py          ← Standalone certificate verifier
├── requirements.txt                ← Pinned dependencies
│
├── configs/
│   ├── base_config.yaml            ← Full run (300 steps, full eval)
│   ├── durableun_config.yaml       ← SAF-specific config
│   └── quick_config.yaml           ← Sanity check (10 steps)
│
├── src/
│   ├── baselines/                  ← GA, NPO, SCRUB, SalUn, RMU, AlphaEdit, GradDiff
│   ├── durableun/
│   │   └── saf.py                  ← DurableUn-SAF v4 (STE full-model)
│   ├── data/
│   │   ├── tofu_dataset.py         ← TOFU forget10/retain90
│   │   ├── muse_dataset.py         ← MUSE-News
│   │   └── wpu_dataset.py          ← WikiBio Person Unlearning (self-contained)
│   ├── evaluation/
│   │   ├── evaluator.py            ← FA, RA, Q-INTk, MIA-AUC
│   │   └── evaluator_additions.py  ← RA-INT4, RA-INT8
│   └── models/
│       └── model_utils.py          ← NF4 + LoRA loader
│
├── experiments/
│   ├── revision_alpha_sweep.py     ← Dense α sweep + SalUn original HPs
│   ├── revision_multiseed.py       ← Multi-seed SAF + SalUn
│   ├── revision_realquant_eval.py  ← Real bitsandbytes PTQ (merge+reload)
│   └── revision_second_arch.py     ← Mistral-7B architecture validation
│
├── figures/                        ← Paper figures (PNG + PDF)
├── results/                        ← Experiment CSVs (all runs)
└── paper/
    ├── neurips2026_durableun.tex   ← LaTeX source
    └── durableun.bib               ← References
```

---

## Reproducing All Paper Numbers

Every number in the paper is traceable to a specific command:

```bash
# Table 1 (7 baselines)
py run.py baseline --datasets tofu --methods ga npo scrub salun rmu alpha_edit graddiff

# Table 2 (Pareto sweep)
py run.py saf --alpha 0.0; py run.py saf --alpha 1.0
py run.py saf --alpha 1.5; py run.py saf --alpha 2.0
py run.py saf --alpha 2.5; py run.py saf --alpha 3.0

# Table 3 (multi-seed)
py -m experiments.revision_multiseed

# Table 4 (SalUn original HPs)
py -m experiments.revision_alpha_sweep --skip_saf

# Table 5 (Mistral-7B)
py -m experiments.revision_second_arch --methods ga

# Certificates
py run.py certificate --checkpoint checkpoints/saf_alpha3p0_tofu_s42

# Appendix D (real PTQ)
py -m experiments.revision_realquant_eval
```

---

## Dataset Croissant RAI Metadata

This repository contains **model evaluation scores** produced by running unlearning experiments on the [TOFU](https://huggingface.co/datasets/locuslab/TOFU) benchmark. It does not contain human-annotated data.

```json
{
  "@type": "sc:Dataset",
  "name": "DurableUn Evaluation Results",
  "description": "Model evaluation scores (FA, RA, Q-INT4) from machine unlearning experiments on TOFU/LLaMA-3-8B.",
  "license": "https://opensource.org/licenses/MIT",
  "annotationsPerItem": "N/A — benchmark contains model evaluation scores, not human annotations.",
  "annotatorDemographics": "N/A — benchmark contains model evaluation scores, not human annotations.",
  "recordSet": [
    {
      "name": "results",
      "description": "CSV files in results/ directory containing per-method evaluation metrics.",
      "field": [
        {"name": "method",     "dataType": "sc:Text",  "description": "Unlearning method name"},
        {"name": "forget_acc", "dataType": "sc:Float", "description": "Token-level forget accuracy (FA)"},
        {"name": "retain_acc", "dataType": "sc:Float", "description": "Token-level retain accuracy (RA)"},
        {"name": "quant_int4", "dataType": "sc:Float", "description": "Forget accuracy after INT4 quantization (Q-INT4)"},
        {"name": "quant_int8", "dataType": "sc:Float", "description": "Forget accuracy after INT8 quantization (Q-INT8)"},
        {"name": "cert",       "dataType": "sc:Text",  "description": "Y/N — empirical (0.05,{BF16,INT8,INT4})-durability certificate"},
        {"name": "seed",       "dataType": "sc:Integer","description": "Random seed"},
        {"name": "dataset",    "dataType": "sc:Text",  "description": "Evaluation dataset (tofu/muse_news/wpu)"}
      ]
    }
  ]
}
```

---

## Runtime Reference (RTX 4090)

| Method | Steps | Runtime | Peak VRAM |
|--------|-------|---------|-----------|
| Task Vector / DARE | — | < 1 min | 18.97 GB |
| AlphaEdit | 300 | 3 min | 18.97 GB |
| GA | 300 | 8 min | 11.20 GB |
| NPO / SCRUB | 300 | 22 min | 18.97 GB |
| GradDiff | 300 | 12 min | 18.97 GB |
| SalUn (uniform HPs) | 300 | 20 min | 22.57 GB |
| SalUn (original HPs) | 500 | 35 min | 22.57 GB |
| **DurableUn-SAF α=1** | 300 | **9 min** | 19.02 GB |
| **DurableUn-SAF α=3** | 300 | **37 min** | 19.02 GB |
| RMU | 300 | ~658 min | 18.97 GB |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The TOFU dataset is MIT licensed. LLaMA-3-8B-Instruct is subject to the [Meta Research License](https://llama.meta.com/llama3/license/).

---

## Citation

```bibtex
@inproceedings{durableun2026,
  title     = {DurableUn: INT4 Quantization as a Recovery Attack on Machine Unlearning,
               the FA--RA--Robustness Trilemma, and Sharpness-Aware Forgetting},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {Anonymous submission}
}
```

---

## Acknowledgements

Built on [TOFU](https://github.com/locuslab/TOFU), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), [PEFT](https://github.com/huggingface/peft), and [Transformers](https://github.com/huggingface/transformers).
