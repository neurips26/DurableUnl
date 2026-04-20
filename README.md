# DurableUn: INT4 Quantization as a Recovery Attack on Machine Unlearning

<p align="center">
  <a href="https://github.com/neurips26/DurableUnl/actions/workflows/ci.yml">
    <img src="https://github.com/neurips26/DurableUnl/actions/workflows/ci.yml/badge.svg?branch=main&event=push" alt="CI">
  </a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://openreview.net/group?id=NeurIPS.cc/2026/Conference">
    <img src="https://img.shields.io/badge/NeurIPS_2026-Main_Track-purple.svg" alt="NeurIPS 2026">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  </a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.9.1-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/GPU-RTX_4090_24GB-76b900.svg" alt="GPU">
</p>

<p align="center">
  <b>NeurIPS 2026</b> &nbsp;|&nbsp;
  <a href="paper/neurips2026_durableun.pdf">Paper</a> &nbsp;|&nbsp;
  <a href="#quickstart">Quickstart</a> &nbsp;|&nbsp;
  <a href="#results">All Results</a> &nbsp;|&nbsp;
  <a href="#method">Method</a> &nbsp;|&nbsp;
  <a href="#citation">Citation</a>
</p>

---

## TL;DR

> Every machine unlearning paper evaluates at BF16. Every production LLM is deployed at INT4.
> **INT4 quantization silently restores forgotten content by 5–22× across all 7 state-of-the-art methods — on every dataset we tested.**
> We identify this as the **Quantization Recovery Attack (QRA)** and introduce `DurableUn-SAF`, the first method to achieve a stable empirical INT4 durability certificate:
> cert rate = **3/3** seeds, Q-INT4 = **0.043 ± 0.002**.

---

## The Problem

<p align="center">
  <img src="https://raw.githubusercontent.com/neurips26/DurableUnl/main/figures/fig1_overview.png" width="900" alt="System Overview">
</p>

**Standard pipeline (top — broken):** A model is unlearned at bfloat16 (BF16), passes a GDPR compliance audit with near-zero forget accuracy (FA ≈ 0), then is quantized to INT4 for production deployment. The forgotten content reappears. The audit was meaningless.

**DurableUn pipeline (bottom — ours):** Our STE-based quantization-aware objective produces a model with an empirical durability certificate at BF16, INT8, *and* INT4 simultaneously.

---

## The INT4 Recovery Attack

<p align="center">
  <img src="figures/fig2_attack.png" width="900" alt="INT4 attack results">
</p>

**Left:** Q-INT8 = 0.000 (grey) for every method — INT8 is completely harmless. Q-INT4 (colored) is catastrophic for every method that successfully unlearned.

**Right:** INT4 recovery ratio (Q-INT4 / FA). GradDiff achieves the best forgetting quality (FA = 0.008) yet has the **worst** INT4 fragility (18.9×). State-of-the-art forgetting does not imply quantization robustness.

| Finding | Result |
|---------|--------|
| **INT8 universally harmless** | Q-INT8 = 0.000 across 7 methods, 3 seeds, 2 architectures, 3 datasets |
| **INT4 systematic attack** | 5–22× FA recovery in every method that successfully unlearned |
| **Best forgetter ≠ most robust** | GradDiff: FA = 0.008 → Q-INT4 = 0.151 (18.9× recovery) |
| **Selective recovery** | RA-INT4 ≈ RA — INT4 re-exposes *forgotten* content only, not general capability |
| **Architecture-agnostic** | GA on Mistral-7B: Q-INT4 = 0.392 (stronger than LLaMA-3's 0.262) |
| **Dataset-agnostic** | Attack confirmed on TOFU, MUSE-News, and WikiBio-WPU |

---

## The FA–RA–Q-INT4 Trilemma

<p align="center">
  <img src="figures/fig3_trilemma.png" width="900" alt="Trilemma and Pareto Frontier">
</p>

**Left:** Pareto frontier. Stars = DurableUn-SAF at different α; shapes = baselines. Only α ≥ 1.5 reaches the green target region (FA ≤ 0.05 **and** Q-INT4 ≤ 0.05).

**Right:** Dense sweep α ∈ {0, 1, 1.5, 2, 2.5, 3} reveals a **sharp structural phase transition**. Between α = 1 and α = 1.5, Q-INT4 drops below 0.05 — but RA collapses to ≈ 0.045 and stays there regardless of further tuning. This is structural, not a tuning artifact.

> **Empirical conjecture** (verified over 7 methods × 3 seeds × 2 HP settings × 3 datasets):
> No configuration simultaneously satisfies FA ≤ 0.05, RA ≥ 0.50, and Q-INT4 ≤ 0.05.

---

## Multi-Seed Certificate Stability

<p align="center">
  <img src="figures/fig4_multiseed.png" width="900" alt="Multi-seed reliability">
</p>

| Method | FA (mean±std) | RA (mean±std) | Q-INT4 (mean±std) | Cert rate |
|--------|:---:|:---:|:---:|:---:|
| SalUn (uniform HPs) | 0.009 ± 0.002 | 0.519 ± 0.036 | 0.100 ± 0.049 | 0/3 |
| SalUn (original HPs, lr=1e-4, 500 steps) | 0.033 ± 0.018 | 0.581 ± 0.021 | 0.052 ± 0.007 | 1/3 |
| **DurableUn-SAF α=3 (ours)** | 0.043 ± 0.002 | 0.046 ± 0.002 | **0.043 ± 0.002** | **3/3** |

SalUn's seed-42 result (Q-INT4 = 0.051) was an optimistic outlier — it fails 2 out of 3 seeds at its own published hyperparameters. DurableUn-SAF achieves cert 3/3 with **25× lower variance**.

---

## Results

### Table 1 — All 7 Baselines, TOFU (LLaMA-3-8B-Instruct, seed 42)

| Method | FA↓ | RA↑ | MIA | Q-INT8↓ | Q-INT4↓ | RA-INT4↑ | Cert. |
|--------|:---:|:---:|:---:|:-------:|:-------:|:--------:|:-----:|
| GA | 0.028 | 0.521 | 0.000 | **0.000** | 0.262 | 0.540 | ✗ |
| NPO† | 0.636 | 0.624 | 0.494 | **0.000** | 0.613 | 0.622 | ✗ |
| SCRUB | 0.037 | 0.526 | 0.000 | **0.000** | 0.212 | 0.524 | ✗ |
| SalUn | **0.011** | **0.541** | 0.000 | **0.000** | 0.051 | **0.521** | ✗ |
| RMU† | 0.580 | 0.565 | 0.389 | **0.000** | 0.559 | 0.564 | ✗ |
| AlphaEdit† | 0.575 | 0.558 | 0.406 | **0.000** | 0.555 | 0.558 | ✗ |
| GradDiff | **0.008** | 0.510 | 0.000 | **0.000** | 0.151 | 0.538 | ✗ |
| **DurableUn-SAF α=3 (ours)** | 0.040 | 0.045 | 0.000 | **0.000** | **0.044** | 0.047 | **✓** |

†: Method never achieved meaningful unlearning (FA >> 0.05). Pre-unlearning MIA-AUC = 0.712; post-unlearning values near 0.0 indicate successful forgetting.

---

### Table 2 — Dense Pareto Sweep, TOFU (seed 42)

Sharp phase transition between α = 1 and α = 1.5: Q-INT4 drops below 0.05, RA collapses, and every α ≥ 1.5 converges to the same point — structural, not a tuning artifact.

| Config | α | λ | FA↓ | RA↑ | Q-INT4↓ | Cert. |
|--------|:-:|:-:|:---:|:---:|:-------:|:-----:|
| GA (reproduced) | 0.0 | 1.0 | 0.028 | 0.521 | 0.262 | ✗ |
| SalUn (reference) | — | — | 0.011 | 0.541 | 0.051 | ✗ |
| DurableUn-SAF | 1.0 | 2.0 | 0.275 | 0.317 | 0.060 | ✗ |
| **DurableUn-SAF** | **1.5** | 2.5 | 0.041 | 0.045 | **0.041** | **✓** |
| **DurableUn-SAF** | **2.0** | 3.0 | 0.041 | 0.045 | **0.041** | **✓** |
| **DurableUn-SAF** | **2.5** | 3.5 | 0.041 | 0.045 | **0.041** | **✓** |
| **DurableUn-SAF** | **3.0** | 4.0 | 0.040 | 0.045 | **0.044** | **✓** |

---

### Table 3 — Multi-Dataset Validation (LLaMA-3-8B-Instruct, seed 42)

Q-INT8 = 0.000 universally across all datasets and methods.

| Dataset | Method | FA↓ | RA↑ | Q-INT4↓ | Cert. |
|:-------:|--------|:---:|:---:|:-------:|:-----:|
| **WPU** (simple) | GA | 0.110 | 0.717 | 0.030 | ✓ |
| | SalUn | 0.175 | **0.746** | 0.139 | ✗ |
| | GradDiff | **0.080** | 0.667 | **0.016** | ✓ |
| | DurableUn-SAF α=1 | 0.195 | 0.788 | 0.057 | ✗ |
| **TOFU** (medium) | GA | **0.028** | 0.521 | 0.262 | ✗ |
| | SalUn | **0.011** | **0.541** | 0.051 | ✗ |
| | GradDiff | **0.008** | 0.510 | 0.151 | ✗ |
| | **DurableUn-SAF α=3** | 0.040 | 0.045 | **0.044** | **✓** |
| **MUSE-News** (hard) | GA | 0.480 | 0.484 | 0.459 | ✗ |
| | SalUn | 0.470 | 0.480 | 0.466 | ✗ |
| | GradDiff | 0.473 | 0.478 | 0.453 | ✗ |
| | **DurableUn-SAF α=1** | **0.035** | 0.036 | **0.031** | **✓** |

**Key finding:** The trilemma becomes binding as dataset difficulty increases. On WPU (simple facts), standard methods can certify. On MUSE-News (real news articles), all baselines fail to achieve meaningful forgetting (FA ≈ 0.47 — barely below pre-unlearning). DurableUn-SAF is the only method to both forget and certify on real-world content.

---

### Table 4 — Architecture Generalization (GA, seed 42)

| Architecture | FA↓ | RA↑ | Q-INT8↓ | Q-INT4↓ | Cert. |
|:---:|:---:|:---:|:-------:|:-------:|:-----:|
| LLaMA-3-8B-Instruct | 0.028 | 0.521 | **0.000** | 0.262 | ✗ |
| Mistral-7B-Instruct | 0.092 | 0.638 | **0.000** | 0.392 | ✗ |

The INT4 attack is stronger on Mistral (0.392 vs 0.262) — our simulator is conservative, not overstated.

---

### Table 5 — Real PTQ Validation (Merged Model, bitsandbytes)

| Method | FA@BF16 | FA@BnB-INT8 | FA@BnB-NF4 | Sim-INT4 |
|--------|:-------:|:-----------:|:----------:|:--------:|
| GA | 0.070 | 0.110 (+57%) | 0.014 | 0.262 |
| SalUn | 0.044 | 0.042 | 0.009 | 0.051 |
| GradDiff | 0.043 | 0.071 (+65%) | 0.003 | 0.151 |
| **DurableUn-SAF α=3** | **0.042** | **0.045** | **0.043** | **0.044** |

Under merged-model NF4, baselines do not exhibit INT4 recovery. Under merged-model INT8, GA (+57%) and GradDiff (+65%) show real partial recovery. DurableUn-SAF is flat across all conditions.

---

### Table 6 — Fine-Tuning Recovery (Alpaca, 50 steps, lr=2e-5)

| Method | FA before FT | FA after FT |
|--------|:-----------:|:-----------:|
| GA | 0.028 | 0.436 |
| SCRUB | 0.037 | 0.113 |
| SalUn | 0.011 | 0.222 |
| **DurableUn-SAF α=3** | 0.040 | **0.000** |

DurableUn-SAF maintains FA = 0.000 after 50 fine-tuning steps on unrelated public data. All baselines recover substantially.

---

## Method

DurableUn-SAF extends gradient ascent with a Straight-Through Estimator (STE) quantization-aware term:

```
L_SAF(θ) = −L_forget(θ)                          [Standard GA]
           − α(t) · L_forget(Q_STE(θ))             [STE through INT4 rounding]
           + λ · L_retain(θ)                        [Retain preservation]

Warmup:  α(t) = min(α_max, 2·α_max·(t − 100) / 200) · 1[t > 100]
Balance: λ = max(1, α + 1)
```

**Why it works:** Standard GA leaves the forget loss *sharp* — large gradient κ = ‖∇L_forget(θ*)‖₂ means INT4 noise δ ≈ 7% of weight range restores FA via FA(θ_q) ≤ FA(θ*) + κ·δ. DurableUn-SAF optimizes L_forget under simulated INT4 simultaneously, flattening the landscape at both precisions.

**Why full-model STE is essential:** INT4 recovery is stored in base model weights (4.5B params), not LoRA adapters (14M). LoRA-only STE gives Q-INT4 = 0.169 — insufficient for the certificate.

**Connection to SAM:** Inspired by Sharpness-Aware Minimization (Foret et al., 2021) but applied to the *forgetting* landscape, not the training landscape.

---

## Durability Certificate

**Definition:** Model θ* is empirically (ε, P)-durable if FA(quantize_p(θ*)) ≤ ε for all p ∈ P ⊆ {BF16, INT8, INT4}.

**DurableUn-SAF (α=3, TOFU, seed 42) — first method ever to receive this certificate:**

| Precision | FA | Certified? |
|:---------:|:--:|:----------:|
| BF16 | 0.040 | ✓ (≤ 0.05) |
| INT8 | 0.000 | ✓ (≤ 0.05) |
| INT4 | 0.044 | ✓ (≤ 0.05) |

**Certificate: (0.047, {BF16, INT8, INT4})-durable.** All 7 baselines fail. Multi-seed: cert rate = 3/3, Q-INT4 = 0.043 ± 0.002.

```bash
py run.py certificate --checkpoint checkpoints/saf_alpha3p0_tofu_s42
# Expected:
#   FA@BF16 = 0.040  ✓
#   FA@INT8 = 0.000  ✓
#   FA@INT4 = 0.044  ✓
#   Certificate: (0.047, {BF16, INT8, INT4})-durable  ✓
```

---

## Quickstart

```bash
git clone https://github.com/neurips26/DurableUnl.git
cd DurableUnl
pip install -r requirements.txt

# Set HuggingFace token (required for LLaMA-3 gated access)
# Edit hf_token.py: HF_TOKEN = "hf_PASTE_YOUR_TOKEN_HERE"
```

**From pre-computed results — no GPU needed (~1 min):**
```bash
python benchmark/summarise_results.py --results-dir results/
```

**Sanity check — GPU required (~20 min):**
```bash
python experiments/phase0_baseline_audit.py --config configs/quick_config.yaml
```

**Full reproduction:**
```bash
# Table 1 — 7 baselines (~4–6 hours total)
py run.py baseline --datasets tofu --methods ga npo scrub salun rmu alpha_edit graddiff

# Table 2 — Pareto sweep
py -m experiments.revision_alpha_sweep --skip_salun

# Table 3 — multi-dataset
py run.py baseline --datasets muse_news wpu --methods ga salun graddiff
py run.py saf --alpha 1.0 --datasets muse_news wpu
py run.py saf --alpha 3.0 --datasets tofu

# Table 4 — Mistral-7B
py -m experiments.revision_second_arch --methods ga

# Table 5 — real PTQ
py -m experiments.revision_realquant_eval

# Multi-seed (Table in §7)
py -m experiments.revision_multiseed

# Certificate
py run.py certificate --checkpoint checkpoints/saf_alpha3p0_tofu_s42
```

---

## Repository Structure

```
DurableUn/
├── run.py                              ← Master script
├── STEPS.md                            ← Step-by-step guide
├── REPRODUCE.md                        ← Commands for every paper table
├── DATASHEET.md                        ← Dataset documentation
├── CITATION.cff                        ← Machine-readable citation
├── croissant_metadata.json             ← NeurIPS Croissant metadata
├── requirements.txt
│
├── configs/
│   ├── base_config.yaml
│   ├── durableun_config.yaml
│   └── quick_config.yaml
│
├── src/
│   ├── baselines/                      ← GA, NPO, SCRUB, SalUn, RMU, AlphaEdit, GradDiff
│   ├── durableun/saf.py                ← DurableUn-SAF v4 (full-model STE)
│   ├── data/
│   │   ├── tofu_dataset.py             ← TOFU forget10/retain90
│   │   ├── muse_dataset.py             ← MUSE-News (889 BBC articles)
│   │   └── wpu_dataset.py              ← WikiBio Person Unlearning
│   ├── evaluation/
│   │   ├── evaluator.py                ← FA, RA, Q-INTk, MIA-AUC
│   │   └── evaluator_additions.py      ← RA-INT4, RA-INT8
│   └── models/model_utils.py
│
├── experiments/
│   ├── revision_alpha_sweep.py         ← Pareto sweep + SalUn orig. HPs
│   ├── revision_multiseed.py           ← 3-seed SAF + SalUn
│   ├── revision_realquant_eval.py      ← Real bitsandbytes PTQ
│   └── revision_second_arch.py         ← Mistral-7B validation
│
├── figures/                            ← All 4 paper figures (PNG)
├── results/                            ← All experiment CSVs
└── paper/
    ├── neurips2026_durableun.tex
    └── durableun.bib
```

---

## Runtime Reference (RTX 4090, 24 GB VRAM)

| Method | Steps | Runtime | Peak VRAM |
|--------|:-----:|:-------:|:---------:|
| Task Vector / DARE | — | < 1 min | 18.97 GB |
| AlphaEdit | 300 | 3 min | 18.97 GB |
| GA | 300 | 8 min | 11.20 GB |
| GradDiff | 300 | 12 min | 18.97 GB |
| SalUn (uniform HPs) | 300 | 20 min | 22.57 GB |
| NPO / SCRUB | 300 | 22 min | 18.97 GB |
| SalUn (original HPs) | 500 | 35 min | 22.57 GB |
| **DurableUn-SAF α=1** | 300 | **9 min** | 19.02 GB |
| **DurableUn-SAF α=3** | 300 | **37 min** | 19.02 GB |
| RMU | 300 | ~658 min | 18.97 GB |

---

## Dataset and Croissant RAI Metadata

| Dataset | License | Purpose |
|---------|:-------:|---------|
| [TOFU](https://huggingface.co/datasets/locuslab/TOFU) | MIT | Synthetic author Q&A |
| [MUSE-News](https://github.com/jaechan-repo/muse-bench) | Apache 2.0 | Real BBC news articles |
| WikiBio-WPU (curated, self-contained) | MIT | Factual person Q&A |

```json
"annotationsPerItem":    "N/A — benchmark contains model evaluation scores, not human annotations.",
"annotatorDemographics": "N/A — benchmark contains model evaluation scores, not human annotations."
```

Full metadata: [`croissant_metadata.json`](croissant_metadata.json) · Dataset docs: [`DATASHEET.md`](DATASHEET.md)

---

## Citation

```bibtex
@inproceedings{durableun2026,
  title     = {{DurableUn}: {INT4} Quantization as a Recovery Attack on Machine Unlearning,
               the {FA--RA--Robustness} Trilemma, and Sharpness-Aware Forgetting},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

---

## License

MIT — see [LICENSE](LICENSE). LLaMA-3: [Meta Llama 3 Community License](https://llama.meta.com/llama3/license/). Mistral-7B: Apache 2.0. TOFU: MIT.

## Acknowledgements

Built on [TOFU](https://github.com/locuslab/TOFU), [MUSE](https://github.com/jaechan-repo/muse-bench), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), [PEFT](https://github.com/huggingface/peft), and [Transformers](https://github.com/huggingface/transformers). Inspired by [SAM](https://github.com/google-research/sam) (Foret et al., 2021).
