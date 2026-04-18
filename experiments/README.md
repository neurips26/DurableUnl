# DurableUn: Quantization-Resistant Machine Unlearning

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Official implementation of **"DurableUn: Quantization-Resistant Machine Unlearning via Sharpness-Aware Forgetting and the FA–RA–Robustness Trilemma"** (NeurIPS 2026).

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Key dependencies (pinned versions used in experiments):
- Python 3.13
- torch==2.9.1+cu126
- transformers==5.2.0
- peft==0.18.1
- bitsandbytes==0.49.2
- datasets==4.4.1

## 🔑 HuggingFace Token Setup

Edit `hf_token.py` (one file only):
```python
HF_TOKEN = "hf_your_token_here"
```

LLaMA-3-8B-Instruct requires a HuggingFace access request at:
https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

## 🚀 Reproducing Paper Results

### Table 1: All Baseline Results (Phase 0)
```bash
python experiments/phase0_baseline_audit.py --config configs/base_config.yaml
```
Expected runtime: ~3 hours on RTX 4090. Results saved to `results/baseline_recovery_*.csv`.

### Table 2: DurableUn-SAF Pareto Sweep
```bash
# All three alpha values (reproduces full Table 2, ~3.5 hours)
python experiments/pareto_sweep.py \
    --config configs/durableun_config.yaml \
    --alphas 0.0 1.0 3.0 \
    --n_steps 300

# Single best result: alpha=3, lambda=4 (~45 min)
python experiments/pareto_sweep.py \
    --config configs/durableun_config.yaml \
    --alphas 3.0 \
    --n_steps 300
```

### Table 3: DurableUn-SAF v3 (FA=0.008, best forget accuracy)
```bash
python experiments/run_durableun.py \
    --config configs/durableun_config.yaml \
    --phase saf
```
This uses alpha=1.0, warmup=100, retain_lambda=2.0 from `durableun_config.yaml`.

### Durability Certificate (Theorem 1)
```bash
python compute_certificate.py \
    --checkpoint checkpoints/saf_alpha_3p0 \
    --epsilon 0.05
```

### Figures
```bash
python experiments/generate_figures.py
```
All figures saved to `figures/` as both `.png` and `.pdf`.

## 📊 Pre-trained Checkpoints

Checkpoints for all experiments are hosted on HuggingFace Hub:
*(De-anonymized link provided upon acceptance)*

For anonymous review, checkpoints are included in the supplementary ZIP.

## 📁 Repository Structure

```
durableun_v2/
├── hf_token.py                    ← YOUR TOKEN HERE (only file to edit)
├── requirements.txt
├── configs/
│   ├── base_config.yaml           ← Phase 0 baseline experiments
│   └── durableun_config.yaml      ← DurableUn-SAF experiments
├── src/
│   ├── baselines/                 ← GA, NPO, SCRUB, SalUn, RMU, AlphaEdit
│   ├── durableun/                 ← SAF, OWD, QRS implementations
│   ├── evaluation/                ← Metrics: FA, RA, Q-INT4, FT attack, MIA
│   └── theory/                   ← Durability certificate
├── experiments/
│   ├── phase0_baseline_audit.py   ← Reproduce Table 1
│   ├── run_durableun.py           ← Train DurableUn-SAF
│   ├── pareto_sweep.py            ← Reproduce Table 2
│   └── generate_figures.py        ← Reproduce all figures
└── compute_certificate.py         ← Reproduce Theorem 1
```

## 📈 Main Results (Table 1)

| Method | FA↓ | RA↑ | Q-INT8↓ | Q-INT4↓ | FT@50↓ | Certificate |
|--------|-----|-----|---------|---------|--------|-------------|
| GA | 0.028 | 0.521 | 0.000 | 0.262 | 0.436 | ✗ |
| NPO | 0.636 | 0.624 | 0.000 | 0.613 | 0.615 | ✗ |
| SCRUB | 0.037 | 0.526 | 0.000 | 0.212 | 0.113 | ✗ |
| SalUn | 0.011 | 0.541 | 0.000 | 0.051 | 0.222 | ✗ |
| RMU | 0.580 | 0.565 | 0.000 | 0.559 | — | ✗ |
| AlphaEdit | 0.575 | 0.558 | 0.000 | 0.555 | — | ✗ |
| **DurableUn-SAF v3** (α=1) | **0.008** | 0.495 | 0.000 | 0.239 | 0.000 | ✗ |
| **DurableUn-SAF** (α=3) | 0.040 | 0.045 | 0.000 | **0.044** | **0.000** | **✓** |

*FA = Forget Accuracy, RA = Retain Accuracy, Q-INT4 = forget accuracy after INT4 quantization, FT@50 = forget accuracy after 50 fine-tuning steps*

## 📜 Citation

```bibtex
@inproceedings{anonymous2026durableun,
  title={DurableUn: Quantization-Resistant Machine Unlearning via 
         Sharpness-Aware Forgetting and the FA--RA--Robustness Trilemma},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
