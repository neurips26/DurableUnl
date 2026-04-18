# Reproduction Guide

This document provides exact commands to reproduce every number in the paper.
Every table row is traceable to a specific command and output CSV.

## Hardware

- GPU: NVIDIA RTX 4090 (24 GB VRAM)
- CUDA: 12.6
- OS: Tested on Windows 11 and Ubuntu 22.04

## Setup

```bash
git clone https://github.com/[anonymous]/DurableUn.git
cd DurableUn
pip install -r requirements.txt
# Edit hf_token.py with your HuggingFace token
```

---

## Table 1 — All 7 baselines under quantization (§4, main paper)

```bash
py run.py baseline --datasets tofu \
  --methods ga npo scrub salun rmu alpha_edit graddiff \
  --seed 42
```

Output: `results/baseline_YYYY-MM-DD_HH-MM-SS.csv`
Runtime: ~4–6 hours total (see runtime table in README)

---

## Table 2 — Pareto sweep α ∈ {0, 1, 1.5, 2, 2.5, 3} (§5, trilemma)

```bash
# Run all α values
py run.py saf --alpha 0.0 --seed 42   # reproduces GA exactly
py run.py saf --alpha 1.0 --seed 42
py run.py saf --alpha 1.5 --seed 42
py run.py saf --alpha 2.0 --seed 42
py run.py saf --alpha 2.5 --seed 42
py run.py saf --alpha 3.0 --seed 42

# Or use the sweep script
py -m experiments.revision_alpha_sweep --skip_salun --seed 42
```

---

## Table 3 — Multi-seed reliability (§6, certificate stability)

```bash
# SAF α=3 across seeds 42, 123, 5508
py run.py saf --alpha 3.0 --seed 42
py run.py saf --alpha 3.0 --seed 123
py run.py saf --alpha 3.0 --seed 5508

# SalUn across seeds 42, 123, 5508 (uniform HPs)
py run.py baseline --datasets tofu --methods salun --seed 123
py run.py baseline --datasets tofu --methods salun --seed 5508

# Generate summary table
py -m experiments.revision_multiseed
```

Output: `results/revision_multiseed_YYYY-MM-DD_HH-MM-SS.csv`

---

## Table 4 — SalUn original HPs (baseline tuning analysis, §6)

```bash
py -m experiments.revision_alpha_sweep --skip_saf --seed 42
py -m experiments.revision_alpha_sweep --skip_saf --seed 123
py -m experiments.revision_alpha_sweep --skip_saf --seed 5508
```

SalUn original HPs: lr=1e-4, 500 steps (Foster et al. 2024).

---

## Table 5 — Mistral-7B architecture validation (§7 / Limitations)

```bash
py -m experiments.revision_second_arch --methods ga --seed 42
```

Expected: FA=0.092, Q-INT8=0.000, Q-INT4=0.392, cert=N
Runtime: ~15 min (model download ~15 GB on first run)

---

## Appendix D — Real bitsandbytes PTQ (merged model)

```bash
py -m experiments.revision_realquant_eval
```

Pipeline: merge LoRA → save BF16 → reload with bitsandbytes INT8/NF4.
Requires ~16 GB free disk per checkpoint.

---

## Appendix E — Fine-tuning recovery attack

```bash
py run.py finetuning_attack --methods ga scrub salun saf_alpha3 --seed 42
```

---

## Certificate verification

```bash
py run.py certificate --checkpoint checkpoints/saf_alpha3p0_tofu_s42
```

Expected output:
```
FA@BF16 = 0.040  ✓
FA@INT8 = 0.000  ✓
FA@INT4 = 0.044  ✓
Certificate: (0.047, {BF16, INT8, INT4})-durable  ✓
```

---

## Pre-computed results

All CSVs in `results/` match the paper numbers exactly. To load:

```python
import pandas as pd
df = pd.read_csv('results/baseline_2026-03-31_13-55-53.csv')
print(df[['method', 'forget_acc', 'retain_acc', 'quant_int4', 'cert']])
```

---

## Verifying individual findings

**Finding 1 (INT8 harmless):** Check `quant_int8` column in any baseline CSV — all values are 0.000.

**Finding 2 (INT4 attack):** Check `quant_int4` column. For methods with `forget_acc < 0.05`, all values are > 0.05.

**Finding 3 (RA-INT4 ≈ RA):** Compare `retain_acc` and `ra_int4` columns — values are within ±0.05.

**Phase transition:** Plot Q-INT4 vs α from `revision_alpha_sweep_*.csv` — sharp drop between α=1 and α=1.5.

**Certificate stability:** Count `cert == "Y"` rows in `revision_multiseed_*.csv` — SAF: 3/3, SalUn uniform: 0/3, SalUn original: 1/3.
