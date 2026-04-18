# DurableUn — Complete Step-by-Step Guide
Everything you need. Uses `py` not `python`.

---

## Project Structure

```
durableun_v2/
│
├── run.py                          ← MASTER SCRIPT (use this for everything)
├── hf_token.py                     ← PUT YOUR HF TOKEN HERE
├── requirements.txt
├── STEPS.md                        ← this file
│
├── configs/
│   ├── base_config.yaml            ← for baselines
│   └── durableun_config.yaml       ← for DurableUn-SAF
│
├── src/
│   ├── data/
│   │   ├── tofu_dataset.py         ← TOFU loader
│   │   ├── muse_dataset.py         ← MUSE-News / MUSE-Books loader
│   │   ├── wpu_dataset.py          ← WikiBio Person Unlearning loader
│   │   ├── dataset_registry.py     ← unified get_dataloaders()
│   │   └── data_utils.py
│   │
│   ├── baselines/
│   │   ├── base.py                 ← BaseUnlearner class
│   │   ├── baseline_registry.py    ← get_baseline("salun", ...) 
│   │   ├── gradient_difference.py  ← GradDiff (new)
│   │   ├── wga.py                  ← WGA / WGA-LP (new)
│   │   ├── tv_distance.py          ← Task Vector / DARE (new)
│   │   ├── langevin_unlearn.py     ← Noisy GA / Langevin (new)
│   │   ├── scrub.py
│   │   └── rmu.py
│   │
│   ├── durableun/
│   │   └── saf.py                  ← DurableUn-SAF (v4, full-model STE)
│   │
│   ├── evaluation/
│   │   ├── evaluator.py            ← FA, RA, Q-INT4, MIA, FT attack
│   │   └── evaluator_additions.py  ← RA-INT4, compute_full_eval()
│   │
│   └── theory/
│       └── certificate.py          ← (ε,δ,P)-durability certificate
│
├── experiments/
│   ├── priority_audit.py           ← focused paper experiment matrix
│   ├── multi_seed_eval.py          ← 3-seed reliability
│   ├── generate_figures.py         ← all paper figures
│   ├── pareto_sweep.py
│   └── gptq_quantization_eval.py
│
├── checkpoints/                    ← saved after each run
├── results/                        ← CSV files
├── logs/                           ← log files
└── figures/                        ← paper figures
```

---

## Step 0 — One-time setup

### 0a. Update your HF token
Open `hf_token.py` and set:
```python
HF_TOKEN = "hf_YOUR_TOKEN_HERE"
```
Get a fresh token at: https://huggingface.co/settings/tokens

### 0b. Check everything works
```
py run.py preflight
```
This checks GPU, token, packages, and dataset connectivity. Fix any FAIL before proceeding.

---

## Step 1 — Training-free baselines (fast, ~2 min, confirms pipeline works)
```
py run.py baseline --datasets tofu --methods tv dare --skip_ft
```
Expected: FA~0.028 for TV (same as GA). These just negate LoRA weights.

---

## Step 2 — GradDiff on TOFU (~12 min)
```
py run.py baseline --datasets tofu --methods graddiff --skip_ft
```

---

## Step 3 — GA and SalUn on TOFU (~50 min total)
```
py run.py baseline --datasets tofu --methods ga salun --skip_ft
```

---

## Step 4 — DurableUn-SAF v3 on TOFU (~25 min, best FA result)
```
py run.py saf --alpha 1.0 --datasets tofu
```

---

## Step 5 — Pareto sweep on TOFU (alpha=0,1,3, ~4 hours total)
```
py run.py pareto --datasets tofu
```
Or run individual alphas:
```
py run.py saf --alpha 0.0 --datasets tofu
py run.py saf --alpha 1.0 --datasets tofu
py run.py saf --alpha 3.0 --datasets tofu
```

---

## Step 6 — Certificate (~15 min, after Step 5)
```
py run.py certificate --checkpoint checkpoints/saf_alpha_3p0_tofu_s42
```
Or try the old checkpoint:
```
py run.py certificate --checkpoint checkpoints/saf_alpha_3p0
```

---

## Step 7 — MUSE-News second dataset (~2 hours for main methods)
```
py run.py baseline --datasets muse_news --methods ga salun graddiff --skip_ft
py run.py saf --alpha 1.0 --datasets muse_news
```

---

## Step 8 — WikiBio third dataset (~2 hours for main methods)
```
py run.py baseline --datasets wpu --methods ga salun graddiff --skip_ft
py run.py saf --alpha 1.0 --datasets wpu
```

---

## Step 9 — Multi-seed reliability (~6 hours, for mean±std in paper)
```
py run.py seeds --methods ga salun durableun_saf_v3 --seeds 42 123 5508
```

---

## Step 10 — Generate all figures
```
py run.py figures
```

---

## Run everything overnight (Steps 1-10 automatically)
```
py run.py full
```
Skips any steps already completed. Safe to interrupt and resume.

---

## Resuming after a crash
Every command supports `--resume` which skips runs with existing `result.json`:
```
py run.py baseline --datasets tofu --methods ga salun --resume
py run.py multi_dataset --datasets tofu muse_news wpu --resume
```

---

## Paper experiment matrix (what goes in which table)

### Table 1 (Main results, Phase 0 baselines + DurableUn):
```
py run.py baseline --datasets tofu --methods ga npo scrub salun rmu alpha_edit --skip_ft
py run.py saf --alpha 1.0 --datasets tofu
py run.py saf --alpha 3.0 --datasets tofu
```

### Table 2 (Pareto sweep):
```
py run.py pareto --datasets tofu
```

### Table 3 (Multi-dataset generalization):
```
py run.py multi_dataset --datasets tofu muse_news wpu --methods ga salun graddiff durableun_saf_v3
```

### Appendix (Modern baselines):
```
py run.py baseline --datasets tofu --methods wga tv dare --skip_ft
```

### Certificate (Theorem 1):
```
py run.py certificate --checkpoint checkpoints/saf_alpha_3p0
```

---

## Complete method list

| Command name           | Paper name       | Training-free? | Time (300 steps) |
|------------------------|------------------|----------------|-----------------|
| ga                     | GA               | No             | ~8 min          |
| npo                    | NPO              | No             | ~22 min         |
| scrub                  | SCRUB            | No             | ~22 min         |
| salun                  | SalUn            | No             | ~25 min         |
| rmu                    | RMU              | No             | ~11 hours       |
| alpha_edit             | AlphaEdit        | No             | ~3 min          |
| graddiff               | GradDiff         | No             | ~12 min         |
| wga                    | WGA              | No             | ~12 min         |
| tv                     | Task Vector      | **YES**        | ~1 min          |
| dare                   | DARE             | **YES**        | ~1 min          |
| durableun_saf_v3       | DurableUn-SAF v3 | No             | ~25 min         |
| durableun_saf_alpha3   | DurableUn-SAF α=3| No             | ~350 min        |

## Complete dataset list

| Dataset name | Paper name              | HuggingFace ID           |
|-------------|-------------------------|--------------------------|
| tofu        | TOFU forget10           | locuslab/TOFU            |
| muse_news   | MUSE-News               | muse-bench/MUSE-News     |
| muse_books  | MUSE-Books              | muse-bench/MUSE-Books    |
| wpu         | WikiBio Person Unlearn  | wiki_bio                 |

---

## Why the current SalUn run is wrong
94 sec/step vs normal 1.14 sec/step = 82x slower.
Kill it with Ctrl+C. The issue was that the saliency mask was recomputing every step.
The new `run.py` calls SalUn correctly (mask computed once before training loop).

---

## Results location
- CSVs: `results/` 
- Checkpoints: `checkpoints/{method}_{dataset}_s{seed}/`
- Certificate: `checkpoints/{method}_{dataset}_s{seed}/certificate.json`
- Figures: `figures/`
- Logs: `logs/`
