# DurableUn v2 — Certified Recovery-Resistant Machine Unlearning

## Step 0 — Set your HuggingFace token (ONE place only)

Edit `hf_token.py` in the project root:

```python
HF_TOKEN = "hf_PASTE_YOUR_TOKEN_HERE"   # ← replace this
```

That's the ONLY file you touch for authentication. The model loader reads it automatically.

---

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Quick sanity check (~20-40 minutes)

Run this FIRST. It uses INT4 loading and only 10 training steps per method.  
If it completes without error, your setup is correct. Then run the full version.

```bash
python experiments/phase0_baseline_audit.py --config configs/quick_config.yaml
```

Expected output: `results/baseline_recovery_YYYY-MM-DD_HH-MM-SS.csv`  
Expected time: ~3-5 minutes per method × 6 methods = ~20-40 minutes total

---

## Step 3 — Full run (~4-6 hours on RTX 4090)

```bash
python experiments/phase0_baseline_audit.py --config configs/base_config.yaml
```

---

## Step 4 — Resume after a crash

Checkpoints are saved after every method. If it crashes, just add `--resume`:

```bash
python experiments/phase0_baseline_audit.py --config configs/base_config.yaml --resume
```

It will skip methods that already have checkpoints and continue from where it stopped.

---

## Step 5 — Run a single method

```bash
python experiments/phase0_baseline_audit.py --config configs/base_config.yaml --methods ga
python experiments/phase0_baseline_audit.py --config configs/base_config.yaml --methods rmu
```

---

## Project structure

```
durableun_v2/
├── hf_token.py              ← YOUR TOKEN GOES HERE (only file to edit)
├── requirements.txt
├── configs/
│   ├── base_config.yaml     ← Full run (300 steps, full eval)
│   └── quick_config.yaml    ← Sanity check (10 steps, skip FT attack)
├── src/
│   ├── utils/
│   │   ├── logging_utils.py ← Timestamps in every log line
│   │   └── checkpoint.py    ← Save/load/resume per method
│   ├── models/
│   │   └── model_utils.py   ← Reads hf_token.py, loads LoRA model
│   ├── baselines/
│   │   ├── base.py          ← BaseUnlearner with correct loss_fn + unlearn()
│   │   ├── gradient_ascent.py
│   │   ├── npo.py
│   │   ├── scrub.py
│   │   ├── salun.py
│   │   ├── rmu.py
│   │   └── alpha_edit.py
│   ├── data/
│   │   ├── tofu_dataset.py
│   │   └── data_utils.py
│   └── evaluation/
│       └── evaluator.py     ← Forget acc, retain acc, MIA, quant recovery, FT recovery
├── experiments/
│   └── phase0_baseline_audit.py   ← Main script
├── checkpoints/             ← Model saved here after each method
├── results/                 ← CSV written here incrementally
├── logs/                    ← Full log with timestamps saved here
└── figures/                 ← Figure 1 saved here
```

---

## Expected runtimes (RTX 4090, base_config.yaml)

| Method     | Steps | Est. Time |
|------------|-------|-----------|
| GA         | 300   | ~15 min   |
| NPO        | 300   | ~20 min   |
| SCRUB      | 300   | ~20 min   |
| SalUn      | 300   | ~20 min   |
| RMU        | 300   | ~25 min   |
| AlphaEdit  | 300   | ~30 min   |
| **Total**  |       | **~2-3 hr** |

(RMU and AlphaEdit had extra overhead due to hooks / SVD — but both should complete well under 1 hour on GPU, not 30 hours.)

---

## Why the previous run took 4 days

1. **RMU ran on CPU** — model device detection returned `meta`, fallback went to CPU → 30 hours for 300 steps
2. **`nn.CrossEntropyLoss()` wrong loss function** — called as `loss_fn(model, batch)` which threw TypeError, caught silently, so `call_unlearner` thought no method existed and re-ran the whole model load
3. **No resume** — a crash meant restarting from scratch
4. **Bad imports** in `evaluator.py` (`from durableun.qrs`) — caused the quantization eval to crash after RMU finally finished

All four bugs are fixed in v2.
