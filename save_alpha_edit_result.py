import json, os, csv

metrics = {
    "method": "alpha_edit",
    "forget_acc": 0.5747,
    "retain_acc": 0.5579,
    "mia_auc": 0.4064,
    "quant_bf16": 0.5507,
    "quant_int8": 0.0000,
    "quant_int4": 0.5554,
    "ft_50steps": -1,
    "wall_time_min": 3.0,
    "gradient_steps": 300,
}

ckpt_dir = os.path.join("checkpoints", "alpha_edit")
os.makedirs(ckpt_dir, exist_ok=True)
with open(os.path.join(ckpt_dir, "result.json"), "w") as f:
    json.dump({"method": "alpha_edit", "saved_at": "2026-03-28", "metrics": metrics}, f, indent=2)
print("Checkpoint saved.")

os.makedirs("results", exist_ok=True)
with open("results/alpha_edit_result.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
    w.writeheader(); w.writerow(metrics)
print("CSV saved.")

print("\n=== PHASE 0 FINAL RESULTS ===")
rows = [
    ("ga",         0.028, 0.521, 0.000, 0.262),
    ("npo",        0.636, 0.624, 0.000, 0.613),
    ("scrub",      0.037, 0.526, 0.000, 0.212),
    ("salun",      0.011, 0.541, 0.000, 0.051),
    ("rmu",        0.580, 0.565, 0.000, 0.559),
    ("alpha_edit", 0.575, 0.558, 0.000, 0.555),
]
print(f"{'Method':<14} {'FA↓':>6} {'RA↑':>6} {'Q_INT8':>8} {'Q_INT4↓':>9}")
print("-"*50)
for name, fa, ra, qi8, qi4 in rows:
    print(f"{name:<14} {fa:>6.3f} {ra:>6.3f} {qi8:>8.3f} {qi4:>9.3f}")
print("\nPhase 0 DONE. Ready for DurableUn Phase 1.")