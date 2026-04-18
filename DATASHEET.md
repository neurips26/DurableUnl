# Datasheet for DurableUn Evaluation Results

Following Gebru et al. (2021), "Datasheets for Datasets."

---

## Motivation

**For what purpose was the dataset created?**
To evaluate machine unlearning methods under INT4/INT8 quantization on LLaMA-3-8B-Instruct with the TOFU benchmark. The dataset consists of model evaluation scores (FA, RA, Q-INT4, Q-INT8, MIA-AUC, cert) produced by running 7 unlearning methods across 3 random seeds.

**Who created the dataset and on behalf of which entity?**
Anonymous authors (NeurIPS 2026 submission). Will be updated upon deanonymisation.

**Who funded the creation?**
Anonymous.

---

## Composition

**What do the instances represent?**
Each row in the CSV files represents one evaluation run: one unlearning method applied to one dataset split (TOFU forget10) on one random seed, producing scalar evaluation metrics.

**How many instances are there?**
Approximately 40+ rows across all CSV files (7 baselines × 1 seed + 6 α values × 3 seeds + SalUn multi-HP × 3 seeds + Mistral-7B).

**Does the dataset contain all possible instances or is it a sample?**
It is the complete set of all runs conducted for the paper. No runs were excluded.

**What data does each instance consist of?**
- `method`: unlearning method name
- `dataset`: evaluation benchmark (tofu)
- `seed`: random seed (42, 123, or 5508)
- `forget_acc` (FA): token-level forget accuracy on TOFU forget10
- `retain_acc` (RA): token-level retain accuracy on TOFU retain90
- `quant_int8` (Q-INT8): forget accuracy after symmetric INT8 quantization
- `quant_int4` (Q-INT4): forget accuracy after symmetric per-row INT4 quantization
- `ra_int4`: retain accuracy after INT4 quantization
- `cert`: Y/N — empirical (0.05,{BF16,INT8,INT4})-durability certificate
- `wall_min`: training time in minutes

**Is there a label or target?**
No human labels. The `cert` column is derived automatically from thresholding Q-INT4.

**Is any information missing?**
No.

**Are there any known errors, sources of noise, or redundancies?**
Timing measurements (wall_min) were affected by Windows background process interference on seeds 123 and 5508 for SAF (475 min vs expected 37 min). The evaluation metrics are unaffected.

**Is the dataset self-contained?**
Yes. All CSV files are included in `results/`. The underlying checkpoints are available upon request.

**Does the dataset contain data that might be considered confidential or sensitive?**
No. All data is model evaluation scores on a synthetic public benchmark (TOFU).

---

## Collection Process

**How was the data associated with each instance acquired?**
Automatically computed by running `run.py` on a single NVIDIA RTX 4090 (24 GB VRAM) with pinned random seeds. Metrics are computed by `src/evaluation/evaluator.py`.

**Was any preprocessing/cleaning/labeling of the data done?**
No. Raw outputs from the evaluation pipeline are stored directly.

**Was the data collected from individuals?**
No. TOFU is a synthetic benchmark of fictitious authors.

**Did the individuals in question consent?**
N/A — no human subjects.

---

## Uses

**Has the dataset been used for any tasks already?**
Yes — to produce the tables and figures in the NeurIPS 2026 submission.

**Is there a repository that links to any or all papers or systems that use the dataset?**
This repository.

**What (other) tasks could the dataset be used for?**
- Benchmarking new unlearning methods under quantization
- Studying the FA–RA–Q-INT4 trilemma
- Reproducing or extending the durability certificate framework

**Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?**
Results are specific to LLaMA-3-8B-Instruct and TOFU. Generalisability to other models and benchmarks is a subject for future work.

---

## Distribution

**How will the dataset be distributed?**
Via this GitHub repository. Results CSVs are included directly.

**When will the dataset be distributed?**
It is already available in this repository.

**Will the dataset be distributed under a copyright or other IP license?**
MIT License.

**Have any third parties imposed IP-based or other restrictions on the data?**
No. The underlying TOFU benchmark is MIT licensed. LLaMA-3-8B model weights are under the Meta Llama 3 Community License; model outputs (evaluation scores) are our own.

---

## Maintenance

**Who is supporting/hosting/maintaining the dataset?**
The authors. Contact information will be added upon deanonymisation.

**How can the owner/curator/manager of the dataset be contacted?**
Via GitHub issues on this repository.

**Will the dataset be updated?**
Future updates may include additional baselines or model architectures.

---

## RAI Fields (NeurIPS Required)

**annotationsPerItem:** N/A — benchmark contains model evaluation scores, not human annotations.

**annotatorDemographics:** N/A — benchmark contains model evaluation scores, not human annotations.
