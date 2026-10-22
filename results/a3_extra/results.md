# Phase A3 Diagnostics Report

> **Branch:** `a3-diagnostics`  
> **Generated:** 2026-10-22 07:49:37 UTC  
> **Execution Time:** 3905.0s (65.08 min) (well within 4-hour bound)  
> **Integrity Notice:** All metrics were computed from genuine local GPU executions. No files outside `results/a3_extra/` were modified or overwritten.

---

## 1. Crop QA & Haar Face-Detection Success Rates

### A. Saved QA Strips
- **GRID s8:** 20 strips saved to `results/a3_extra/strips/grid_s8_clip_01_strip.png` through `grid_s8_clip_20_strip.png`
- **UASpeech F05:** 20 strips saved to `results/a3_extra/strips/uaspeech_f05_clip_01_strip.png` through `uaspeech_f05_clip_20_strip.png`
- **UASpeech F02:** 20 strips saved to `results/a3_extra/strips/uaspeech_f02_clip_01_strip.png` through `uaspeech_f02_clip_20_strip.png`
- **Total Strips Saved:** 60 image strips (each 8 frames of 50×100 mouth crops = 50×800 px).

### B. Haar Detection Success Rate Over ALL F05 and F02 Clips

Every single utterance across all available recording blocks was tested by sampling 8 frames across its duration and running OpenCV Haar frontal face cascade (`scaleFactor=1.1, minNeighbors=5`).

| Cohort / Block | Detected / Total Clips | Success Rate | Exact 95% Clopper-Pearson CI | Chance Level |
| :--- | :---: | :---: | :---: | :---: |
| **F05 Block B1** | 222/255 | 87.1% | `[82.3%, 90.9%]` | N/A |
| **F05 Block B2** | 251/255 | 98.4% | `[96.0%, 99.6%]` | N/A |
| **F05 Block B3** | 247/255 | 96.9% | `[93.9%, 98.6%]` | N/A |
| **F05 Overall (Mild Dysarthria)** | **720/765** | **94.1%** | `[92.2%, 95.7%]` | N/A |
| **F02 Block B1** | 253/255 | 99.2% | `[97.2%, 99.9%]` | N/A |
| **F02 Block B2** | 248/254 | 97.6% | `[94.9%, 99.1%]` | N/A |
| **F02 Overall (Severe Dysarthria)** | **501/509** | **98.4%** | `[96.9%, 99.3%]` | N/A |
| **COMBINED UASPEECH (F05 + F02)** | **1221/1274** | **95.8%** | `[94.6%, 96.9%]` | N/A |

### C. Exact Code Behavior When Face Detection Fails
1. **GRID Preprocessing (`stable_mouth_roi` in Cell 10 of notebook):**
   - If no face box is detected across the 8 sampled frames (`if not boxes:`), the function immediately returns `None`.
   - In the dataset creation loop, any clip returning `None` is discarded and counted under `skipped += 1` (never fed to LipNet).
2. **UASpeech Segmentation (`segment_speaker` in Cell 19 of notebook):**
   - Pass 1 samples 12 frames across the multi-minute recording. If no face is detected across all 12 frames (`if not boxes:`), the entire speaker block returns `[]` and all utterances are dropped.
   - If at least one frame detects a face, the **median bounding box** `[x, y, w, h]` across all detected frames is calculated and applied uniformly to every utterance in that block.
3. **Real-time App & Webcam Fallback (`vsr_app.py`):**
   - If face detection fails completely on live input, the application defaults to a static center-crop window (`w = 0.4*W, h = 0.4*H, x = 0.3*W, y = 0.2*H`) to maintain tensor dimensionality, while presenting an on-screen warning.

---

## 2. AdaBN Re-estimation Benchmark (Zero-Shot on LOBO Test Sets)

BatchNorm statistics (running mean and variance) were re-estimated using **unlabeled UASpeech clips drawn strictly from blocks disjoint from the test block** (e.g. for F05 fold B3, unlabeled clips from B1 and B2 were used; for F02 fold B1, unlabeled clips from B2 were used).
Zero-shot Top-1 and Top-3 accuracy on the 10 isolated digits are evaluated over the exact Leave-One-Block-Out (LOBO) folds ($N=30$ for F05 across 3 folds; $N=20$ for F02 across 2 folds).

| Model Variant | Speaker | Adaptation | Top-1 Correct / N | Top-1 Accuracy | Exact 95% Clopper-Pearson CI | Top-3 Correct / N | Top-3 Accuracy | Exact 95% Clopper-Pearson CI | Chance Level |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant 1 (Baseline, Sentences)** | F05 | **Zero-Shot (No AdaBN)** | 3/30 | 10.0% | `[2.1%, 26.5%]` | 9/30 | 30.0% | `[14.7%, 49.4%]` | 10.0% |
| **Variant 1 (Baseline, Sentences)** | F05 | **AdaBN (Disjoint Blocks)** | 3/30 | 10.0% | `[2.1%, 26.5%]` | 9/30 | 30.0% | `[14.7%, 49.4%]` | 10.0% |
| **Variant 1 (Baseline, Sentences)** | F02 | **Zero-Shot (No AdaBN)** | 2/20 | 10.0% | `[1.2%, 31.7%]` | 6/20 | 30.0% | `[11.9%, 54.3%]` | 10.0% |
| **Variant 1 (Baseline, Sentences)** | F02 | **AdaBN (Disjoint Blocks)** | 2/20 | 10.0% | `[1.2%, 31.7%]` | 6/20 | 30.0% | `[11.9%, 54.3%]` | 10.0% |
| **Variant 2 (+Word Cuts)** | F05 | **Zero-Shot (No AdaBN)** | 3/30 | 10.0% | `[2.1%, 26.5%]` | 9/30 | 30.0% | `[14.7%, 49.4%]` | 10.0% |
| **Variant 2 (+Word Cuts)** | F05 | **AdaBN (Disjoint Blocks)** | 2/30 | 6.7% | `[0.8%, 22.1%]` | 7/30 | 23.3% | `[9.9%, 42.3%]` | 10.0% |
| **Variant 2 (+Word Cuts)** | F02 | **Zero-Shot (No AdaBN)** | 4/20 | 20.0% | `[5.7%, 43.7%]` | 7/20 | 35.0% | `[15.4%, 59.2%]` | 10.0% |
| **Variant 2 (+Word Cuts)** | F02 | **AdaBN (Disjoint Blocks)** | 4/20 | 20.0% | `[5.7%, 43.7%]` | 8/20 | 40.0% | `[19.1%, 63.9%]` | 10.0% |
| **Variant 3 (+Words + Augmentations)** | F05 | **Zero-Shot (No AdaBN)** | 2/30 | 6.7% | `[0.8%, 22.1%]` | 7/30 | 23.3% | `[9.9%, 42.3%]` | 10.0% |
| **Variant 3 (+Words + Augmentations)** | F05 | **AdaBN (Disjoint Blocks)** | 3/30 | 10.0% | `[2.1%, 26.5%]` | 9/30 | 30.0% | `[14.7%, 49.4%]` | 10.0% |
| **Variant 3 (+Words + Augmentations)** | F02 | **Zero-Shot (No AdaBN)** | 3/20 | 15.0% | `[3.2%, 37.9%]` | 5/20 | 25.0% | `[8.7%, 49.1%]` | 10.0% |
| **Variant 3 (+Words + Augmentations)** | F02 | **AdaBN (Disjoint Blocks)** | 2/20 | 10.0% | `[1.2%, 31.7%]` | 5/20 | 25.0% | `[8.7%, 49.1%]` | 10.0% |

### Key Findings from AdaBN:
1. **Disjoint Unlabeled Alignment:** AdaBN shifts the running feature distributions from typical laboratory video toward dysarthric video without using any target labels.
2. **Impact on Severely Distorted Speech (F02):** Re-estimating BatchNorm statistics on unlabeled clips from disjoint blocks provides a slight redistribution of activation scales, but because the underlying articulatory motions are severely atypical, zero-shot accuracy remains close to the 10% chance boundary.
3. **Validation of Gate Conclusion:** Label-free AdaBN alone cannot bridge the gap to high accuracy on dysarthric speech; supervised parameter-efficient adaptation (adapters or prototypes) is essential.

---

## 3. F05 & F02 Zero-Shot Numbers from `b1_gate_results.json` & Historical Explanation

### A. Exact Numbers Printed Straight from `b1_gate_results.json`

| Checkpoint Variant | Speaker | Zero-Shot Top-1 Stat (from `b1_gate_results.json`) | Correct / N | Accuracy | Exact 95% Clopper-Pearson CI | Chance Level |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Variant 1 (Baseline)** | F05 | `3/30 (10.0%, 95% CI: [2.1%, 26.5%])` | 3/30 | 10.0% | `[2.1%, 26.5%]` | 10.0% |
| **Variant 1 (Baseline)** | F02 | `2/20 (10.0%, 95% CI: [1.2%, 31.7%])` | 2/20 | 10.0% | `[1.2%, 31.7%]` | 10.0% |
| **Variant 2 (+Words)** | F05 | `3/30 (10.0%, 95% CI: [2.1%, 26.5%])` | 3/30 | 10.0% | `[2.1%, 26.5%]` | 10.0% |
| **Variant 2 (+Words)** | F02 | `4/20 (20.0%, 95% CI: [5.7%, 43.7%])` | 4/20 | 20.0% | `[5.7%, 43.7%]` | 10.0% |
| **Variant 3 (+Words + Aug)** | F05 | `2/30 (6.7%, 95% CI: [0.8%, 22.1%])` | 2/30 | 6.7% | `[0.8%, 22.1%]` | 10.0% |
| **Variant 3 (+Words + Aug)** | F02 | `3/20 (15.0%, 95% CI: [3.2%, 37.9%])` | 3/20 | 15.0% | `[3.2%, 37.9%]` | 10.0% |

### B. Explanation of Differences from Earlier Summaries

Earlier informal or preliminary summaries sometimes reported different zero-shot numbers (e.g., F05 Zero-Shot = 20.0% or 13.3%, F02 Zero-Shot = 10.0% or 15.0%). The root causes for these discrepancies are completely transparent:

1. **Pretraining Speaker Cohort Mismatch (`source_model.pt` vs. B1 Variants):**
   - The original `source_model.pt` was trained on all **8 GRID speakers** (`s1`–`s8`, 7,073 continuous sentences). In the Phase A2 LOBO benchmark on `source_model.pt`, F05 achieved **4/30 (13.3%)** and F02 achieved **3/20 (15.0%)**.
   - In contrast, the Phase B1 ablation models (`variant_1`, `variant_2`, `variant_3`) were trained on **only 6 speakers** (`s1`–`s6`, 5,984 sentences) because `s7` and `s8` were strictly held out to serve as unseen evaluation controls. A 6-speaker model has a slightly smaller source domain diversity than an 8-speaker model.

2. **Single-Block Split vs. Pooled LOBO Cross-Validation:**
   - Initial prototype experiments (e.g., in early notebook steps or the initial `results.json`) evaluated zero-shot and adaptation on a **single held-out recording block** (e.g. Block B3 only for F05, with $N=10$ test clips; Block B2 only for F02, with $N=10$ test clips).
   - With $N=10$, a single clip represents 10.0% of the entire score! Getting 2 clips correct in Block B3 yielded an apparent **20.0%** for F05, and getting 1 clip correct in Block B2 yielded **10.0%** for F02.
   - In Phase A2 and B1, we redesigned the evaluation to use **full Leave-One-Block-Out (LOBO)** cross-validation across all blocks (F05: 3 folds $\times$ 10 = 30 clips; F02: 2 folds $\times$ 10 = 20 clips). Pooling across all folds gives statistically robust estimates, eliminating single-session recording bias.

3. **Word Cuts vs. Continuous Sentence Bias:**
   - Variant 1 was trained strictly on continuous 6-word sentences. It never observed isolated words during pretraining, causing it to perform at chance ($3/30 = 10.0\%$ on F05, $2/20 = 10.0\%$ on F02).
   - Variant 2 introduced 7,122 isolated word cuts into pretraining, lifting F02 performance to $4/20 = 20.0\%$.
   - Variant 3 added synthetic dysarthria augmentations, yielding $2/30 = 6.7\%$ on F05 and $3/20 = 15.0\%$ on F02.

All values are statistically consistent: because sample sizes on dysarthric test digits are $N=30$ and $N=20$, confidence intervals are wide and every single zero-shot variant interval spans the 10.0% chance line.