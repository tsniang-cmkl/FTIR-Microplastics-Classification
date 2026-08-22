# FTIR Microplastics Classification

Deep learning pipeline for classifying microplastics from Fourier-Transform Infrared (FTIR) spectra into 30 polymer classes, under severe membrane-filter interference — developed during a research internship at [CMKL University](https://www.cmkl.ac.th/), Bangkok, supervised by **Dr. Sarun Gulyanon**.

> **Headline result:** a model trained only on synthetic data collapsed from 95.3% to 23.5% accuracy when tested on real spectra. Domain-balanced joint training closed that gap to 99.3% at a cost of just 0.6 points on synthetic accuracy. See [§ The Domain Generalization Gap](#the-domain-generalization-gap).

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [The Domain Generalization Gap](#the-domain-generalization-gap)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Model Weights](#model-weights)
- [Setup](#setup)
- [Usage](#usage)
- [Negative Results](#negative-results)
- [Additional Validation Studies](#additional-validation-studies)
- [Known Limitations & Open Directions](#known-limitations--open-directions)
- [Acknowledgments](#acknowledgments)

---

## Overview

Microplastic identification by FTIR spectroscopy is standard practice, but environmental samples are almost always measured on membrane filters, whose own spectral signature interferes with — and can dominate — the polymer signal. This project builds and validates a multi-task deep learning pipeline that:

1. Classifies a spectrum into one of 30 polymer classes, and
2. Simultaneously denoises it (reconstructs the clean underlying spectrum),

using **PatchTST**, a patch-based Transformer architecture, extended with a **Conformer** backbone and trained via a three-phase protocol (self-supervised pretraining → linear probing → full fine-tuning).

Two data sources are used throughout:

| Source | Size | Nature |
|---|---|---|
| Real (CSV) | ~3,600 spectra | Laboratory-measured, real instrument noise |
| Synthetic (.npy) | 18,000 spectra | Official dataset, numerically simulated filter interference at 3 cumulative noise levels, true noisy/clean pairs |

---

## Key Results

| Metric | Value |
|---|---|
| Best test accuracy, synthetic data (Upto30SNR, hardest condition) | **95.25%** |
| Best test accuracy, real data (jointly-trained model) | **99.25%** |
| Total accuracy gain from architecture evolution alone | +22.14 pts (73.11% → 95.25%) |
| Domain gap closed (real-data accuracy, .npy-only → jointly-trained) | 23.47% → 99.25% (+75.78 pts) |
| Best classical ML baseline (.npy) | 93.72% (XGBoost) — PatchTST +1.53 pts |

Full ablation, sensitivity, and negative-result tables are in [`docs/results_log.md`](docs/results_log.md).

---

## The Domain Generalization Gap

The central finding of this work. A model trained exclusively on the synthetic dataset — 95.25% on its own official test set — was evaluated on held-out real spectra never seen during training:

```
                              Test .npy (synthetic)   Test CSV (real)
.npy-only model                     95.25%                23.47%   ⚠️ collapse
Jointly-trained model               94.62%                99.25%   ✅ resolved
```

**Diagnosis:** the model specialized on the specific simulated filter-interference signature, not general noise-robust polymer chemistry. A normalization control (applying .npy-style min-max scaling to CSV spectra) did not close the gap, confirming a representational rather than a scale-mismatch problem.

**Resolution:** joint training on both domains, using a weighted random sampler to correct the ~7:1 volume imbalance so both are seen with comparable frequency each epoch. CSV denoising targets (no true noisy/clean pairs available) use a per-class mean clean spectrum as proxy.

Independent verification on the full available real-data pool (2,054 spectra): **99.76%**, ruling out memorization of a lucky test split.

Three deliberately tested alternatives — curriculum learning, sequential transfer learning, and self-supervised-only exposure — all failed to close the gap; see [Negative Results](#negative-results). Together they support the conclusion that **concurrent, supervised exposure to both domains** is what the fix actually requires.

---

## Architecture

```
Raw spectrum (L=6700, 650–4000 cm⁻¹)
        │  patch embedding (P=320, S=256 → 26 tokens) + learned positional embedding
        ▼
┌─────────────────────────────────────────────┐
│         Conformer Block  ×  5                │
│  d_model=256 · 16 heads · d_ff=512            │
│                                                │
│   FFN½ (SwiGLU) → Attention → Conv(k=31)      │
│              → FFN½ (SwiGLU) → LayerNorm       │
└─────────────────────────────────────────────┘
        │  shared latent representation
        ├──────────────────────┬─────────────────────
        ▼                      ▼
 Classification Head     Denoising Head
 attention pooling        patch projection
 → MLP → 30 classes       → fold-average → clean spectrum

 Loss = α · CrossEntropy(class) + β · MSE(denoise),  β = 0.5
 Trained jointly (not sequentially) across all three phases.
```

**Training protocol** (3 phases):
1. **Self-supervised pretraining** — 40% of patches masked in contiguous blocks; backbone learns to reconstruct them from context, no labels.
2. **Linear probing** — backbone frozen, only the two task heads trained.
3. **Full fine-tuning** — entire network trained end-to-end at reduced learning rates.

**Design choices, each individually ablated** (see [`docs/results_log.md`](docs/results_log.md) §1):
- SwiGLU-gated FFN (learned, per-dimension gating) instead of GELU
- Attention pooling (learned patch weighting) instead of mean pooling
- Block masking (contiguous region) instead of random-patch masking
- Conformer block (conv + attention per layer) instead of plain Transformer

---

## Repository Structure

```
FTIR-Microplastics-Classification/
├── README.md
├── docs/
│   └── results_log.md              # Full results log (all tables, all experiments)
├── notebooks/
│   ├── PatchTST_MultiTask_Mixed.ipynb        # Main pipeline — produces the reference model
│   ├── RF_GB_Baseline_npy.ipynb              # Classical ML baselines (RF / XGBoost / LightGBM)
│   ├── PatchTST_CSV_Inference.ipynb          # Standalone inference / domain-gap evaluation
│   └── experiments/
│       ├── negative_results/
│       │   ├── PatchTST_TransferLearning.ipynb
│       │   └── PatchTST_SSLMixed_NpyOnly.ipynb
│       ├── sensitivity/
│       │   ├── PatchTST_CSV_Sensitivity.ipynb
│       │   └── RF_GB_CSV_Sensitivity_Mixed.ipynb
│       └── validation/
│           ├── PatchTST_SSL_Embeddings_XGB.ipynb
│           ├── PatchTST_CSV_WithClean_RunB.ipynb
│           └── PatchTST_CSV_Augmented.ipynb
```

---

## Data

Data is **not included in this repository** (large files, restricted access). Both sources are interpolated onto a common 6,700-point grid (650–4000 cm⁻¹, 0.5 cm⁻¹ step).

- **Synthetic (.npy):** official dataset provided by Dr. Sarun Gulyanon. Contact the supervisor for access.
- **Real (CSV):** laboratory measurements, heterogeneous CSV files and proprietary OMNIC (.SPG) binaries, parsed via the pipeline in `PatchTST_MultiTask_Mixed.ipynb` (data-loading section).

---

## Model Weights

Trained weights are hosted on Google Drive (not committed to this repository):

📁 **[Model weights — Google Drive](https://drive.google.com/drive/folders/1c5IcsIIXmGnhoMAaSw3otU8KbwitXiUg?usp=drive_link)**

| File | Description | Test .npy | Test CSV |
|---|---|---|---|
| `final_model_mixed.pth` | **Reference model** — jointly trained on both domains | 94.6–94.7% | 99.0–99.3% |
| `final_model_v2.pth` | .npy-only model — kept to reproduce the domain-gap finding | 95.25% | 23.47% |
| `ssl_backbone_mixed.pth` | Phase-1 SSL backbone only (for reproducing Phase 2/3 without re-running Phase 1) | — | — |

---

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn xgboost lightgbm seaborn tensorboard pandas numpy matplotlib
```

Tested on Python 3.10, PyTorch (CUDA 11.8 build).

---

## Usage

**Reproduce the reference model:**
```bash
jupyter notebook notebooks/PatchTST_MultiTask_Mixed.ipynb
```
Runs all three phases (SSL pretraining → probing → fine-tuning) on the joint .npy + CSV dataset and evaluates on both held-out test sets.

**Run inference / reproduce the domain-gap evaluation only:**
```bash
jupyter notebook notebooks/PatchTST_CSV_Inference.ipynb
```
Loads a saved checkpoint from `final_model_v2.pth` or `final_model_mixed.pth` (see [Model Weights](#model-weights)) and evaluates on real CSV spectra.

Each notebook's data-loading cells expect `.npy` and CSV data under `~/data/` — adjust `NPY_ROOT` / `CSV_ROOT` at the top of each notebook to match your environment.

---

## Negative Results

Documented transparently, as each failure clarifies why the joint-training solution works. Full detail in [`docs/results_log.md`](docs/results_log.md) §4.

| Approach | Result | Notes |
|---|---|---|
| Curriculum learning (easy → hard noise) | 82.10% vs. 92.49% direct | Filter interference is structured, not generic noise |
| Sequential transfer (.npy → fine-tune CSV) | 72.5% / 90.5% vs. 94.6% / 99.3% joint | Catastrophic forgetting on the source domain |
| Self-supervised exposure alone (imbalanced pool) | 26.75% on CSV | Barely above the 23.47% no-exposure baseline |
| Self-supervised exposure alone (rebalanced 18k/18k pool) | ~35% on CSV | Still far short of joint supervised training |

---

## Additional Validation Studies

Requested by Dr. Sarun as follow-ups to the main domain-gap finding. Full tables in [`docs/results_log.md`](docs/results_log.md) §5.

- **Data-volume sensitivity** (PatchTST and classical ML): accuracy on real data plateaus down to ~800 training spectra, degrades sharply below ~260.
- **Relative importance of each domain**: a CSV-only model (no .npy at all) collapses to 7.27% on synthetic data — the gap is symmetric.
- **Quality of SSL-only representations**: embeddings from Phase 1 alone match hand-crafted features on real data, but fall ~10 points short on the hardest synthetic condition.
- **CSV augmentation by pairwise averaging**: small net gain (~97.2% vs. 96.9% average across both domains) when real-data volume is synthetically expanded to 18,000 spectra.

---

## Known Limitations & Open Directions

- Real-data test set is modest in size (order of a few hundred spectra); per-class estimates for rare polymers carry non-trivial variance.
- Real membrane-filter (as opposed to CSV lab-measurement) evaluation is preliminary and inconclusive at this stage.
- Per-class error analysis for the joint-training model is not yet done (is the 0.6-pt synthetic-domain cost evenly distributed, or concentrated in specific classes?).
- Synthetic data augmentation via masked reconstruction from the pretrained backbone has not been attempted (only pairwise averaging has been tested).

---

## Acknowledgments

This work was carried out during a research internship at CMKL University, Bangkok, under the supervision of **Dr. Sarun Gulyanon**, who defined the project's scientific direction and the validation programme documented in [`docs/results_log.md`](docs/results_log.md).
