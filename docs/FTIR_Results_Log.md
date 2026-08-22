# FTIR Microplastics Classification — Results Log

*Théo-Sambacor Niang · CMKL University · Supervised by Dr. Sarun Gulyanon*

---

## 1. Architecture Evolution (Ablation Study)

All results on the official synthetic (.npy) test set, hardest noise condition (Upto30SNR), unless stated otherwise. Each row adds one component on top of the previous, validated in isolation.

| Configuration | Test Accuracy | SNR Gain |
|---|---|---|
| GELU + mean pooling (baseline) | 73.11% | +5.82 dB |
| + SwiGLU-gated FFN with local convolution | 89.69% | +6.16 dB |
| + Attention pooling (replacing mean pooling) | 92.49% | +7.27 dB |
| + Block masking (replacing random masking) | 92.49% | +7.06 dB |
| + Conformer block, P=320/S=256, n=4, d_model=128 | 94.09% | +8.61 dB |
| + Capacity scaling, d_model=256, n_heads=8 | 94.83% | +9.66 dB |
| + n_heads=16 (same config) | 94.51% | +10.10 dB |
| + n_layers=5, d_model=256, n_heads=8 | 95.18% | +9.88 dB |
| **Final: n_layers=5, d_model=256, n_heads=16** | **95.25%** | **+10.01 dB** |

**Total gain: +22.14 points, +4.19 dB, from baseline to final architecture.**

### 1.1 Patch geometry sweep (Conformer, n_layers=4, d_model=128)

| Patch / Stride | N patches | Test Accuracy |
|---|---|---|
| P=128, S=64 | 105 | 88.99–89.69% |
| P=256, S=192 | 35 | 94.09% |
| **P=320, S=256** | **26** | **94.09% (best)** |
| P=384, S=288 | 23 | 93.52% |

Inverted-U relationship — neither more nor fewer patches than the optimum helps.

### 1.2 Other ablations tested

| Parameter | Values tested | Conclusion |
|---|---|---|
| beta (denoising loss weight) | 0.2, 0.5 | 0.5 optimal on .npy (true pairs); 0.2 optimal on CSV-only (proxy target) |
| n_heads | 4, 8, 16 | 8 best for accuracy; 16 best for SNR |
| n_layers | 3, 4, 5, 6 | 5 optimal at d_model=256; 4 optimal at d_model=128 |

---

## 2. Classical ML Baselines

### 2.1 On real (CSV) data — smaller, chemically well-structured

| Model | Test Accuracy |
|---|---|
| Random Forest + PCA | 98.13% |
| PatchTST (early architecture) | 88.22% |

### 2.2 On synthetic (.npy) data — larger, harder noise condition

| Model | Test Accuracy |
|---|---|
| Random Forest | 93.06% |
| LightGBM | 93.58% |
| XGBoost | 93.72% |
| **PatchTST Conformer (final)** | **95.25%** |

**Interpretation:** on the smaller, well-structured real dataset, classical ML exploits known structure efficiently with limited data. On the larger, harder synthetic dataset, the deep-learning advantage emerges as expected.

---

## 3. The Domain Generalization Gap

### 3.1 Discovery

| Evaluation | Accuracy |
|---|---|
| .npy-trained model → .npy official test | 95.25% |
| .npy-trained model → real CSV test | **23.47%** |

**Root cause:** the model specialized on the specific simulated membrane-filter interference signature, not general noise-robust polymer chemistry. Control test (applying .npy-style min-max normalization to CSV) did not close the gap — confirms a representational failure, not a scale mismatch.

### 3.2 Resolution — joint (mixed) training

Weighted random sampler corrects the ~7:1 volume imbalance (.npy vs. CSV) so both domains are seen with comparable frequency per epoch.

| Evaluation | .npy-only model | Jointly-trained model |
|---|---|---|
| Test .npy (official) | 95.25% | 94.62% (also reproduced: 94.72%) |
| Test CSV (real, held out) | 23.47% | 99.25% (also reproduced: 99.00%) |

Independent verification on the full available real-data pool (2,054 spectra, including some overlapping training): **99.76%** — confirms generalization, not memorization.

*(Run-to-run variance of ±0.2–0.3 pts between reproductions of the same config is normal GPU/sampler stochasticity, not a regression.)*

---

## 4. Negative Results

### 4.1 Curriculum learning

Pretrain on easy noise (Upto10SNR), fine-tune on hard noise (Upto30SNR).

| Strategy | Test Accuracy (Upto30SNR) |
|---|---|
| Direct training on Upto30SNR | 92.49% |
| Curriculum: Upto10SNR → fine-tune Upto30SNR | 82.10% |

**Interpretation:** filter interference is a structured, specific signal, not generic noise a curriculum can ease into.

### 4.2 Sequential transfer learning

Train fully on .npy, then fine-tune on CSV only (no joint exposure).

| | .npy accuracy | CSV accuracy |
|---|---|---|
| Sequential transfer | 72.52% | 90.50% |
| Joint training (reference) | 94.62% | 99.25% |

**Interpretation:** severe catastrophic forgetting — worse than joint training on *both* domains.

### 4.3 Self-supervised exposure alone (unbalanced pool)

SSL sees both domains (imbalanced: ~1,864 CSV vs. 17,100 .npy); only .npy used for supervised Phase 2/3.

| | .npy accuracy | CSV accuracy |
|---|---|---|
| SSL mixed (imbalanced) → Phase 2/3 npy-only | 95.08% | 26.75% |
| Reference (.npy-only model, no CSV exposure at all) | 95.25% | 23.47% |

Only +3.28 pts over the no-exposure baseline — unsupervised exposure alone barely helps.

### 4.4 Self-supervised exposure alone (rebalanced pool)

Same idea, but SSL pool rebalanced to 18,000 .npy + 18,000 augmented CSV (see §5.4), Phase 2/3 still npy-only.

| | CSV accuracy |
|---|---|
| SSL balanced (18k/18k) → Phase 2/3 npy-only | ~35% (exploratory, not fully conclusive) |

Still far below the 99%+ achieved by joint supervised training — confirms unsupervised exposure, even rebalanced, does not substitute for supervised signal from the target domain.

**Combined conclusion from §4.2–4.4:** closing the domain gap requires *supervised, concurrent* exposure to both domains — not sequential, not curriculum-ordered, not unsupervised-only.

---

## 5. Additional Validation Studies

### 5.1 Data-volume sensitivity — PatchTST (mixed model)

Nested subsampling (each smaller subset ⊂ next larger one), fixed test set.

| CSV Train Fraction | CSV Train N | Test .npy | Test CSV |
|---|---|---|---|
| 1.00 | 1,864 | 95.03% | 98.50% |
| 0.70 | 1,304 | 94.71% | 98.00% |
| 0.43 | 801 | 94.92% | 97.75% |
| 0.14 | 260 | 94.63% | 92.25% |
| 0.07 | 130 | 94.85% | 81.25% |

Plateau down to ~800 spectra; sharp degradation below ~260.

### 5.2 Data-volume sensitivity — Classical ML (same structure)

Same nested subsampling, same seed, RF/XGBoost/LightGBM on combined .npy + CSV-fraction training pool.

| CSV Train Fraction | CSV Train N | Test .npy | Test CSV |
|---|---|---|---|
| 1.00 | 1,864 | 93.29% | 98.00% |
| 0.70 | 1,304 | 93.46% | 97.50% |
| 0.43 | 801 | 93.09% | 97.25% |
| 0.14 | 260 | 92.73% | 89.75% |
| 0.07 | 130 | 93.36% | 81.75% |

Classical ML follows the same sensitivity profile as PatchTST, consistently 1–2 pts lower throughout.

### 5.3 Relative importance of the .npy domain

| Run | Description | Test CSV | Test .npy |
|---|---|---|---|
| Run A (historical) | CSV only, clean spectra used only as denoising proxy, older architecture config | 94.21% | — |
| Run B | CSV only + clean spectra added directly as classification examples, no .npy at all | 98.75% | **7.27%** |

**Interpretation:** the domain gap is symmetric — a model trained on CSV alone collapses on .npy just as severely as the reverse. Neither data source substitutes for the other.

### 5.4 Quality of self-supervised (SSL-only) representations

XGBoost / LightGBM trained directly on backbone embeddings after Phase 1 only (mean-pooled, no supervised fine-tuning, no attention pooling).

| | Test .npy | Test CSV |
|---|---|---|
| Classical ML on raw spectra / PCA (reference) | 93.29% | 98.00% |
| XGBoost on SSL-only embeddings | 82.68% | 97.50% |
| LightGBM on SSL-only embeddings | 83.06% | 98.00% |
| PatchTST full classification (reference) | 95.03% | 98.50% |

**Interpretation:** SSL-only embeddings are competitive with hand-crafted features on real (CSV) data, but ~10 points short on the harder synthetic condition — explicit supervision (attention pooling + fine-tuning) remains necessary to extract signal from severely degraded spectra.

### 5.5 CSV data augmentation by pairwise averaging

Synthetic real-like spectra generated by averaging two same-class real spectra (random pairs, train split only), up to 600/class (18,000 total, matching .npy scale). Reuses existing SSL backbone (Phase 2/3 only).

| Run | Test .npy | Test CSV |
|---|---|---|
| Reference (no augmentation, CSV ~1,864) | 94.62% | 99.25% |
| Augmented, run 1 | 95.72% | 98.75% |
| Augmented, run 2 | 95.42% | 98.87% |
| **Augmented, average** | **~95.57%** | **~98.81%** |

Small net gain in average (~97.19% vs. 96.94% for the reference), with an asymmetric trade: +0.8–1.1 pt on .npy, −0.4–0.5 pt on CSV. Consistent with averaging reducing independent noise, making synthetic spectra statistically closer to clean references than genuinely new measurements.

---

## 6. Exploratory Work — Real Membrane Filters

Preliminary evaluation on genuine membrane-filter measurements: (a) real microplastic-on-filter samples (no ground-truth label, confidence examined qualitatively), and (b) pure filter-substrate spectra with no microplastic present (false-positive robustness check). **Results inconclusive at this stage** — flagged as an open direction, not a validated finding.

---

## 7. In Progress

- **Full mixed pipeline with real Phase 1 SSL pretraining on augmented CSV** (18,000 .npy + 18,000 augmented CSV from the start, rather than reusing the existing SSL backbone). Notebook built, not yet run at time of writing.
- Per-class error analysis (whether the ~0.6 pt synthetic-domain cost of joint training is evenly distributed or concentrated in specific polymer classes).
- Deeper validation campaign on real membrane-filter measurements.

---

*Log compiled from experiment tracking (TensorBoard HParams dashboards) and notebook run outputs throughout the internship.*
