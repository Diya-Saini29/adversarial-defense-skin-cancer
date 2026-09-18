# 🛡️ Adversarial Robustness of Dermatological Image Classifiers

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange.svg)]()

A research project on the **adversarial robustness** of deep learning models for skin lesion classification, spanning **multi-dataset training**, **red-team attack evaluation**, and **blue-team defense mechanisms**.

---

## 📖 Overview

Deep learning models achieve impressive accuracy on dermatological image classification, but are they **secure**? This project investigates that question through three phases:

1. **Baseline Modeling** — Build strong classification baselines across four public dermatology datasets
2. **Adversarial Attacks (Red Team)** — Systematically probe model vulnerability using FGSM and PGD attacks
3. **Defense Mechanisms (Blue Team)** — Harden models using adversarial training and (in progress) MDDA domain alignment

**Core architecture:** `DenseNet121` + `CBAM attention` + `Focal Loss`, applied consistently across all datasets for reproducibility.

---

## 🎯 Objectives

- Establish **reproducible baselines** for skin lesion classification across four datasets
- **Quantify adversarial vulnerability** using industry-standard attack metrics (ASR, Robust Accuracy, Confidence Drop)
- **Design and validate defense mechanisms** that preserve clean accuracy while improving robustness
- Provide a **cross-dataset robustness comparison** as a publication-grade research contribution

---

## 🏗️ Architecture Pipeline
                    ┌─────────────────────────────────────┐
                    │   INPUT                             │
                    │   Dermoscopic Image (RGB)           │
                    │   Any resolution                    │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │   PREPROCESSING                     │
                    │   • CLAHE in LAB color space        │
                    │     (clip=2.0, grid=8×8)            │
                    │   • Resize → 224 × 224              │
                    │   • ImageNet normalization          │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │   DATA AUGMENTATION (train only)    │
                    │   • RandomHorizontalFlip (p=0.5)    │
                    │   • RandomVerticalFlip   (p=0.3)    │
                    │   • RandomRotation       (±30°)     │
                    │   • ColorJitter          (±0.15)    │
                    │   • RandomAffine         (10%)      │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │   BACKBONE                          │
                    │   DenseNet121                       │
                    │   (ImageNet pretrained)             │
                    │   → 1024-channel feature map        │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │   ATTENTION                         │
                    │   CBAM                              │
                    │   • Channel Attention (reduction=16)│
                    │   • Spatial Attention (kernel=7×7)  │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │   CLASSIFICATION HEAD               │
                    │   • Global Average Pooling          │
                    │   • Dropout (p=0.3)                 │
                    │   • Linear (1024 → 8 classes)       │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │   TRAINING CONFIGURATION            │
                    │   • Loss:      Focal Loss (α=0.25,  │
                    │                γ=2.0)               │
                    │   • Optimizer: AdamW (lr=3e-4,      │
                    │                wd=1e-3)             │
                    │   • Scheduler: CosineAnnealingLR    │
                    │                (T_max=40)           │
                    │   • Batch:     32                   │
                    │   • Epochs:    40                   │
                    └─────────────────────────────────────┘
---

## 📊 Datasets

| Dataset | Images | Classes | Baseline Test Acc | Macro F1 |
|---------|--------|---------|-------------------|----------|
| **HAM10000** | 10,015 | 7 | **87.72%** | 81.57% |
| **ISIC 2019** | 25,258 | 8 | **84.04%** | 78.28% |
| **BCN20000** | 18,946 | 8 | **83.14%** | 79.62% |
| **MILK10k** (dermoscopic only) | 2,622 | 7* | **70.30%** | 63.58% |
| **Combined BCN+MILK** | 15,035 | 8 | **63.65%** | 56.73% |

> *MILK10k lacks SCC class after label harmonization → effective 7 classes.

All datasets are preprocessed identically (CLAHE in LAB) with the same augmentation pipeline — ensuring **cross-dataset reproducibility**.

---

## ⚔️ Adversarial Attacks (Red Team)

### Attack Suite

| Attack | Type | Iterations | Description |
|--------|------|------------|-------------|
| **FGSM** | Single-step | 1 | Fast Gradient Sign Method — fast baseline attack |
| **PGD** | Iterative | 10 | Projected Gradient Descent — strong attack |
| **Spatial** | Masked | — | Full / top-half / bottom-half / patch-based |

### Threat Surfaces

- **Full-image attacks** — perturbations across entire image
- **Partial-image attacks** — top or bottom 50% only
- **Patch attacks** — 16×16, 32×32, 48×48, 64×64 localized patches

### Perturbation Budgets (ε)

`1/255, 2/255, 4/255, 8/255, 16/255` — measuring model vulnerability across a range of imperceptible perturbations.

### Key Findings

| Dataset | FGSM ASR (full, ε=8/255) | PGD ASR (full, ε=8/255) |
|---------|--------------------------|--------------------------|
| HAM10000 | 97.5% | **100%** |
| ISIC 2019 | 98.5% | **100%** |
| MILK10k | 99.6% | **100%** |

**⚠️ All baseline models are catastrophically vulnerable to full-image PGD attacks (100% ASR).**

**Notably:** Patch attacks (13–20% ASR) are **far less effective** — suggesting CBAM provides some natural spatial robustness by attending to lesion regions.

---

## 🛡️ Defense Mechanisms (Blue Team)

### 1. Adversarial Training (Completed on 3/4 datasets)

**Recipe (identical across all datasets):**

```python
MULTI_ATTACK_CONFIG = {
    'num_epochs': 20,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epsilon': 4/255,           # training perturbation budget
    'pgd_iter': 7,               # PGD iterations during training
    'pgd_alpha': 1/255,
    'clean_loss_weight': 0.3,    # mixed clean/adv loss
    'adv_loss_weight': 0.7,
}
Multi-attack training: At every batch, one of six attack types is randomly selected:

FGSM full-image

PGD full-image

PGD top-half

PGD bottom-half

PGD center patch

PGD top-center patch

This forces the model to be robust against a distribution of threats rather than a single attack.

Defense Results
Dataset	Full-Image ASR (avg)	Partial-Image ASR	Patch ASR
MILK10k	91.55% → 61.43% (Δ +30.1%)	73.72% → 40.37% (Δ +33.4%)	15.88% → 5.89% (Δ +10.0%)
Combined BCN+MILK	evaluation in progress	pending	pending
ISIC 2019	training in progress	pending	pending
Best-case improvement (MILK10k): PGD at ε=1/255: ASR 72.6% → 18.5% (Δ +54.1%)

2. MDDA — Multi-Domain Distribution Alignment (In Progress)
Extending the defense pipeline with MDDA to:

Learn domain-invariant features across BCN and MILK source distributions

Improve generalization to unseen domains and distribution-shift evasion attacks

Complement adversarial training with a feature-level robustness mechanism

Pipeline:

text
Adversarial Training → MDDA (domain alignment) → Unified Defended Model
Targets a comprehensive defense evaluated against the full FGSM/PGD attack suite.

📁 Repository Structure
text
.
├── data/                                # Dataset directories (not committed)
│   ├── HAM10000/
│   ├── ISIC2019_7class_splits/
│   ├── BCN20000/
│   ├── MILK10k/
│   └── COMBINED_BCN_MILK/
│
├── notebooks/
│   ├── 01_HAM10000_baseline.ipynb       # Baseline training
│   ├── 02_ISIC2019_baseline.ipynb
│   ├── 03_BCN20000_baseline.ipynb
│   ├── 04_MILK10k_baseline.ipynb
│   ├── 05_combined_baseline.ipynb
│   ├── 06_attack_evaluation.ipynb        # Red-team attacks
│   ├── 07_adversarial_training.ipynb     # Blue-team defense
│   └── 08_mdda_defense.ipynb             # MDDA (in progress)
│
├── src/
│   ├── models/
│   │   └── densenet_cbam.py              # Model architecture
│   ├── attacks/
│   │   ├── fgsm.py
│   │   ├── pgd.py
│   │   └── spatial.py
│   ├── defenses/
│   │   ├── adversarial_training.py
│   │   └── mdda.py                       # In progress
│   ├── data/
│   │   ├── preprocessing.py              # CLAHE, transforms
│   │   └── dataset.py                    # Dataset classes
│   └── utils/
│       ├── metrics.py                    # ASR, Robust Acc, etc.
│       └── visualization.py
│
├── results/
│   ├── baseline/                         # Baseline CSVs + plots
│   ├── attacks/                          # Attack result CSVs
│   ├── defense/                          # Defense result CSVs
│   └── comparison/                       # Cross-dataset comparisons
│
├── requirements.txt
├── README.md
└── LICENSE
🛠️ Technologies & Frameworks
Category	Tools
Deep Learning	PyTorch, Torchvision
Attention	CBAM (custom implementation)
Loss	Focal Loss (custom implementation)
Preprocessing	OpenCV (CLAHE), Pillow
Metrics	Scikit-learn
Visualization	Matplotlib, Seaborn
Environment	Google Colab, Kaggle (T4/P100)
🚀 Getting Started
1. Clone the Repository
bash
git clone https://github.com/<your-username>/adversarial-dermatology.git
cd adversarial-dermatology
2. Install Dependencies
bash
pip install -r requirements.txt
3. Download Datasets
Datasets are not included in the repo due to size. Download from:

HAM10000: ISIC Archive

ISIC 2019: ISIC Challenge 2019

BCN20000: Figshare

MILK10k: Harvard Dataverse

Place them under data/ following the directory structure above.

4. Run Baseline Training
Open notebooks/04_MILK10k_baseline.ipynb and run all cells. Replace DATASET_PATH with your local path.

5. Run Attack Evaluation
Open notebooks/06_attack_evaluation.ipynb. Load a trained checkpoint and run the full attack suite.

6. Run Adversarial Training
Open notebooks/07_adversarial_training.ipynb. Trains a defended model on the multi-attack objective.

📈 Results Summary
Baseline vs Defended (MILK10k)
Attack	Baseline ASR	Defended ASR	Δ Improvement
FGSM (ε=1/255)	65.3%	17.0%	+48.4%
FGSM (ε=2/255)	85.2%	35.5%	+49.7%
PGD (ε=1/255)	72.6%	18.5%	+54.1%
PGD (ε=2/255)	96.4%	42.9%	+53.5%
Partial (avg)	73.7%	40.4%	+33.4%
Patch (avg)	15.9%	5.9%	+10.0%
Defense consistently reduces attack success rate by 10–54% across all threat surfaces.

🚧 Project Status
Phase	HAM10000	ISIC 2019	BCN20000	MILK10k	Combined
Baseline training	✅	✅	✅	✅	✅
Attack evaluation	✅	✅	⏳	✅	⏳
Adversarial training	—	⏳	✅	✅	✅
Defended attack eval	—	⏳	⏳	✅	⏳
MDDA defense	—	—	—	—	⏳
Legend: ✅ Done · ⏳ In progress · — Not applicable

🔬 Research Contributions
Reproducible multi-dataset pipeline — Same architecture and hyperparameters across four dermatology datasets, enabling fair cross-dataset comparison.

Systematic threat model — Full-image, partial-image, and patch attacks at six perturbation levels reveal that patch attacks are markedly less effective than expected.

Multi-attack adversarial training — Random attack sampling during training produces a defense robust against six different attack vectors.

(In progress) MDDA defense — Extending adversarial training with domain alignment to improve robustness against distribution-shift attacks.

📚 References
Huang, G., et al. (2017). Densely Connected Convolutional Networks. CVPR.

Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. ECCV.

Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV.

Goodfellow, I. J., et al. (2014). Explaining and Harnessing Adversarial Examples. ICLR.

Madry, A., et al. (2017). Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR.

Tschandl, P., et al. (2018). The HAM10000 Dataset. Scientific Data.

Combalia, M., et al. (2019). BCN20000: Dermoscopic Lesions in the Wild. arXiv:1908.02288.

Daneshjou, R., et al. (2022). Disparities in Dermatology AI Performance. Science Advances.
