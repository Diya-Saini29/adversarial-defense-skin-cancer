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

```text
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
```

**Why this pipeline?**
- **CLAHE** enhances local contrast, critical for subtle lesion boundaries.
- **CBAM** lets the model attend to clinically relevant lesion regions rather than background skin.
- **Focal Loss** counteracts the severe class imbalance (e.g., NV ≈ 67% in HAM10000).
- **WeightedRandomSampler** further balances the training distribution.

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

## ⚔️ Phase 2: Adversarial Attacks (Red Team)

The attack suite is designed to systematically evaluate how easily a classifier can be **fooled** by imperceptible perturbations.

### 🎯 Threat Model

| Property | Value |
|----------|-------|
| **Adversary Knowledge** | White-box (full access to model gradients) |
| **Adversary Goal** | Untargeted misclassification (any wrong class) |
| **Perturbation Constraint** | L∞ bounded by ε |
| **Attack Types** | FGSM (single-step), PGD (iterative) |
| **Regions Attacked** | Full, top-half, bottom-half, localized patches |

### 🧪 Attack Suite

| Attack | Type | Iterations | Description |
|--------|------|------------|-------------|
| **FGSM** | Single-step | 1 | Fast Gradient Sign Method — fast baseline attack |
| **PGD** | Iterative | 10 | Projected Gradient Descent — strong attack |
| **Spatial** | Masked | — | Full / top-half / bottom-half / patch-based |

### 🗺️ Threat Surfaces

- **Full-image attacks** — perturbations across the entire image
- **Partial-image attacks** — perturbations restricted to top or bottom 50%
- **Patch attacks** — perturbations localized to 16×16, 32×32, 48×48, or 64×64 patches

### 📐 Perturbation Budgets (ε)

`1/255, 2/255, 4/255, 8/255, 16/255` — measuring model vulnerability across a range of imperceptible perturbations.

### 📉 Attack Configuration

```python
# FGSM
fgsm_attack(model, images, labels, epsilon)

# PGD
pgd_attack(model, images, labels, epsilon, alpha=epsilon/4, num_iter=10)

# Spatial masking
apply_masked_attack(model, images, labels, mask, attack_fn, ...)
```

### 🔍 Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| **Attack Success Rate (ASR)** | % of correctly-classified inputs that flip to wrong label under attack |
| **Robust Accuracy** | Accuracy on adversarially-perturbed inputs |
| **Accuracy Drop** | Clean accuracy − Robust accuracy |
| **Confidence Drop** | Mean (clean confidence − adversarial confidence) |
| **L∞ / L₂ Norm** | Mean perturbation magnitude |

### ⚠️ Key Findings

| Dataset | FGSM ASR (full, ε=8/255) | PGD ASR (full, ε=8/255) |
|---------|--------------------------|--------------------------|
| HAM10000 | 97.5% | **100%** |
| ISIC 2019 | 98.5% | **100%** |
| MILK10k | 99.6% | **100%** |

**🚨 All baseline models are catastrophically vulnerable to full-image PGD attacks (100% ASR).**

**📌 Notable observations:**

- **PGD > FGSM** — iterative attacks are consistently more effective than single-step
- **Partial-image attacks** achieve 58–80% ASR — a significant portion of the image can be corrupted with little detection
- **Patch attacks are surprisingly ineffective** (13–20% ASR) — CBAM's spatial attention appears to provide natural robustness to localized perturbations
- **Confidence drop is negative for high-ε attacks** — the model becomes *overconfident* in its wrong predictions, a well-known symptom of adversarial ML vulnerability

---

## 🛡️ Phase 3: Defense Mechanisms (Blue Team)

### 🥇 Defense 1: Multi-Attack Adversarial Training (Completed on 3/4 datasets)

The primary defense strategy retrains the model on **adversarially-perturbed examples** generated on-the-fly during training.

#### 🎛️ Training Recipe (identical across all datasets)

```python
MULTI_ATTACK_CONFIG = {
    'num_epochs': 20,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-3,
    'epsilon': 4/255,           # training perturbation budget
    'pgd_iter': 7,               # PGD iterations during training
    'pgd_alpha': 1/255,
    'clean_loss_weight': 0.3,    # mixed clean/adv loss
    'adv_loss_weight': 0.7,
    'early_stop_patience': 6,
}
```

#### 🎲 Multi-Attack Sampling Strategy

At **every training batch**, one of **six attack types** is randomly selected with uniform probability:

```
1. FGSM full-image
2. PGD full-image
3. PGD top-half
4. PGD bottom-half
5. PGD center patch (112×112)
6. PGD top-center patch (112×112)
```

This forces the model to be robust against **a distribution of threats** rather than a single attack — critically important for real-world deployment where the attack type is unknown.

#### 🔄 Training Loop

```
For each batch (x, y):
    1. Compute clean loss:       L_clean = Focal(model(x), y)
    2. Sample random attack type: a ~ Uniform{6 attacks}
    3. Generate adversarial:      x_adv = Attack_a(model, x, y)
    4. Compute adversarial loss:  L_adv = Focal(model(x_adv), y)
    5. Combined loss:             L = 0.3·L_clean + 0.7·L_adv
    6. Backpropagate and update
```

#### 📊 Defense Results

| Dataset | Full-Image ASR (avg) | Partial-Image ASR | Patch ASR |
|---------|---------------------|-------------------|-----------|
| **MILK10k** | 91.55% → **61.43%** (Δ +30.1%) | 73.72% → **40.37%** (Δ +33.4%) | 15.88% → **5.89%** (Δ +10.0%) |
| **Combined BCN+MILK** | Evaluated below ⬇️ | Evaluated below ⬇️ | Evaluated below ⬇️ |
| **ISIC 2019** | *training in progress* | *pending* | *pending* |

**Best-case improvement (MILK10k):** PGD at ε=1/255: **ASR 72.6% → 18.5%** (Δ **+54.1%**)

#### 🏆 MILK10k — Detailed Before/After Comparison

| Attack | Baseline ASR | Defended ASR | Δ Improvement |
|--------|--------------|--------------|---------------|
| FGSM (ε=1/255) | 65.3% | 17.0% | **+48.4%** |
| FGSM (ε=2/255) | 85.2% | 35.5% | **+49.7%** |
| FGSM (ε=4/255) | 96.4% | 57.1% | **+39.2%** |
| FGSM (ε=8/255) | 99.6% | 79.9% | **+19.7%** |
| PGD (ε=1/255) | 72.6% | 18.5% | **+54.1%** |
| PGD (ε=2/255) | 96.4% | 42.9% | **+53.5%** |
| PGD (ε=4/255) | 100% | 74.5% | **+25.5%** |
| Partial-image (avg) | 73.7% | 40.4% | **+33.4%** |
| Patch (avg) | 15.9% | 5.9% | **+10.0%** |

**Interpretation:**
- **Large gains** at low perturbation budgets (ε=1/255, 2/255) — model learns to resist subtle attacks
- **Dwindling gains** at high ε — some attacks (ε=16/255) are impossible to defend without trade-offs
- **Partial + patch attacks** gain meaningfully, showing the defense generalizes to non-training threat surfaces

#### 🏆 Combined BCN+MILK — Defense Training Summary

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | **69.65%** (Epoch 18) |
| Val Macro F1 at best | 0.6260 |
| Epochs run | 20/20 |
| Attack usage balance | 15.4% – 19.7% per attack type |
| Early stopping triggered | No (patience 2/6) |

> **Note:** The defended Combined model outperformed its clean-trained baseline on val accuracy (69.65% vs ~65%), suggesting multi-attack training also acts as a **regularizer**.

**⏳ Defended-model attack evaluation on Combined is currently in progress.**

---

### 🥈 Defense 2: MDDA — Multi-Domain Distribution Alignment (In Progress)

While adversarial training hardens the model against **input-space perturbations**, it does not address **distribution-shift attacks** — where the test distribution diverges from the training distribution. **MDDA extends the defense pipeline** to close this gap.

#### 🎯 Motivation

- **BCN and MILK** originate from different clinics, imaging devices, and patient populations → domain shift
- Adversarial robustness alone does not guarantee **cross-domain robustness**
- MDDA learns **domain-invariant features** that are robust to both adversarial perturbations *and* distribution shift

#### 🏗️ Extended Defense Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   DEFENSE PIPELINE                          │
│                                                             │
│  ┌───────────────────┐         ┌───────────────────────┐    │
│  │  DEFENSE 1:       │         │  DEFENSE 2:           │    │
│  │  Adversarial      │   →     │  MDDA                 │    │
│  │  Training         │         │  (Multi-Domain        │    │
│  │  (Completed)      │         │   Distribution        │    │
│  │                   │         │   Alignment)          │    │
│  │  • PGD-AT         │         │                       │    │
│  │  • Multi-attack   │         │  • Domain-invariant   │    │
│  │  • Mixed loss     │         │    feature learning   │    │
│  │                   │         │  • Distribution align │    │
│  └───────────────────┘         └───────────────────────┘    │
│                                                             │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  UNIFIED DEFENDED MODEL                                     │
│  Robust to:                                                 │
│  • Adversarial perturbations (FGSM, PGD, partial, patch)    │
│  • Cross-domain distribution shift                          │
└─────────────────────────────────────────────────────────────┘
```

#### 🎛️ MDDA Design Goals

| Goal | Mechanism |
|------|-----------|
| Domain-invariant features | Align feature distributions between BCN and MILK source domains |
| Cross-domain robustness | Reduce performance gap on held-out target domain |
| Adversarial robustness | Combine with multi-attack adversarial training |
| Generalization | Preserve clean accuracy while improving robust accuracy |

#### 📅 MDDA Roadmap

| Step | Status |
|------|--------|
| Domain characterization (BCN vs MILK feature analysis) | ⏳ Planned |
| MDDA loss design (alignment + classification) | ⏳ Planned |
| Joint training (adv + MDDA objective) | ⏳ Planned |
| Full attack suite evaluation | ⏳ Planned |
| Cross-dataset comparison | ⏳ Planned |

#### 🎯 Expected Contribution

A **unified defense framework** combining adversarial training and multi-domain distribution alignment — evaluated against the same FGSM/PGD attack suite as Defense 1, enabling a fair **baseline vs adv-trained vs MDDA** comparison across all four datasets.

---

## 🔬 Phase 4: Comparative Security Study (In Progress)

The final phase assembles all results into a **cross-dataset robustness comparison** suitable for publication.

### 📊 Comparison Framework

For each dataset and each model variant (Baseline / Adv-Trained / MDDA), the following are measured:

| Metric Category | Metrics |
|-----------------|---------|
| **Clean Performance** | Accuracy, Macro F1, Weighted F1 |
| **Full-Image Robustness** | ASR, Robust Acc, Confidence Drop (at ε ∈ {1,2,4,8,16}/255) |
| **Partial-Image Robustness** | ASR, Robust Acc (top/bottom half) |
| **Patch Robustness** | ASR, Robust Acc (16×16 → 64×64) |
| **Per-Class Robustness** | Per-class recall/precision under attack |
| **Computational Cost** | Training time, inference latency |

### 🎯 Target Outputs

- **Robustness curves** — ASR vs ε for FGSM and PGD, per dataset
- **Model-vs-attack heatmap** — which attacks defeat which defenses
- **Cross-dataset table** — how defense effectiveness transfers across datasets
- **Security report** — actionable robustness recommendations for medical AI deployment
---

## 🛠️ Technologies & Frameworks

| Category | Tools |
|----------|-------|
| **Deep Learning** | PyTorch, Torchvision |
| **Attention** | CBAM (custom implementation) |
| **Loss** | Focal Loss (custom implementation) |
| **Preprocessing** | OpenCV (CLAHE), Pillow |
| **Metrics** | Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Google Colab, Kaggle (T4/P100) |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/adversarial-dermatology.git
cd adversarial-dermatology
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Datasets

Datasets are not included in the repo due to size. Download from:

- **HAM10000:** [ISIC Archive](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- **ISIC 2019:** [ISIC Challenge 2019](https://challenge2019.isic-archive.com/)
- **BCN20000:** [Figshare](https://figshare.com/articles/journal_contribution/BCN20000_Dermoscopic_Lesions_in_the_Wild/24140028)
- **MILK10k:** [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FSXRAQ)

Place them under `data/` following the directory structure above.

### 4. Run Baseline Training

Open `notebooks/04_MILK10k_baseline.ipynb` and run all cells. Replace `DATASET_PATH` with your local path.

### 5. Run Attack Evaluation

Open `notebooks/06_attack_evaluation.ipynb`. Load a trained checkpoint and run the full attack suite.

### 6. Run Adversarial Training

Open `notebooks/07_adversarial_training.ipynb`. Trains a defended model using the multi-attack objective.

### 7. Run MDDA Defense (in progress)

Open `notebooks/08_mdda_defense.ipynb` once Defense 1 completes across all datasets.

---

## 📈 Results Summary

### Baseline vs Defended (MILK10k) — Best Case

| Attack | Baseline ASR | Defended ASR | Δ Improvement |
|--------|--------------|--------------|---------------|
| FGSM (ε=1/255) | 65.3% | 17.0% | **+48.4%** |
| FGSM (ε=2/255) | 85.2% | 35.5% | **+49.7%** |
| PGD (ε=1/255) | 72.6% | 18.5% | **+54.1%** |
| PGD (ε=2/255) | 96.4% | 42.9% | **+53.5%** |
| Partial (avg) | 73.7% | 40.4% | **+33.4%** |
| Patch (avg) | 15.9% | 5.9% | **+10.0%** |

**Defense consistently reduces attack success rate by 10–54%** across all threat surfaces.

---

## 🚧 Project Status

| Phase | HAM10000 | ISIC 2019 | BCN20000 | MILK10k | Combined |
|-------|----------|-----------|----------|---------|----------|
| Baseline training | ✅ | ✅ | ✅ | ✅ | ✅ |
| Attack evaluation (baseline) | ✅ | ✅ | ⏳ | ✅ | ⏳ |
| Adversarial training (Defense 1) | — | ⏳ | ✅ | ✅ | ✅ |
| Defended attack eval | — | ⏳ | ⏳ | ✅ | ⏳ |
| MDDA defense (Defense 2) | — | — | — | — | ⏳ |
| Cross-dataset comparison | — | — | — | — | ⏳ |

**Legend:** ✅ Done · ⏳ In progress · — Not applicable

---

## 🔬 Research Contributions

1. **Reproducible multi-dataset pipeline** — Same architecture and hyperparameters across four dermatology datasets, enabling fair cross-dataset comparison.
2. **Systematic threat model** — Full-image, partial-image, and patch attacks at six perturbation levels reveal that patch attacks are markedly less effective than expected.
3. **Multi-attack adversarial training** — Random attack sampling during training produces a defense robust against six different attack vectors.
4. **(In progress) MDDA defense** — Extending adversarial training with domain alignment to improve robustness against distribution-shift attacks.
5. **(Targeted) Unified security report** — Cross-dataset robustness comparison of baseline vs adv-trained vs MDDA defenses.

---

## 📚 References

1. Huang, G., et al. (2017). *Densely Connected Convolutional Networks.* CVPR.
2. Woo, S., et al. (2018). *CBAM: Convolutional Block Attention Module.* ECCV.
3. Lin, T. Y., et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.
4. Goodfellow, I. J., et al. (2014). *Explaining and Harnessing Adversarial Examples.* ICLR.
5. Madry, A., et al. (2017). *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR.
6. Tschandl, P., et al. (2018). *The HAM10000 Dataset.* Scientific Data.
7. Combalia, M., et al. (2019). *BCN20000: Dermoscopic Lesions in the Wild.* arXiv:1908.02288.
8. Daneshjou, R., et al. (2022). *Disparities in Dermatology AI Performance.* Science Advances.

---

## 📬 Contact

**[Your Name]**
- Email: [your.email@example.com]
- LinkedIn: [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)
- GitHub: [github.com/your-username](https://github.com/your-username)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Datasets provided by ISIC Archive, Harvard Dataverse, and Figshare
- Compute resources from Google Colab and Kaggle
- Architecture inspired by prior work in dermatology AI robustness

---

⭐ **If you find this project useful, please consider giving it a star!**
