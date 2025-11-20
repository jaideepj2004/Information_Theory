# ISIB — Improved Structured Information Bottleneck

> **Project**: Improved Structured Information Bottleneck (ISIB)  
> **Authors / Team**: Jaideep Jaiswal (202211032), Ayush Gupta (202211007), Sanskar Koserval (202211077)  
> **Supervisor**: Dr. Jignesh Patel  
> **Description**: ISIB extends the Structured Information Bottleneck (SIB) idea by adding multi-encoders, attention-based fusion, and supervised contrastive learning. It improves representation quality and classification accuracy across MNIST, Fashion-MNIST, SVHN, and CIFAR-10.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Highlights / Contributions](#highlights--contributions)  
- [Results Summary](#results-summary)  
- [Repository Structure](#repository-structure)  
- [Requirements](#requirements)  
- [Quick Start](#quick-start)  
  - [1. Hyperparameter Search (HPO)](#1-hyperparameter-search-hpo)  
  - [2. Extended Training (final model)](#2-extended-training-final-model)  
  - [3. Evaluate / Inference](#3-evaluate--inference)  
- [Important File Paths and Outputs](#important-file-paths-and-outputs)  
- [Design / Algorithm Summary](#design--algorithm-summary)  
- [Tips & Troubleshooting](#tips--troubleshooting)  
- [References / Base Paper](#references--base-paper)  
- [License & Contact](#license--contact)

---

## Project Overview

ISIB addresses four practical limitations identified in the base SIB paper:

1. **Approximation errors** in prior I(X;Z) estimators — we use VAE + KL for Gaussian latents to reduce estimator bias.  
2. **Information loss** due to single encoder compression — we introduce multi-encoders (main + 2 auxiliary) to preserve complementary information.  
3. **Redundancy** among auxiliary encoders — attention-based fusion dynamically weights encoders so redundant features get low weight.  
4. **Poor accuracy / efficiency** on harder datasets — improved architecture, training recipe and SupCon yield stronger accuracy with modest parameter count.

Architectural highlights:
- Three encoders: `main (32-d)`, `aux1 (16-d)`, `aux2 (16-d)`  
- Attention Fusion: query from main, key/value from concatenation of all latents  
- Classifier head + projection head (for supervised contrastive loss)  
- Decoder for reconstruction (VAE-style) and KL regularization  
- Loss = CE + β * KL + λ * SupCon (+ optional reconstruction term)

---

## Highlights / Contributions

- Multi-encoder latent representation (multi-view)  
- Attention-based fusion to adaptively combine latents  
- Supervised contrastive loss (SupCon) for tighter class clusters  
- Practical training recipe: cosine LR, Adam / AdamW, grad clipping, data augmentations  
- Full HPO over β and learning rate; extended training on best configs  
- Evaluated on MNIST, Fashion-MNIST, SVHN, CIFAR-10 with improved accuracies

---

## Results Summary

| Dataset        | Best Accuracy | Best Epoch | β     | LR      | λ_supcon |
|---------------:|--------------:|-----------:|------:|--------:|---------:|
| MNIST          | 99.54%        | 75         | 0.015 | 0.0008  | 0.5      |
| Fashion-MNIST  | 92.47%        | 20 (HPO)   | 0.005 | 0.001   | 0.5      |
| SVHN           | 94.81%        | 75         | 0.01  | 0.0008  | 0.5      |
| CIFAR-10       | 85.18%        | 99         | 0.007 | 0.001   | 0.5      |

> Note: Results (models, JSON, history `.npz`) are saved under `/kaggle/working/` in the experiment scripts.

---

## Repository Structure (suggested)

