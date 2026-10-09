# Collision Candidate Signal Ablation Benchmark Results
**Split**: `split_in_distribution` (`test`) | **Videos Evaluated**: 15

## 1. Episode-Deduplicated Performance Comparison
| Configuration | TP | FP (Ep) | FN | Precision | Recall | F1 Score | FP / Video | Med Lead Time | Med Abs Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Current Multi-Signal)** | 5 | 39 | 10 | **0.114** | **0.333** | **0.169** | 2.60 | +-1.09s (-19f) | 27 frames |
| **Stage 1: Distance Only (<= 3.5m)** | 1 | 10 | 14 | **0.091** | **0.067** | **0.077** | 0.67 | +-0.12s (-3f) | 3 frames |
| **Stage 2: Distance + IoU (<= 3.5m & IoU >= 0.15)** | 1 | 9 | 14 | **0.100** | **0.067** | **0.080** | 0.60 | +-0.12s (-3f) | 3 frames |
| **Stage 3: Distance + Convergence (Closing, TCA < 2s)** | 2 | 11 | 13 | **0.154** | **0.133** | **0.143** | 0.73 | +-0.03s (-1f) | 1 frames |
| **Stage 4: Distance + Convergence + Decel (a < -3m/s2)** | 1 | 11 | 14 | **0.083** | **0.067** | **0.074** | 0.73 | +-0.04s (-1f) | 1 frames |
| **Stage 5: Dist + Conv + Decel + Persistence (>= 2 frames)** | 0 | 10 | 15 | **0.000** | **0.000** | **0.000** | 0.67 | +0.00s (+0f) | 0 frames |

## 2. Impact of Event Deduplication: Episode-Level vs. Raw Trigger Noise
| Configuration | Raw FP | Raw FP / Vid | Raw Precision | Ep FP | Ep FP / Vid | **Episode Precision** | Precision Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Current Multi-Signal)** | 170 | 11.3 | 0.029 | 39 | 2.60 | **0.114** | **4.0x** |
| **Stage 1: Distance Only (<= 3.5m)** | 234 | 15.6 | 0.004 | 10 | 0.67 | **0.091** | **21.4x** |
| **Stage 2: Distance + IoU (<= 3.5m & IoU >= 0.15)** | 229 | 15.3 | 0.004 | 9 | 0.60 | **0.100** | **23.0x** |
| **Stage 3: Distance + Convergence (Closing, TCA < 2s)** | 88 | 5.9 | 0.022 | 11 | 0.73 | **0.154** | **6.9x** |
| **Stage 4: Distance + Convergence + Decel (a < -3m/s2)** | 51 | 3.4 | 0.019 | 11 | 0.73 | **0.083** | **4.3x** |
| **Stage 5: Dist + Conv + Decel + Persistence (>= 2 frames)** | 17 | 1.1 | 0.000 | 10 | 0.67 | **0.000** | **1.0x** |