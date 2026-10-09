# Collision Candidate Ablation & False Positive Diagnostic Benchmark
**Split Type**: `split_in_distribution` (`test`) | **Videos Evaluated**: 15 | **Ground Truth Tolerance**: $\pm 35$ frames (~1.17s)

## 1. Episode-Level Deduplicated Performance Comparison
| Configuration | TP | FP (Ep) | FN | Precision | Recall | F1 Score | FP / Video | Med Lead Time | Med Abs Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Multi-Signal Active Count)** | 2 | 15 | 13 | **0.118** | **0.133** | **0.125** | 1.00 | +0.83s (+4f) | 24 frames |
| **Stage 1: Distance Only (<= 3.5m)** | 1 | 10 | 14 | **0.091** | **0.067** | **0.077** | 0.67 | +-0.12s (-3f) | 3 frames |
| **Stage 2: Distance + IoU (<= 3.5m & IoU >= 0.15)** | 1 | 9 | 14 | **0.100** | **0.067** | **0.080** | 0.60 | +-0.12s (-3f) | 3 frames |
| **Stage 3: Distance + Convergence (Dist<=4.0m, TCA<2s)** | 2 | 11 | 13 | **0.154** | **0.133** | **0.143** | 0.73 | +-0.03s (-1f) | 1 frames |
| **Stage 4: Distance + Convergence + Decel (a<-3.5m/s2)** | 2 | 11 | 13 | **0.154** | **0.133** | **0.143** | 0.73 | +-0.05s (-2f) | 2 frames |
| **Stage 5: Dist + Conv + Decel + Track-ID Persistence (>=2f)** | 2 | 9 | 13 | **0.182** | **0.133** | **0.154** | 0.60 | +-0.09s (-2f) | 2 frames |
| **Stage 6: Dist + Conv + Decel + Spatial Continuity (>=2f)** | 2 | 9 | 13 | **0.182** | **0.133** | **0.154** | 0.60 | +-0.09s (-2f) | 2 frames |

## 2. Impact of Event Deduplication: Episode Clustering vs. Raw Frame Triggers
| Configuration | Raw FP | Raw FP / Vid | Raw Precision | Ep FP | Ep FP / Vid | **Episode Precision** | Precision Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Multi-Signal Active Count)** | 2354 | 156.9 | 0.001 | 15 | 1.00 | **0.118** | **138.6x** |
| **Stage 1: Distance Only (<= 3.5m)** | 234 | 15.6 | 0.004 | 10 | 0.67 | **0.091** | **21.4x** |
| **Stage 2: Distance + IoU (<= 3.5m & IoU >= 0.15)** | 229 | 15.3 | 0.004 | 9 | 0.60 | **0.100** | **23.0x** |
| **Stage 3: Distance + Convergence (Dist<=4.0m, TCA<2s)** | 89 | 5.9 | 0.022 | 11 | 0.73 | **0.154** | **7.0x** |
| **Stage 4: Distance + Convergence + Decel (a<-3.5m/s2)** | 53 | 3.5 | 0.036 | 11 | 0.73 | **0.154** | **4.2x** |
| **Stage 5: Dist + Conv + Decel + Track-ID Persistence (>=2f)** | 16 | 1.1 | 0.111 | 9 | 0.60 | **0.182** | **1.6x** |
| **Stage 6: Dist + Conv + Decel + Spatial Continuity (>=2f)** | 22 | 1.5 | 0.083 | 9 | 0.60 | **0.182** | **2.2x** |

## 3. False Positive Failure Mode Taxonomy Breakdown (Episodes)
| Configuration | Normal Following | Lane Merge | Crossing Traffic | Stopped Traffic | Tracker Artifact | Calibration Artifact | Other | Total FP Ep |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Multi-Signal Active Count)** | 0 | 0 | 0 | 0 | 13 | 2 | 0 | **15** |
| **Stage 1: Distance Only (<= 3.5m)** | 0 | 0 | 0 | 0 | 8 | 2 | 0 | **10** |
| **Stage 2: Distance + IoU (<= 3.5m & IoU >= 0.15)** | 0 | 0 | 0 | 0 | 7 | 2 | 0 | **9** |
| **Stage 3: Distance + Convergence (Dist<=4.0m, TCA<2s)** | 0 | 0 | 0 | 0 | 11 | 0 | 0 | **11** |
| **Stage 4: Distance + Convergence + Decel (a<-3.5m/s2)** | 0 | 0 | 0 | 0 | 10 | 1 | 0 | **11** |
| **Stage 5: Dist + Conv + Decel + Track-ID Persistence (>=2f)** | 0 | 0 | 0 | 0 | 8 | 1 | 0 | **9** |
| **Stage 6: Dist + Conv + Decel + Spatial Continuity (>=2f)** | 0 | 0 | 0 | 0 | 8 | 1 | 0 | **9** |