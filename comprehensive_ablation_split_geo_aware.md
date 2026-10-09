# Collision Candidate Ablation & False Positive Diagnostic Benchmark
**Split Type**: `split_geo_aware` (`test`) | **Videos Evaluated**: 15 | **Ground Truth Tolerance**: $\pm 35$ frames (~1.17s)

## 1. Episode-Level Deduplicated Performance Comparison
| Configuration | TP | FP (Ep) | FN | Precision | Recall | F1 Score | FP / Video | Med Lead Time | Med Abs Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Multi-Signal Active Count)** | 1 | 10 | 14 | **0.091** | **0.067** | **0.077** | 0.67 | +2.80s (+28f) | 28 frames |
| **Stage 1: Distance Only (<= 3.5m)** | 0 | 13 | 15 | **0.000** | **0.000** | **0.000** | 0.87 | +0.00s (+0f) | 0 frames |
| **Stage 2: Distance + IoU (<= 3.5m & IoU >= 0.15)** | 0 | 12 | 15 | **0.000** | **0.000** | **0.000** | 0.80 | +0.00s (+0f) | 0 frames |
| **Stage 3: Distance + Convergence (Dist<=4.0m, TCA<2s)** | 2 | 12 | 13 | **0.143** | **0.133** | **0.138** | 0.80 | +-1.64s (-17f) | 17 frames |
| **Stage 4: Distance + Convergence + Decel (a<-3.5m/s2)** | 2 | 13 | 13 | **0.133** | **0.133** | **0.133** | 0.87 | +-1.66s (-18f) | 18 frames |
| **Stage 5: Dist + Conv + Decel + Track-ID Persistence (>=2f)** | 1 | 12 | 14 | **0.077** | **0.067** | **0.071** | 0.80 | +-0.13s (-4f) | 4 frames |
| **Stage 6: Dist + Conv + Decel + Spatial Continuity (>=2f)** | 1 | 12 | 14 | **0.077** | **0.067** | **0.071** | 0.80 | +-0.13s (-4f) | 4 frames |

## 2. Impact of Event Deduplication: Episode Clustering vs. Raw Frame Triggers
| Configuration | Raw FP | Raw FP / Vid | Raw Precision | Ep FP | Ep FP / Vid | **Episode Precision** | Precision Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Multi-Signal Active Count)** | 3417 | 227.8 | 0.000 | 10 | 0.67 | **0.091** | **310.7x** |
| **Stage 1: Distance Only (<= 3.5m)** | 280 | 18.7 | 0.000 | 13 | 0.87 | **0.000** | **1.0x** |
| **Stage 2: Distance + IoU (<= 3.5m & IoU >= 0.15)** | 275 | 18.3 | 0.000 | 12 | 0.80 | **0.000** | **1.0x** |
| **Stage 3: Distance + Convergence (Dist<=4.0m, TCA<2s)** | 108 | 7.2 | 0.018 | 12 | 0.80 | **0.143** | **7.9x** |
| **Stage 4: Distance + Convergence + Decel (a<-3.5m/s2)** | 64 | 4.3 | 0.030 | 13 | 0.87 | **0.133** | **4.4x** |
| **Stage 5: Dist + Conv + Decel + Track-ID Persistence (>=2f)** | 22 | 1.5 | 0.043 | 12 | 0.80 | **0.077** | **1.8x** |
| **Stage 6: Dist + Conv + Decel + Spatial Continuity (>=2f)** | 30 | 2.0 | 0.032 | 12 | 0.80 | **0.077** | **2.4x** |

## 3. False Positive Failure Mode Taxonomy Breakdown (Episodes)
| Configuration | Normal Following | Lane Merge | Crossing Traffic | Stopped Traffic | Tracker Artifact | Calibration Artifact | Other | Total FP Ep |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Multi-Signal Active Count)** | 0 | 0 | 0 | 0 | 8 | 2 | 0 | **10** |
| **Stage 1: Distance Only (<= 3.5m)** | 0 | 0 | 0 | 0 | 10 | 3 | 0 | **13** |
| **Stage 2: Distance + IoU (<= 3.5m & IoU >= 0.15)** | 0 | 0 | 0 | 0 | 9 | 3 | 0 | **12** |
| **Stage 3: Distance + Convergence (Dist<=4.0m, TCA<2s)** | 0 | 0 | 0 | 0 | 12 | 0 | 0 | **12** |
| **Stage 4: Distance + Convergence + Decel (a<-3.5m/s2)** | 0 | 0 | 0 | 0 | 12 | 1 | 0 | **13** |
| **Stage 5: Dist + Conv + Decel + Track-ID Persistence (>=2f)** | 0 | 0 | 0 | 0 | 11 | 1 | 0 | **12** |
| **Stage 6: Dist + Conv + Decel + Spatial Continuity (>=2f)** | 0 | 0 | 0 | 0 | 11 | 1 | 0 | **12** |