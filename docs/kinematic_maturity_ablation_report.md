# Kinematic Maturity Gate & Physical Validity Benchmark Report
**Evaluated Videos**: 15 per split | **Ground Truth Window**: $\pm 35$ frames (~1.17s) | **Episode Clustering**: $\Delta f \le 25$ frames

## 1. Episode-Level Deduplicated Performance Comparison
### Split: `split_in_distribution`
| Configuration | TP | FP (Ep) | FN | Precision | Recall | F1 Score | FP / Video | Med Lead Time | Med Abs Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Untouched Multi-Signal)** | 2 | 15 | 13 | **0.118** | **0.133** | **0.125** | 1.00 | +0.83s (+4f) | 24 frames |
| **Stage 6 (Spatial Continuity)** | 2 | 9 | 13 | **0.182** | **0.133** | **0.154** | 0.60 | +-0.09s (-2f) | 2 frames |
| **Stage 6 + Track Maturity (age >= 3f)** | 2 | 9 | 13 | **0.182** | **0.133** | **0.154** | 0.60 | +-0.09s (-2f) | 2 frames |
| **Stage 6 + Maturity + Kinematic Validity (Warm-up & Physical Bounds)** | 2 | 6 | 13 | **0.250** | **0.133** | **0.174** | 0.40 | +-0.09s (-2f) | 2 frames |

### Split: `split_geo_aware`
| Configuration | TP | FP (Ep) | FN | Precision | Recall | F1 Score | FP / Video | Med Lead Time | Med Abs Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Untouched Multi-Signal)** | 1 | 10 | 14 | **0.091** | **0.067** | **0.077** | 0.67 | +2.80s (+28f) | 28 frames |
| **Stage 6 (Spatial Continuity)** | 1 | 12 | 14 | **0.077** | **0.067** | **0.071** | 0.80 | +-0.13s (-4f) | 4 frames |
| **Stage 6 + Track Maturity (age >= 3f)** | 1 | 12 | 14 | **0.077** | **0.067** | **0.071** | 0.80 | +-0.13s (-4f) | 4 frames |
| **Stage 6 + Maturity + Kinematic Validity (Warm-up & Physical Bounds)** | 1 | 7 | 14 | **0.125** | **0.067** | **0.087** | 0.47 | +-0.13s (-4f) | 4 frames |

## 2. Impact of Deduplication: Episode Clustering vs. Raw Frame Triggers
| Split | Configuration | Raw FP | Raw FP / Vid | Raw Prec | Ep FP | Ep FP / Vid | **Episode Precision** | Precision Gain |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **In-Dist** | **Baseline (Untouched Multi-Signal)** | 2354 | 156.9 | 0.001 | 15 | 1.00 | **0.118** | **138.6x** |
| **In-Dist** | **Stage 6 (Spatial Continuity)** | 22 | 1.5 | 0.083 | 9 | 0.60 | **0.182** | **2.2x** |
| **In-Dist** | **Stage 6 + Track Maturity (age >= 3f)** | 22 | 1.5 | 0.083 | 9 | 0.60 | **0.182** | **2.2x** |
| **In-Dist** | **Stage 6 + Maturity + Kinematic Validity (Warm-up & Physical Bounds)** | 9 | 0.6 | 0.182 | 6 | 0.40 | **0.250** | **1.4x** |
| **Geo-Aware** | **Baseline (Untouched Multi-Signal)** | 3417 | 227.8 | 0.000 | 10 | 0.67 | **0.091** | **310.7x** |
| **Geo-Aware** | **Stage 6 (Spatial Continuity)** | 30 | 2.0 | 0.032 | 12 | 0.80 | **0.077** | **2.4x** |
| **Geo-Aware** | **Stage 6 + Track Maturity (age >= 3f)** | 30 | 2.0 | 0.032 | 12 | 0.80 | **0.077** | **2.4x** |
| **Geo-Aware** | **Stage 6 + Maturity + Kinematic Validity (Warm-up & Physical Bounds)** | 10 | 0.7 | 0.091 | 7 | 0.47 | **0.125** | **1.4x** |

## 3. False Positive Failure Mode Taxonomy Breakdown (Episodes)
| Split | Configuration | Normal Following | Lane Merge | Crossing Traffic | Stopped Traffic | Tracker Artifact | Calibration Artifact | Other | Total FP Ep |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **In-Dist** | **Baseline (Untouched Multi-Signal)** | 0 | 0 | 0 | 0 | 13 | 2 | 0 | **15** |
| **In-Dist** | **Stage 6 (Spatial Continuity)** | 0 | 0 | 0 | 0 | 8 | 1 | 0 | **9** |
| **In-Dist** | **Stage 6 + Track Maturity (age >= 3f)** | 0 | 0 | 0 | 0 | 8 | 1 | 0 | **9** |
| **In-Dist** | **Stage 6 + Maturity + Kinematic Validity (Warm-up & Physical Bounds)** | 0 | 0 | 0 | 0 | 6 | 0 | 0 | **6** |
| **Geo-Aware** | **Baseline (Untouched Multi-Signal)** | 0 | 0 | 0 | 0 | 8 | 2 | 0 | **10** |
| **Geo-Aware** | **Stage 6 (Spatial Continuity)** | 0 | 0 | 0 | 0 | 11 | 1 | 0 | **12** |
| **Geo-Aware** | **Stage 6 + Track Maturity (age >= 3f)** | 0 | 0 | 0 | 0 | 11 | 1 | 0 | **12** |
| **Geo-Aware** | **Stage 6 + Maturity + Kinematic Validity (Warm-up & Physical Bounds)** | 0 | 0 | 0 | 0 | 7 | 0 | 0 | **7** |