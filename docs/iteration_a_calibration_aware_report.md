# Iteration A: Calibration-Aware Collision Geometry Benchmark Report
**Evaluated Splits**: `split_in_distribution` (15 videos) vs `split_geo_aware` (15 videos) | **GT Window**: $\pm 35$ frames (~1.17s)

## 1. Primary Incident Detection Metrics
### Split: `split_in_distribution`
| Configuration | TP | FP (Ep) | FN | Precision | Recall | F1 Score | FP / Vid | Raw FP / Vid | Med Lead Time | Med Abs Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Untouched Multi-Signal)** | 2 | 15 | 13 | **0.118** | **0.133** | **0.125** | 1.00 | 156.9 | +0.83s (+4f) | 24 frames |
| **Previous Best (Fixed Metric Distance Gate)** | 2 | 3 | 13 | **0.400** | **0.133** | **0.200** | 0.20 | 0.2 | +-0.09s (-2f) | 2 frames |
| **Iteration A (Calibration-Aware Collision Geometry)** | 5 | 22 | 10 | **0.185** | **0.333** | **0.238** | 1.47 | 1.9 | +0.80s (+19f) | 20 frames |

### Split: `split_geo_aware`
| Configuration | TP | FP (Ep) | FN | Precision | Recall | F1 Score | FP / Vid | Raw FP / Vid | Med Lead Time | Med Abs Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Untouched Multi-Signal)** | 1 | 10 | 14 | **0.091** | **0.067** | **0.077** | 0.67 | 227.8 | +2.80s (+28f) | 28 frames |
| **Previous Best (Fixed Metric Distance Gate)** | 1 | 3 | 14 | **0.250** | **0.067** | **0.105** | 0.20 | 0.2 | +-0.13s (-4f) | 4 frames |
| **Iteration A (Calibration-Aware Collision Geometry)** | 4 | 21 | 11 | **0.160** | **0.267** | **0.200** | 1.40 | 2.0 | +1.47s (+26f) | 26 frames |

## 2. Temporal False Positive Taxonomy Breakdown
| Split | Configuration | Pre-Impact FP | Post-Impact FP | Repeated Incident FP | Total FP Episodes |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **In-Dist** | **Baseline (Untouched Multi-Signal)** | 9 | 6 | 5 | **15** |
| **In-Dist** | **Previous Best (Fixed Metric Distance Gate)** | 0 | 3 | 3 | **3** |
| **In-Dist** | **Iteration A (Calibration-Aware Collision Geometry)** | 6 | 16 | 14 | **22** |
| **Geo-Aware** | **Baseline (Untouched Multi-Signal)** | 6 | 4 | 3 | **10** |
| **Geo-Aware** | **Previous Best (Fixed Metric Distance Gate)** | 0 | 3 | 3 | **3** |
| **Geo-Aware** | **Iteration A (Calibration-Aware Collision Geometry)** | 3 | 18 | 16 | **21** |