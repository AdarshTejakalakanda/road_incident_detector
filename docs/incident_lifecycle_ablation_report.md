# Incident Lifecycle Management & FN Diagnostic Benchmark Report
**Evaluated Splits**: `split_in_distribution` (15 videos) vs `split_geo_aware` (15 videos) | **GT Window**: $\pm 35$ frames (~1.17s)

## 1. Primary Incident Detection Metrics
### Split: `split_in_distribution`
| Configuration | TP | FP (Ep) | FN | Precision | Recall | F1 Score | FP / Vid | Raw FP / Vid | Med Lead Time | Med Abs Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Untouched Multi-Signal)** | 2 | 15 | 13 | **0.118** | **0.133** | **0.125** | 1.00 | 156.9 | +0.83s (+4f) | 24 frames |
| **Stage 6 + Maturity + Validity (Accepted Config)** | 2 | 6 | 13 | **0.250** | **0.133** | **0.174** | 0.40 | 0.6 | +-0.09s (-2f) | 2 frames |
| **Stage 6 + Maturity + Validity + Incident Lifecycle** | 2 | 3 | 13 | **0.400** | **0.133** | **0.200** | 0.20 | 0.2 | +-0.09s (-2f) | 2 frames |

### Split: `split_geo_aware`
| Configuration | TP | FP (Ep) | FN | Precision | Recall | F1 Score | FP / Vid | Raw FP / Vid | Med Lead Time | Med Abs Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Untouched Multi-Signal)** | 1 | 10 | 14 | **0.091** | **0.067** | **0.077** | 0.67 | 227.8 | +2.80s (+28f) | 28 frames |
| **Stage 6 + Maturity + Validity (Accepted Config)** | 1 | 7 | 14 | **0.125** | **0.067** | **0.087** | 0.47 | 0.7 | +-0.13s (-4f) | 4 frames |
| **Stage 6 + Maturity + Validity + Incident Lifecycle** | 1 | 4 | 14 | **0.200** | **0.067** | **0.100** | 0.27 | 0.3 | +-0.13s (-4f) | 4 frames |

## 2. Temporal False Positive Taxonomy Breakdown
| Split | Configuration | Pre-Impact FP | Post-Impact FP | Repeated Incident FP | Total FP Episodes |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **In-Dist** | **Baseline (Untouched Multi-Signal)** | 9 | 6 | 5 | **15** |
| **In-Dist** | **Stage 6 + Maturity + Validity (Accepted Config)** | 0 | 6 | 6 | **6** |
| **In-Dist** | **Stage 6 + Maturity + Validity + Incident Lifecycle** | 0 | 3 | 3 | **3** |
| **Geo-Aware** | **Baseline (Untouched Multi-Signal)** | 6 | 4 | 3 | **10** |
| **Geo-Aware** | **Stage 6 + Maturity + Validity (Accepted Config)** | 1 | 6 | 6 | **7** |
| **Geo-Aware** | **Stage 6 + Maturity + Validity + Incident Lifecycle** | 1 | 3 | 3 | **4** |