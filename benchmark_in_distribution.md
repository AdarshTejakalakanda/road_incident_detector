# Phase 1 Incident Detection Quantitative Benchmark Report
**Split Evaluated**: `test (split_in_distribution)`  
**Total Videos Analyzed**: 15

## 1. Overall Incident Performance Summary
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Evaluated Videos** | 15 | Total sample size |
| **True Positives (TP)** | 9 | Ground-truth accidents successfully triggered within tolerance |
| **False Positives (FP)** | 190 | Candidates triggered outside accident window |
| **False Negatives (FN)** | 6 | Ground-truth accidents missed by detector |
| **Precision** | **0.045** | TP / (TP + FP) |
| **Recall** | **0.600** | TP / (TP + FN) |
| **F1 Score** | **0.084** | Harmonic mean of Precision & Recall |
| **FP per Video** | **12.67** | Average false alarms per processed video |
| **Mean Temporal Error** | -21.8 frames | Average frame offset ($f_{\text{pred}} - f_{\text{gt}}$) |
| **Median Temporal Error** | -23.0 frames | Robust central tendency |
| **Mean Detection Latency** | -1.30 sec | Average seconds from impact to trigger |
| **Median Detection Latency** | -1.13 sec | Robust detection delay |

---

## 2. Stratified Robustness Evaluation
### By Weather
| Category | Videos | TP | FP | FN | Precision | Recall | F1 | FP/Vid | Med Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `normal` | 12 | 7 | 183 | 5 | 0.037 | 0.583 | 0.069 | 15.25 | -1.13s |
| `rain` | 2 | 1 | 3 | 1 | 0.250 | 0.500 | 0.333 | 1.50 | -1.35s |
| `snow` | 1 | 1 | 4 | 0 | 0.200 | 1.000 | 0.333 | 4.00 | -0.57s |

### By Scene Layout
| Category | Videos | TP | FP | FN | Precision | Recall | F1 | FP/Vid | Med Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `city_street` | 1 | 1 | 3 | 0 | 0.250 | 1.000 | 0.400 | 3.00 | -1.35s |
| `highway` | 4 | 0 | 5 | 4 | 0.000 | 0.000 | 0.000 | 1.25 | +0.00s |
| `signalized_intersection` | 9 | 7 | 169 | 2 | 0.040 | 0.778 | 0.076 | 18.78 | -0.90s |
| `simple_intersection` | 1 | 1 | 13 | 0 | 0.071 | 1.000 | 0.133 | 13.00 | -2.90s |

### By Day Time
| Category | Videos | TP | FP | FN | Precision | Recall | F1 | FP/Vid | Med Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `day` | 10 | 7 | 180 | 3 | 0.037 | 0.700 | 0.071 | 18.00 | -1.35s |
| `night` | 5 | 2 | 10 | 3 | 0.167 | 0.400 | 0.235 | 2.00 | -0.63s |

### By Quality
| Category | Videos | TP | FP | FN | Precision | Recall | F1 | FP/Vid | Med Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Fine` | 4 | 4 | 96 | 0 | 0.040 | 1.000 | 0.077 | 24.00 | -1.01s |
| `Good` | 3 | 3 | 69 | 0 | 0.042 | 1.000 | 0.080 | 23.00 | -0.77s |
| `Poor` | 3 | 1 | 7 | 2 | 0.125 | 0.333 | 0.182 | 2.33 | -2.21s |
| `Very_Poor` | 5 | 1 | 18 | 4 | 0.053 | 0.200 | 0.083 | 3.60 | -2.90s |

### By Incident Type
| Category | Videos | TP | FP | FN | Precision | Recall | F1 | FP/Vid | Med Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `rear-end` | 3 | 1 | 10 | 2 | 0.091 | 0.333 | 0.143 | 3.33 | -2.21s |
| `single` | 3 | 1 | 3 | 2 | 0.250 | 0.333 | 0.286 | 1.00 | -1.35s |
| `t-bone` | 9 | 7 | 177 | 2 | 0.038 | 0.778 | 0.073 | 19.67 | -0.90s |

---

## 3. Per-Video Evaluation Breakdown
| Video ID | GT Frame | Pred Frame | Result | Temp Error | Latency | FP Count | Type | Weather | Quality | Layout |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Z4kg2Ev3vhk_00` | 170 | - | _MISS (FN)_ | - | - | 0 | rear-end | normal | Very_Poor | highway |
| `unS0-TLF1ao_00` | 64 | - | _MISS (FN)_ | - | - | 0 | t-bone | normal | Very_Poor | signalized_intersection |
| `nAXTthLfgtI_00` | 62 | - | _MISS (FN)_ | - | - | 0 | single | normal | Very_Poor | highway |
| `fzWY0vLAXzI_00` | 88 | 76 | **HIT (TP)** | -12 f | -0.57s | 4 | t-bone | snow | Fine | signalized_intersection |
| `vxZ47DoPNY8_00` | 50 | 26 | **HIT (TP)** | -24 f | -1.35s | 3 | single | rain | Good | city_street |
| `OXVXsdDT9QQ_00` | 17 | 2 | **HIT (TP)** | -15 f | -0.90s | 1 | t-bone | normal | Fine | signalized_intersection |
| `VozrseCzhQc_00` | 84 | - | _MISS (FN)_ | - | - | 0 | single | rain | Poor | highway |
| `kABWTDKgc9U_00` | 197 | 174 | **HIT (TP)** | -23 f | -0.77s | 57 | t-bone | normal | Good | signalized_intersection |
| `ftk_fSc4haM_00` | 130 | 121 | **HIT (TP)** | -9 f | -0.36s | 9 | t-bone | normal | Good | signalized_intersection |
| `BgdKGnFqz0I_00` | 125 | 97 | **HIT (TP)** | -28 f | -1.13s | 33 | t-bone | normal | Fine | signalized_intersection |
| `LiM5LPSZoow_00` | 53 | 30 | **HIT (TP)** | -23 f | -1.54s | 58 | t-bone | normal | Fine | signalized_intersection |
| `iA_u73BG4ZM_00` | 116 | - | _MISS (FN)_ | - | - | 5 | rear-end | normal | Very_Poor | highway |
| `maJGw3hQs_s_00` | 31 | 2 | **HIT (TP)** | -29 f | -2.90s | 13 | t-bone | normal | Very_Poor | simple_intersection |
| `aPsS_HaokoU_00` | 80 | - | _MISS (FN)_ | - | - | 2 | t-bone | normal | Poor | signalized_intersection |
| `aF2Hbp8C23A_00` | 45 | 12 | **HIT (TP)** | -33 f | -2.21s | 5 | rear-end | normal | Poor | signalized_intersection |
