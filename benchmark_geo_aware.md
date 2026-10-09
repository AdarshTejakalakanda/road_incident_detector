# Phase 1 Incident Detection Quantitative Benchmark Report
**Split Evaluated**: `test (split_geo_aware)`  
**Total Videos Analyzed**: 15

## 1. Overall Incident Performance Summary
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Evaluated Videos** | 15 | Total sample size |
| **True Positives (TP)** | 6 | Ground-truth accidents successfully triggered within tolerance |
| **False Positives (FP)** | 286 | Candidates triggered outside accident window |
| **False Negatives (FN)** | 9 | Ground-truth accidents missed by detector |
| **Precision** | **0.021** | TP / (TP + FP) |
| **Recall** | **0.400** | TP / (TP + FN) |
| **F1 Score** | **0.039** | Harmonic mean of Precision & Recall |
| **FP per Video** | **19.07** | Average false alarms per processed video |
| **Mean Temporal Error** | -27.3 frames | Average frame offset ($f_{\text{pred}} - f_{\text{gt}}$) |
| **Median Temporal Error** | -26.0 frames | Robust central tendency |
| **Mean Detection Latency** | -1.98 sec | Average seconds from impact to trigger |
| **Median Detection Latency** | -2.21 sec | Robust detection delay |

---

## 2. Stratified Robustness Evaluation
### By Weather
| Category | Videos | TP | FP | FN | Precision | Recall | F1 | FP/Vid | Med Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `normal` | 13 | 6 | 286 | 7 | 0.021 | 0.462 | 0.039 | 22.00 | -2.21s |
| `rain` | 2 | 0 | 0 | 2 | 0.000 | 0.000 | 0.000 | 0.00 | +0.00s |

### By Scene Layout
| Category | Videos | TP | FP | FN | Precision | Recall | F1 | FP/Vid | Med Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `grade_separated_intersection` | 1 | 0 | 0 | 1 | 0.000 | 0.000 | 0.000 | 0.00 | +0.00s |
| `highway` | 9 | 2 | 151 | 7 | 0.013 | 0.222 | 0.025 | 16.78 | -2.23s |
| `signalized_intersection` | 4 | 3 | 122 | 1 | 0.024 | 0.750 | 0.047 | 30.50 | -1.54s |
| `simple_intersection` | 1 | 1 | 13 | 0 | 0.071 | 1.000 | 0.133 | 13.00 | -2.90s |

### By Day Time
| Category | Videos | TP | FP | FN | Precision | Recall | F1 | FP/Vid | Med Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `day` | 10 | 6 | 286 | 4 | 0.021 | 0.600 | 0.040 | 28.60 | -2.21s |
| `night` | 5 | 0 | 0 | 5 | 0.000 | 0.000 | 0.000 | 0.00 | +0.00s |

### By Quality
| Category | Videos | TP | FP | FN | Precision | Recall | F1 | FP/Vid | Med Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Fine` | 2 | 2 | 151 | 0 | 0.013 | 1.000 | 0.026 | 75.50 | -1.90s |
| `Good` | 1 | 1 | 57 | 0 | 0.017 | 1.000 | 0.034 | 57.00 | -0.77s |
| `Poor` | 8 | 2 | 60 | 6 | 0.032 | 0.250 | 0.057 | 7.50 | -2.21s |
| `Very_Poor` | 4 | 1 | 18 | 3 | 0.053 | 0.250 | 0.087 | 4.50 | -2.90s |

### By Incident Type
| Category | Videos | TP | FP | FN | Precision | Recall | F1 | FP/Vid | Med Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `rear-end` | 5 | 3 | 156 | 2 | 0.019 | 0.600 | 0.037 | 31.20 | -2.21s |
| `single` | 4 | 0 | 0 | 4 | 0.000 | 0.000 | 0.000 | 0.00 | +0.00s |
| `t-bone` | 6 | 3 | 130 | 3 | 0.023 | 0.500 | 0.043 | 21.67 | -1.54s |

---

## 3. Per-Video Evaluation Breakdown
| Video ID | GT Frame | Pred Frame | Result | Temp Error | Latency | FP Count | Type | Weather | Quality | Layout |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Z4kg2Ev3vhk_00` | 170 | - | _MISS (FN)_ | - | - | 0 | rear-end | normal | Very_Poor | highway |
| `UarP8qU1S-c_00` | 130 | - | _MISS (FN)_ | - | - | 0 | single | normal | Poor | highway |
| `UarP8qU1S-c_01` | 30 | - | _MISS (FN)_ | - | - | 0 | single | normal | Poor | highway |
| `nAXTthLfgtI_00` | 62 | - | _MISS (FN)_ | - | - | 0 | single | normal | Very_Poor | highway |
| `VozrseCzhQc_00` | 84 | - | _MISS (FN)_ | - | - | 0 | single | rain | Poor | highway |
| `kABWTDKgc9U_00` | 197 | 174 | **HIT (TP)** | -23 f | -0.77s | 57 | t-bone | normal | Good | signalized_intersection |
| `LiM5LPSZoow_00` | 53 | 30 | **HIT (TP)** | -23 f | -1.54s | 58 | t-bone | normal | Fine | signalized_intersection |
| `F0NHENn-HS4_00` | 39 | 5 | **HIT (TP)** | -34 f | -2.26s | 93 | rear-end | normal | Fine | highway |
| `maJsgJsapH4_00` | 69 | 47 | **HIT (TP)** | -22 f | -2.21s | 53 | rear-end | normal | Poor | highway |
| `iA_u73BG4ZM_00` | 116 | - | _MISS (FN)_ | - | - | 5 | rear-end | normal | Very_Poor | highway |
| `maJGw3hQs_s_00` | 31 | 2 | **HIT (TP)** | -29 f | -2.90s | 13 | t-bone | normal | Very_Poor | simple_intersection |
| `aPsS_HaokoU_00` | 80 | - | _MISS (FN)_ | - | - | 2 | t-bone | normal | Poor | signalized_intersection |
| `aF2Hbp8C23A_00` | 45 | 12 | **HIT (TP)** | -33 f | -2.21s | 5 | rear-end | normal | Poor | signalized_intersection |
| `82sQ7N-aHOw_00` | 15 | - | _MISS (FN)_ | - | - | 0 | t-bone | normal | Poor | grade_separated_intersection |
| `bJMc3sNP2AU_00` | 32 | - | _MISS (FN)_ | - | - | 0 | t-bone | rain | Poor | highway |
