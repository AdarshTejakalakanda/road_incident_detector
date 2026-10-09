# Iteration B: Selective Detector Recovery Benchmark Report
**Evaluated Splits**: `split_in_distribution` (15 videos) vs `split_geo_aware` (15 videos) | **GT Tolerance**: $\pm 35$ frames (~1.17s)

## 1. Primary Comparative Incident Detection Metrics
### Split: `split_in_distribution`
| Configuration | TP | FP (Ep) | FN | Precision | Recall | F1 Score | FP / Vid | Raw FP / Vid | Med Lead Time | Med Abs Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iteration A (Calibration-Aware)** | 4 | 7 | 11 | **0.364** | **0.267** | **0.308** | 0.47 | 0.7 | +0.72s (+16f) | 20 frames |
| **Iteration B (Selective Recovery + Restored Lifecycle)** | 4 | 7 | 11 | **0.364** | **0.267** | **0.308** | 0.47 | 0.6 | +1.47s (+18f) | 22 frames |

### Split: `split_geo_aware`
| Configuration | TP | FP (Ep) | FN | Precision | Recall | F1 Score | FP / Vid | Raw FP / Vid | Med Lead Time | Med Abs Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iteration A (Calibration-Aware)** | 4 | 9 | 11 | **0.308** | **0.267** | **0.286** | 0.60 | 1.0 | +1.47s (+26f) | 26 frames |
| **Iteration B (Selective Recovery + Restored Lifecycle)** | 5 | 9 | 10 | **0.357** | **0.333** | **0.345** | 0.60 | 1.1 | +2.15s (+25f) | 25 frames |

## 2. Temporal False Positive Breakdown
| Split | Configuration | Pre-Impact FP | Post-Impact FP | Repeated Incident FP | Total FP Episodes |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **In-Dist** | **Iteration A (Calibration-Aware)** | 4 | 3 | 3 | **7** |
| **In-Dist** | **Iteration B (Selective Recovery + Restored Lifecycle)** | 5 | 2 | 2 | **7** |
| **Geo-Aware** | **Iteration A (Calibration-Aware)** | 2 | 7 | 6 | **9** |
| **Geo-Aware** | **Iteration B (Selective Recovery + Restored Lifecycle)** | 3 | 6 | 4 | **9** |

## 3. Detailed Ground Truth vs. Predicted Timing Breakdown (Iteration B)
| Split | Video ID | GT Annotation Frame | Predicted Trigger Frame | Temporal Error | Lead Time | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **In-Dist** | `Z4kg2Ev3vhk_00` | 170 | None | -- | -- | **FN** |
| **In-Dist** | `unS0-TLF1ao_00` | 64 | None | -- | -- | **FN** |
| **In-Dist** | `nAXTthLfgtI_00` | 62 | None | -- | -- | **FN** |
| **In-Dist** | `fzWY0vLAXzI_00` | 88 | None | -- | -- | **FN** |
| **In-Dist** | `vxZ47DoPNY8_00` | 50 | None | -- | -- | **FN** |
| **In-Dist** | `OXVXsdDT9QQ_00` | 17 | 37 | -20f | -1.20s | **TP** |
| **In-Dist** | `VozrseCzhQc_00` | 84 | None | -- | -- | **FN** |
| **In-Dist** | `kABWTDKgc9U_00` | 197 | None | -- | -- | **FN** |
| **In-Dist** | `ftk_fSc4haM_00` | 130 | None | -- | -- | **FN** |
| **In-Dist** | `BgdKGnFqz0I_00` | 125 | None | -- | -- | **FN** |
| **In-Dist** | `LiM5LPSZoow_00` | 53 | 41 | +12f | +0.80s | **TP** |
| **In-Dist** | `iA_u73BG4ZM_00` | 116 | None | -- | -- | **FN** |
| **In-Dist** | `maJGw3hQs_s_00` | 31 | 6 | +25f | +2.50s | **TP** |
| **In-Dist** | `aPsS_HaokoU_00` | 80 | None | -- | -- | **FN** |
| **In-Dist** | `aF2Hbp8C23A_00` | 45 | 13 | +32f | +2.15s | **TP** |
| **Geo-Aware** | `Z4kg2Ev3vhk_00` | 170 | None | -- | -- | **FN** |
| **Geo-Aware** | `UarP8qU1S-c_00` | 130 | None | -- | -- | **FN** |
| **Geo-Aware** | `UarP8qU1S-c_01` | 30 | None | -- | -- | **FN** |
| **Geo-Aware** | `nAXTthLfgtI_00` | 62 | None | -- | -- | **FN** |
| **Geo-Aware** | `VozrseCzhQc_00` | 84 | None | -- | -- | **FN** |
| **Geo-Aware** | `kABWTDKgc9U_00` | 197 | None | -- | -- | **FN** |
| **Geo-Aware** | `LiM5LPSZoow_00` | 53 | 41 | +12f | +0.80s | **TP** |
| **Geo-Aware** | `F0NHENn-HS4_00` | 39 | 4 | +35f | +2.33s | **TP** |
| **Geo-Aware** | `maJsgJsapH4_00` | 69 | 66 | +3f | +0.30s | **TP** |
| **Geo-Aware** | `iA_u73BG4ZM_00` | 116 | None | -- | -- | **FN** |
| **Geo-Aware** | `maJGw3hQs_s_00` | 31 | 6 | +25f | +2.50s | **TP** |
| **Geo-Aware** | `aPsS_HaokoU_00` | 80 | None | -- | -- | **FN** |
| **Geo-Aware** | `aF2Hbp8C23A_00` | 45 | 13 | +32f | +2.15s | **TP** |
| **Geo-Aware** | `82sQ7N-aHOw_00` | 15 | None | -- | -- | **FN** |
| **Geo-Aware** | `bJMc3sNP2AU_00` | 32 | None | -- | -- | **FN** |