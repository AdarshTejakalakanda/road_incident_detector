# Road Surveillance Incident Detection System — Phase 1 Baseline

A clean, measurable, and reproducible baseline pipeline for road surveillance incident detection on local recorded incident video datasets.

---

## 1. System Pipeline Architecture

```
Real Incident Dataset (Downloads/archive/)
        │
        ├── Video (Real CCTV incident recordings)
        │     ↓
        │   [1. Video Ingestion & Extraction] (pts_ns = int(frame_idx * (1e9 / fps)))
        │     ↓
        │   [2. YOLO26 Detection] (yolo26n.pt, class vocabulary logged at startup)
        │     ↓
        │   [3. ByteTrack Tracker] (Persistent IDs, ground-contact points)
        │     ↓
        │   [4. Camera Calibration] (Planar Homography H: (u,v) -> (X,Y) metric meters)
        │     ↓
        │   [5. OpenCV Geometry Engine] (Velocity, Speed km/h, Heading, Convergence)
        │     ↓
        │   [6. Event Engine: 7 Candidate Detectors]
        │     ├── 1. Wrong-Way Vehicle
        │     ├── 2. Collision Candidate
        │     ├── 3. Stalled Vehicle (with congestion differentiation)
        │     ├── 4. Pedestrian Hazard in Active Lane
        │     ├── 5. Animal Hazard (capability-dependent)
        │     ├── 6. Debris Candidate (capability-dependent)
        │     └── 7. Stranded Heavy Vehicle
        │     │
        │     ├──► [7. Evidence Store] (pre/post-clip, keyframes, telemetry, metadata)
        │     └──► [8. Visualization] (HUD, trajectory trails, alerts, video export)
        │
        └── Metadata (metadata-real.csv)
              ↓
        [9. Event Evaluation Benchmark]
              ├── Metrics: TP, FP, FN, Precision, Recall, F1, Latency, Temporal Error
              ├── Stratified by Weather (normal, rain, snow)
              ├── Stratified by Scene Layout (highway, signalized intersection, etc.)
              ├── Stratified by Day Time (day, night)
              ├── Stratified by Quality (Excellent, Good, Fine, Poor, Very_Poor)
              └── Stratified by Splits (split_in_distribution, split_geo_aware)
```

---

## 2. Architectural Boundaries & Constraints

Phase 1 strictly establishes a measurable **baseline**. The following components are explicitly excluded in Phase 1:
- **NO** Qwen3.8-27B or VLM reasoning models.
- **NO** YOLO26-seg, YOLO26-OBB, YOLO26-depth, or YOLO26-pose.
- **NO** specialist smoke/fire models or monocular depth estimators.
- **NO** live CCTV / RTSP camera streams (recorded local datasets only).
- **NO** cloud storage, Kubernetes, SMS/email alerting, or webhooks.

---

## 3. Environment Setup

### Prerequisites
- Windows 10/11 x64
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)

### Installation Commands
```powershell
# 1. Create Python 3.11 virtual environment
uv venv .venv --python 3.11

# 2. Activate environment
.venv\Scripts\activate

# 3. Install core dependencies
uv pip install --python .venv\Scripts\python.exe numpy opencv-python pydantic pyyaml scipy shapely supervision ultralytics pytest
```

---

## 4. Execution Commands

### Run Full Test Suite (22 Unit + Integration Tests)
```powershell
.venv\Scripts\pytest tests/ -v
```

### Run Pipeline on Real Incident Video
```powershell
.venv\Scripts\python scripts/run_pipeline.py --video "C:/Users/KalakandaAdarshTeja/Downloads/archive/real_videos/Z4kg2Ev3vhk_00.mp4" --max-frames 100
```
Optional flags:
- `--display`: Display real-time playback window with HUD.
- `--no-save-video`: Skip exporting annotated `.mp4` video.
- `--max-videos N`: Limit processing to first $N$ videos.
- `--max-frames N`: Limit frames per video.

### Run Quantitative Evaluation Benchmark
```powershell
# Evaluate on 5 sample videos from the test split
.venv\Scripts\python scripts/evaluate.py --split test --split-type split_in_distribution --max-videos 5 --report-out benchmark_report.md

# Evaluate full split
.venv\Scripts\python scripts/evaluate.py --split test --split-type split_in_distribution --report-out full_benchmark_report.md
```

---

## 5. Output Artifacts & Evidence Package

### Evidence Package Structure
When an incident candidate triggers, an evidence package is generated:
```
evidence/
  YYYY-MM-DD/
    video_id/
      event_id/
        ├── clip/
        │   └── event_<event_id>.mp4       (Pre-event + post-event footage)
        ├── frames/
        │   └── keyframe_<frame_idx>.jpg   (Annotated impact keyframe)
        ├── telemetry/
        │   └── telemetry_<event_id>.json  (Ground coords, speed km/h, acceleration, TCA/DCA)
        └── metadata.json                  (Event metadata, confidence, trigger signals)
```

### Processed Annotated Videos
Output MP4 videos containing bounding boxes, ground-contact indicators, trajectory trails, speed HUD, and event alert banners are saved to:
`output/videos/<video_id>_annotated.mp4`
