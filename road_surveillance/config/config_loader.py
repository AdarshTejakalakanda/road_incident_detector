"""Configuration Loader and Dataclasses for Road Surveillance Incident Detection."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import yaml
import os


@dataclass
class DatasetConfig:
    path: str = "C:/Users/KALAKA~1/Downloads/archive/real_videos"
    recursive: bool = False
    extensions: List[str] = field(default_factory=lambda: [".mp4", ".avi", ".mov"])


@dataclass
class PlaybackConfig:
    display: bool = False
    save_processed_video: bool = True
    realtime: bool = False
    output_dir: str = "./output/videos"


@dataclass
class DetectionConfig:
    model_path: str = "yolo26n.pt"
    confidence_threshold: float = 0.25
    input_resolution: Union[int, str] = "native"
    device: str = "cpu"
    frame_sampling_rate: int = 1


@dataclass
class TrackingConfig:
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    frame_rate: int = 30


@dataclass
class LaneConfig:
    lane_id: str
    name: str = ""
    is_active: bool = True
    permitted_direction: Tuple[float, float] = (0.0, 1.0)
    polygon_ground: List[List[float]] = field(default_factory=list)


@dataclass
class CalibrationProfileConfig:
    source_points: List[List[float]] = field(default_factory=list)
    destination_points: List[List[float]] = field(default_factory=list)
    homography_matrix: Optional[List[List[float]]] = None
    lanes: List[LaneConfig] = field(default_factory=list)
    source_resolution: Optional[List[int]] = None
    is_calibrated: bool = False


@dataclass
class GeometryConfig:
    smoothing_window: int = 5
    stationary_speed_threshold_kmh: float = 3.0
    min_stationary_duration_sec: float = 3.5
    collision_tca_threshold_sec: float = 2.0
    collision_distance_threshold_m: float = 4.0
    collision_decel_threshold_ms2: float = -3.5


@dataclass
class WrongWayConfig:
    enabled: bool = True
    min_speed_kmh: float = 5.0
    cos_threshold: float = -0.5
    min_consecutive_frames: int = 4


@dataclass
class CollisionConfig:
    enabled: bool = True
    distance_decay_threshold_m: float = 1.5
    tca_threshold_sec: float = 2.0
    deceleration_threshold_ms2: float = -3.5
    post_proximity_distance_m: float = 4.5
    iou_overlap_threshold: float = 0.05
    # Parameterized experimental thresholds
    max_ground_distance_m: float = 4.0
    max_perspective_distance_m: float = 12.0
    min_iou_overlap: float = 0.15
    min_persistence_frames: int = 2
    temporal_cluster_gap_frames: int = 25
    min_track_age: int = 8


@dataclass
class StalledVehicleConfig:
    enabled: bool = True
    min_stationary_sec: float = 3.5
    congestion_radius_m: float = 10.0
    congestion_vehicle_threshold: int = 3


@dataclass
class PedestrianHazardConfig:
    enabled: bool = True
    min_consecutive_frames: int = 3


@dataclass
class AnimalHazardConfig:
    enabled: bool = True
    min_consecutive_frames: int = 3


@dataclass
class DebrisCandidateConfig:
    enabled: bool = True
    min_stationary_sec: float = 4.0


@dataclass
class HeavyVehicleConfig:
    enabled: bool = True
    min_stationary_sec: float = 3.5


@dataclass
class SingleVehicleConfig:
    enabled: bool = True
    min_track_age: int = 8
    video_warmup_frames: int = 15
    deceleration_threshold_ms2: float = -4.0
    severe_impact_decel_ms2: float = -14.0
    yaw_rate_threshold_deg: float = 30.0
    aspect_ratio_change_threshold: float = 0.35
    min_persistence_frames: int = 2


@dataclass
class DistantConflictConfig:
    enabled: bool = True
    min_track_age: int = 8
    video_warmup_frames: int = 15
    deceleration_threshold_ms2: float = -3.5
    tca_max_sec: float = 2.0
    dca_norm_max: float = 1.2
    dist_norm_max: float = 3.5
    min_closing_speed_kmh: float = 12.0
    min_persistence_frames: int = 2


@dataclass
class EventsConfig:
    wrong_way: WrongWayConfig = field(default_factory=WrongWayConfig)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    single_vehicle: SingleVehicleConfig = field(default_factory=SingleVehicleConfig)
    distant_conflict: DistantConflictConfig = field(default_factory=DistantConflictConfig)
    stalled_vehicle: StalledVehicleConfig = field(default_factory=StalledVehicleConfig)
    pedestrian_hazard: PedestrianHazardConfig = field(default_factory=PedestrianHazardConfig)
    animal_hazard: AnimalHazardConfig = field(default_factory=AnimalHazardConfig)
    debris_candidate: DebrisCandidateConfig = field(default_factory=DebrisCandidateConfig)
    heavy_vehicle: HeavyVehicleConfig = field(default_factory=HeavyVehicleConfig)


@dataclass
class EvidenceConfig:
    output_dir: str = "./evidence"
    pre_event_seconds: float = 3.0
    post_event_seconds: float = 3.0
    max_buffer_seconds: float = 10.0


@dataclass
class EvaluationConfig:
    metadata_path: str = "C:/Users/KALAKA~1/Downloads/archive/metadata-real.csv"
    split: str = "test"
    split_type: str = "split_in_distribution"


@dataclass
class VerificationConfig:
    enabled: bool = True
    api_base: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen3.8-27B"
    temperature: float = 0.0
    timeout: float = 45.0
    pre_offset_frames: int = 15
    post_offset_frames: int = 15
    uncertain_policy: str = "keep"  # "keep" or "drop"


@dataclass
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    calibration: Dict[str, CalibrationProfileConfig] = field(default_factory=dict)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)


def _parse_dict_to_dataclass(cls, data: dict):
    if not isinstance(data, dict):
        return data
    kwargs = {}
    for f in cls.__dataclass_fields__.values():
        if f.name in data:
            val = data[f.name]
            if hasattr(f.type, "__dataclass_fields__"):
                kwargs[f.name] = _parse_dict_to_dataclass(f.type, val)
            else:
                kwargs[f.name] = val
    return cls(**kwargs)


def load_config(config_path: str = "road_surveillance/config/default_config.yaml") -> AppConfig:
    """Load application configuration from YAML file."""
    if not os.path.exists(config_path):
        return AppConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cfg = AppConfig()

    if "dataset" in data:
        cfg.dataset = DatasetConfig(**data["dataset"])
    if "playback" in data:
        cfg.playback = PlaybackConfig(**data["playback"])
    if "detection" in data:
        cfg.detection = DetectionConfig(**data["detection"])
    if "tracking" in data:
        cfg.tracking = TrackingConfig(**data["tracking"])
    if "geometry" in data:
        cfg.geometry = GeometryConfig(**data["geometry"])
    if "evidence" in data:
        cfg.evidence = EvidenceConfig(**data["evidence"])
    if "evaluation" in data:
        cfg.evaluation = EvaluationConfig(**data["evaluation"])
    if "verification" in data:
        cfg.verification = VerificationConfig(**data["verification"])

    # Calibration profiles
    if "calibration" in data:
        for profile_name, prof_data in data["calibration"].items():
            lanes = []
            for l in prof_data.get("lanes", []):
                lanes.append(LaneConfig(**l))
            cfg.calibration[profile_name] = CalibrationProfileConfig(
                source_points=prof_data.get("source_points", []),
                destination_points=prof_data.get("destination_points", []),
                homography_matrix=prof_data.get("homography_matrix"),
                lanes=lanes,
                source_resolution=prof_data.get("source_resolution"),
                is_calibrated=prof_data.get("is_calibrated", False),
            )

    # Events
    if "events" in data:
        edata = data["events"]
        cfg.events = EventsConfig(
            wrong_way=WrongWayConfig(**edata.get("wrong_way", {})),
            collision=CollisionConfig(**edata.get("collision", {})),
            single_vehicle=SingleVehicleConfig(**edata.get("single_vehicle", {})),
            distant_conflict=DistantConflictConfig(**edata.get("distant_conflict", {})),
            stalled_vehicle=StalledVehicleConfig(**edata.get("stalled_vehicle", {})),
            pedestrian_hazard=PedestrianHazardConfig(**edata.get("pedestrian_hazard", {})),
            animal_hazard=AnimalHazardConfig(**edata.get("animal_hazard", {})),
            debris_candidate=DebrisCandidateConfig(**edata.get("debris_candidate", {})),
            heavy_vehicle=HeavyVehicleConfig(**edata.get("heavy_vehicle", {})),
        )

    return cfg
