"""Modules package for surveillance components."""
from .lighting_gate import (
    LightingCondition,
    LightingGate,
    GammaPreprocessor,
    ClahePreprocessor,
    IlluminationClassifier,
)
from .vocabulary_gate import (
    VEHICLE_SUPERCLASS,
    VocabularyGate,
    compute_bev_footprint,
    is_point_in_polygon,
)
from .detector_quality_gate import (
    DetectionRecord,
    DetectorQualityGate,
)
from .evidence_builder import (
    select_temporal_evidence_frames,
    render_actor_overlays,
    create_stage_t_composite_strip,
)
from .vlm_arbitrator import (
    create_evidence_strip,
    passes_telemetry_prefilter,
    GroqArbitrator,
)
from .single_vehicle import (
    calculate_heading_deviation,
    distance_to_polygon_boundary,
    check_has_lead_vehicle,
    compute_single_vehicle_evidence,
)

__all__ = [
    "LightingCondition",
    "LightingGate",
    "GammaPreprocessor",
    "ClahePreprocessor",
    "IlluminationClassifier",
    "VEHICLE_SUPERCLASS",
    "VocabularyGate",
    "compute_bev_footprint",
    "is_point_in_polygon",
    "DetectionRecord",
    "DetectorQualityGate",
    "create_evidence_strip",
    "passes_telemetry_prefilter",
    "GroqArbitrator",
    "calculate_heading_deviation",
    "distance_to_polygon_boundary",
    "check_has_lead_vehicle",
    "compute_single_vehicle_evidence",
]
