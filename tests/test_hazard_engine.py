"""Unit tests for Iteration G: Road Obstacle and Hazard Detection Engine."""

import pytest
import numpy as np

from road_surveillance.tracking.track_state import Track, TrackletState
from road_surveillance.geometry.spatial_metrics import TrackKinematics
from road_surveillance.events.traffic_flow import SpatialTrafficFlowField
from road_surveillance.events.hazard.vehicle_response import (
    VehicleResponseDetector,
    VehicleResponseClass,
    HazardZone,
)
from road_surveillance.events.hazard.stalled_vehicle_sm import (
    StalledVehicleStateMachine,
    StalledVehicleState,
    StalledVehicleManager,
)
from road_surveillance.events.hazard.debris_hazard import DebrisHazardModule
from road_surveillance.events.hazard.animal_hazard import AnimalHazardModule
from road_surveillance.events.hazard.hazard_engine import HazardEngine


def _make_track(
    tid: int,
    class_label: str = "car",
    bbox: tuple = (100.0, 100.0, 50.0, 80.0),
    age: int = 10,
) -> Track:
    tr = Track(
        track_id=tid,
        class_id=2,
        class_label=class_label,
        current_bbox_xywh=bbox,
        detection_confidence=0.90,
        track_age=age,
    )
    return tr


def _make_kinematics(
    tid: int,
    speed_kmh: float = 50.0,
    heading_vec: tuple = (0.0, 1.0),
    ground_pos: tuple = (10.0, 20.0),
    acceleration_mag: float = 0.0,
    stationary_duration_sec: float = 0.0,
    is_in_active_lane: bool = True,
    is_calibrated: bool = True,
    lane_id: str = "lane_1",
) -> TrackKinematics:
    return TrackKinematics(
        track_id=tid,
        frame_idx=10,
        pts_ns=1000000,
        ground_pos=ground_pos,
        velocity_vector=(heading_vec[0] * (speed_kmh / 3.6), heading_vec[1] * (speed_kmh / 3.6)),
        speed_mps=speed_kmh / 3.6,
        speed_kmh=speed_kmh,
        acceleration_vector=(0.0, acceleration_mag),
        acceleration_mag=acceleration_mag,
        heading_deg=90.0,
        heading_vector=heading_vec,
        stationary_duration_sec=stationary_duration_sec,
        lane_id=lane_id,
        is_in_active_lane=is_in_active_lane,
        is_ground_valid=True,
        is_calibrated=is_calibrated,
    )


def test_vehicle_response_detector_braking_and_swerve():
    detector = VehicleResponseDetector(approach_radius_m=30.0)

    # Hazard zone at (10.0, 30.0)
    hz = HazardZone(
        hazard_id="hazard_1",
        ground_pos=(10.0, 30.0),
        norm_pos=(0.5, 0.5),
        approach_radius_m=30.0,
        is_calibrated=True,
    )

    # Vehicle approaching from (10.0, 10.0), moving towards (10.0, 30.0)
    # Frame 0: Vehicle enters approach zone at 50 km/h
    tr1 = _make_track(tid=1, bbox=(500.0, 500.0, 60.0, 90.0))
    kin1_entry = _make_kinematics(tid=1, speed_kmh=50.0, heading_vec=(0.0, 1.0), ground_pos=(10.0, 10.0))

    detector.update([hz], {1: tr1}, {1: kin1_entry}, frame_idx=0, pts_ns=0)

    # Frame 5: Vehicle brakes heavily as it gets closer (decel = -4.5 m/s^2, speed down to 20 km/h)
    kin1_brake = _make_kinematics(
        tid=1, speed_kmh=20.0, heading_vec=(0.0, 1.0), ground_pos=(10.0, 22.0), acceleration_mag=-4.5
    )
    reactions = detector.update([hz], {1: tr1}, {1: kin1_brake}, frame_idx=5, pts_ns=166000000)

    assert "hazard_1" in reactions
    assert len(reactions["hazard_1"]) >= 1
    rx = reactions["hazard_1"][0]
    assert rx.vehicle_tid == 1
    assert rx.response == VehicleResponseClass.BRAKING
    assert rx.is_evasive is True
    assert rx.max_deceleration_ms2 == pytest.approx(4.5, abs=0.1)


def test_vehicle_response_detector_lane_change():
    detector = VehicleResponseDetector(approach_radius_m=30.0)
    hz = HazardZone(
        hazard_id="hazard_2",
        ground_pos=(10.0, 30.0),
        norm_pos=(0.5, 0.5),
        approach_radius_m=30.0,
        is_calibrated=True,
    )

    tr2 = _make_track(tid=2)
    # Frame 0: in lane_1
    kin2_entry = _make_kinematics(tid=2, speed_kmh=45.0, heading_vec=(0.0, 1.0), ground_pos=(10.0, 10.0), lane_id="lane_1")
    detector.update([hz], {2: tr2}, {2: kin2_entry}, frame_idx=0, pts_ns=0)

    # Frame 6: vehicle changes to lane_2 with lateral swerve
    kin2_avoid = _make_kinematics(tid=2, speed_kmh=40.0, heading_vec=(0.3, 0.95), ground_pos=(13.5, 25.0), lane_id="lane_2")
    detector.update([hz], {2: tr2}, {2: kin2_avoid}, frame_idx=6, pts_ns=200000000)

    # Vehicle departs zone at frame 10
    kin2_exit = _make_kinematics(tid=2, speed_kmh=40.0, heading_vec=(0.0, 1.0), ground_pos=(13.5, 65.0), lane_id="lane_2")
    reactions = detector.update([hz], {2: tr2}, {2: kin2_exit}, frame_idx=10, pts_ns=333000000)

    assert len(reactions["hazard_2"]) >= 1
    rx = reactions["hazard_2"][0]
    assert rx.vehicle_tid == 2
    assert rx.response in (VehicleResponseClass.LANE_CHANGE, VehicleResponseClass.EVASIVE_DRIFT)
    assert rx.is_evasive is True


def test_stalled_vehicle_state_machine_progression_and_corroboration():
    sm = StalledVehicleStateMachine(track_id=10, min_stationary_duration_sec=2.0, unassisted_hazard_duration_sec=5.0)

    # 1. Initially moving
    kin = _make_kinematics(tid=10, speed_kmh=40.0)
    assert sm.step(kin, is_in_active_corridor=True, neighbor_stopped_count=0, approaching_evasive_count=0, frame_idx=1) == StalledVehicleState.MOVING

    # 2. Slowing down
    kin_slow = _make_kinematics(tid=10, speed_kmh=8.0, acceleration_mag=-2.0)
    assert sm.step(kin_slow, is_in_active_corridor=True, neighbor_stopped_count=0, approaching_evasive_count=0, frame_idx=10) == StalledVehicleState.SLOWING

    # 3. Stopped for 2.5 seconds (dt=0.5s x 5 steps)
    kin_stop = _make_kinematics(tid=10, speed_kmh=0.0)
    for f in range(5):
        sm.step(kin_stop, is_in_active_corridor=True, neighbor_stopped_count=0, approaching_evasive_count=0, frame_idx=20 + f, dt_sec=0.5)

    assert sm.state == StalledVehicleState.STATIONARY

    # 4. Approaching vehicle exhibits evasive reaction -> transitions immediately to ROADWAY_HAZARD
    sm.step(kin_stop, is_in_active_corridor=True, neighbor_stopped_count=0, approaching_evasive_count=1, frame_idx=30, dt_sec=0.1)
    assert sm.state == StalledVehicleState.ROADWAY_HAZARD
    assert sm.confidence >= 0.80

    # 5. Vehicle resumes driving -> exits to MOVING
    kin_resume = _make_kinematics(tid=10, speed_kmh=25.0)
    sm.step(kin_resume, is_in_active_corridor=True, neighbor_stopped_count=0, approaching_evasive_count=0, frame_idx=40, dt_sec=0.1)
    assert sm.state == StalledVehicleState.MOVING


def test_stalled_vehicle_traffic_signal_congestion_rejection():
    sm = StalledVehicleStateMachine(track_id=11, congestion_stopped_threshold=2)

    kin_stop = _make_kinematics(tid=11, speed_kmh=0.0)
    for f in range(10):
        # 3 surrounding vehicles also stopped (traffic light red)
        sm.step(kin_stop, is_in_active_corridor=True, neighbor_stopped_count=3, approaching_evasive_count=0, frame_idx=f, dt_sec=0.5)

    # Must be categorized as NORMAL_STOP (red light queue), NOT ROADWAY_HAZARD
    assert sm.state == StalledVehicleState.NORMAL_STOP
    assert sm.confidence == 0.0

    # Now the signal turns green: neighbor stopped count drops to 0, but vehicle 11 remains stuck
    for f in range(5):
        sm.step(kin_stop, is_in_active_corridor=True, neighbor_stopped_count=0, approaching_evasive_count=0, frame_idx=15 + f, dt_sec=0.5)

    # Transits back to STATIONARY
    assert sm.state == StalledVehicleState.STATIONARY


def test_debris_hazard_module_with_evasive_corroboration():
    debris_mod = DebrisHazardModule(min_reactions_to_confirm=1)
    response_det = VehicleResponseDetector(approach_radius_m=30.0)

    # Stationary box on road
    tr_box = _make_track(tid=99, class_label="box", bbox=(400.0, 600.0, 40.0, 30.0))
    kin_box = _make_kinematics(tid=99, speed_kmh=0.0, ground_pos=(5.0, 15.0), stationary_duration_sec=4.0)

    # Approach zone
    hz = HazardZone(
        hazard_id="debris_99",
        ground_pos=(5.0, 15.0),
        norm_pos=(0.4, 0.6),
        approach_radius_m=30.0,
        is_calibrated=True,
        source_track_id=99,
    )

    # Approaching car swerves around box
    tr_car = _make_track(tid=12, class_label="car")
    kin_car_entry = _make_kinematics(tid=12, speed_kmh=40.0, heading_vec=(0.0, 1.0), ground_pos=(5.0, 0.0))
    response_det.update([hz], {99: tr_box, 12: tr_car}, {99: kin_box, 12: kin_car_entry}, frame_idx=0, pts_ns=0)

    kin_car_swerve = _make_kinematics(tid=12, speed_kmh=35.0, heading_vec=(0.4, 0.9), ground_pos=(8.5, 14.0))
    response_det.update([hz], {99: tr_box, 12: tr_car}, {99: kin_box, 12: kin_car_swerve}, frame_idx=5, pts_ns=166000000)

    candidates = debris_mod.evaluate_candidates(
        frame_idx=10,
        pts_ns=333000000,
        video_id="test_video",
        tracks={99: tr_box},
        kinematics={99: kin_box},
        response_detector=response_det,
        flow_field=None,
        frame_shape=(1080, 1920),
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.event_type == "debris_hazard"
    assert cand.primary_track_id == 99
    assert cand.confidence >= 0.75
    assert cand.signals["corroborated_by_traffic"] is True


def test_animal_hazard_module():
    animal_mod = AnimalHazardModule(min_consecutive_frames=3)
    response_det = VehicleResponseDetector()

    tr_dog = _make_track(tid=50, class_label="dog", bbox=(300.0, 400.0, 30.0, 30.0))
    kin_dog = _make_kinematics(tid=50, speed_kmh=5.0, ground_pos=(2.0, 10.0))

    # Evaluate 4 consecutive frames
    for f in range(4):
        cands = animal_mod.evaluate_candidates(
            frame_idx=f,
            pts_ns=f * 33000000,
            video_id="test_video",
            tracks={50: tr_dog},
            kinematics={50: kin_dog},
            response_detector=response_det,
            flow_field=None,
            frame_shape=(1080, 1920),
        )

    assert len(cands) == 1
    cand = cands[0]
    assert cand.event_type == "animal_hazard"
    assert cand.primary_track_id == 50
    assert cand.signals["animal_class"] == "dog"
    assert cand.confidence >= 0.60


def test_hazard_engine_end_to_end():
    engine = HazardEngine(min_stalled_duration_sec=1.0, unassisted_stalled_duration_sec=3.0)

    # Flow learning vehicle moving downwards
    tr_flow = _make_track(tid=100, class_label="car")
    kin_flow = _make_kinematics(tid=100, speed_kmh=50.0, heading_vec=(0.0, 1.0), ground_pos=(10.0, 10.0))

    # Stalled vehicle in active lane
    tr_stalled = _make_track(tid=101, class_label="car", bbox=(600.0, 600.0, 60.0, 80.0))
    kin_stalled = _make_kinematics(tid=101, speed_kmh=0.0, ground_pos=(10.0, 40.0), stationary_duration_sec=0.0)

    # Approaching vehicle
    tr_app = _make_track(tid=102, class_label="car", bbox=(600.0, 300.0, 60.0, 80.0))
    kin_app = _make_kinematics(tid=102, speed_kmh=45.0, heading_vec=(0.0, 1.0), ground_pos=(10.0, 20.0))

    # Step through 10 frames
    all_emitted = []
    for f in range(10):
        # Update stationary duration on stalled car
        kin_stalled.stationary_duration_sec = f * 0.5
        # Approaching car brakes heavily at frame 5
        if f >= 5:
            kin_app.speed_kmh = 10.0
            kin_app.acceleration_mag = -4.0
            kin_app.ground_pos = (10.0, 32.0)

        cands = engine.process_frame(
            frame_idx=f,
            pts_ns=f * 500000000,
            video_id="test_hazard_e2e",
            tracks={100: tr_flow, 101: tr_stalled, 102: tr_app},
            kinematics={100: kin_flow, 101: kin_stalled, 102: kin_app},
            dt_sec=0.5,
        )
        all_emitted.extend(cands)

    # Stalled vehicle should be emitted
    stalled_events = [c for c in all_emitted if c.event_type == "stalled_vehicle"]
    assert len(stalled_events) >= 1
    ev = stalled_events[0]
    assert ev.primary_track_id == 101
    assert ev.confidence >= 0.70
    assert "evasive" in ev.description.lower() or "maneuver" in ev.description.lower()
