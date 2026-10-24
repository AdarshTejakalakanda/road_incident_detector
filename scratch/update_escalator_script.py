with open('modules/scene_escalator.py', 'r', encoding='utf-8') as f:
    content = f.read()

target1 = """    def evaluate_candidate(
        self,
        candidate: Union[CandidateProposal, Dict[str, Any]],
        track_history: Dict[int, List[TrackState]],
        fps: float = 30.0
    ) -> SceneEscalationResult:
        if isinstance(candidate, dict):
            candidate = CandidateProposal.from_dict(candidate)
        f0 = candidate.frame_idx
        tids = candidate.track_ids
        horizon_frames = int(round(1.5 * fps))
        f_end = f0 + horizon_frames"""

repl1 = """    def evaluate_candidate(
        self,
        candidate: Union[CandidateProposal, Dict[str, Any]],
        track_history: Dict[int, List[TrackState]],
        fps: float = 30.0,
        horizon_seconds: float = 1.5,
    ) -> SceneEscalationResult:
        if isinstance(candidate, dict):
            candidate = CandidateProposal.from_dict(candidate)
        f0 = candidate.frame_idx
        tids = candidate.track_ids
        horizon_frames = int(round(horizon_seconds * fps))
        f_end = f0 + horizon_frames"""

assert target1 in content, "target1 not found"
content = content.replace(target1, repl1, 1)

target2 = """        # Physical Consequence Safeguard: Rear-end collisions (Type 1 & Type 2) require real kinetic impact
        # (either sustained velocity drop dv_sust >= 4.0 km/h or kinetic arrest tau_rest >= 0.25 s).
        # Cruising vehicles with dv_sust < 4.0 and tau_rest < 0.25 are passing/adjacent-lane flow.
        if ctype in ('Type 1', 'Type 2') and dv_sust < 4.0 and tau_rest < 0.25:
            escalated = False"""

repl2 = """        # Physical Consequence Safeguard: Rear-end collisions (Type 1 & Type 2) require real kinetic impact
        # (either sustained velocity drop dv_sust >= 4.0 km/h or kinetic arrest tau_rest >= 0.25 s).
        # Cruising vehicles with dv_sust < 4.0 and tau_rest < 0.25 are passing/adjacent-lane flow.
        if ctype in ('Type 1', 'Type 2') and dv_sust < 4.0 and tau_rest < 0.25:
            escalated = False

        # Kinetic Aftermath Rest Safeguard:
        # In physics, severe crashes lead to kinetic energy dissipation and sustained vehicle rest
        # (speed <= 5.0 km/h for >= 0.25s). Passing / cruising traffic where vehicles maintain continuous
        # speed without any rest (tau_rest < 0.25 s) is suppressed as benign roadway flow, unless an explicit
        # post_event_rest signal is asserted.
        has_post_rest_signal = bool(candidate.signals.get('post_event_rest', False))
        if tau_rest < 0.25 and not has_post_rest_signal:
            escalated = False"""

assert target2 in content, "target2 not found"
content = content.replace(target2, repl2, 1)

with open('modules/scene_escalator.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated modules/scene_escalator.py successfully")
