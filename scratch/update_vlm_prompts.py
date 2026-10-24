with open('modules/vlm_arbitrator.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''SINGLE_VEHICLE_PROMPT = """You are an expert Traffic Incident Verification Specialist in an Intelligent Transportation System.
You are evaluating a potential single-vehicle incident (barrier impact, loss of control, run-off-road) from video surveillance.

Input Filmstrip Description:
- Panel 1 (APPROACH): Pre-event vehicle approach.
- Panel 2 (CANDIDATE IMPACT): Point of maximum kinematic disruption or proximity to boundary.
- Panel 3 (POST-EVENT): Immediate post-incident position, demonstrating recovery, rest, or trajectory arrest.

Telemetry: Vehicle exhibited a heading shift and sharp velocity dissipation near the roadway boundary.
Determine if the vehicle experienced an actual incident (spin-out, barrier/curb strike, rollover, or acute roadway loss of control) OR if this is benign behavior (controlled lane departure, safe shoulder pull-off, routine exit ramp curve).

Return ONLY valid JSON:
{
  \"physical_contact_observed\": true | false,
  \"structural_damage_or_deformation\": true | false,
  \"post_impact_displacement_or_rest\": true | false,
  \"benign_alternative\": \"none\" | \"normal_turn\" | \"queue_stop\" | \"lane_change\" | \"camera_artifact\",
  \"confidence\": 0.0 to 1.0,
  \"incident_type\": \"spin_out\" | \"barrier_strike\" | \"rollover\" | \"controlled_maneuver\" | \"benign_stop\",
  \"reasoning\": \"brief visual description\"
}
\"\"\"

MULTI_VEHICLE_PROMPT = """You are an expert Traffic Incident Verification Specialist in an Intelligent Transportation System.
Analyze the attached single-image composite filmstrip showing 3 sequential temporal stages:
1) APPROACH, 2) CANDIDATE IMPACT, 3) POST-EVENT.
Determine if a collision occurred between the vehicles.

Return ONLY a strict JSON object with fields:
- physical_contact_observed (bool)
- structural_damage_or_deformation (bool)
- post_impact_displacement_or_rest (bool)
- benign_alternative (string: 'none', 'normal_turn', 'queue_stop', 'lane_change', or 'camera_artifact')
- confidence (float 0.0 - 1.0)
- collision_type (string: 'rear_end_collision', 'sideswipe', 't_bone_collision', or 'none')
- rationale (string explanation)
\"\"\"'''

repl = '''SINGLE_VEHICLE_PROMPT = """You are an expert Traffic Incident Verification Specialist in an Intelligent Transportation System.
You are evaluating a potential single-vehicle incident (barrier impact, loss of control, spin-out, rollover) from video surveillance.

Input Evidence Triptych Description (Fixed Collision Zone across 3 Panels):
- Panel 1 (PRE-EVENT: T-5s): Pre-event vehicle approach 5.0 seconds before trigger. Normal roadway cruising flow.
- Panel 2 (TRIGGER / IMPACT: T): Physical collision or acute kinematic disruption against barrier / roadway boundary.
- Panel 3 (POST-EVENT AFTERMATH: T+5s): 5.0 seconds after impact. Inspect whether the vehicle came to a complete kinetic rest / vehicle immobilization, hazard lights, traffic avoidance, or if it safely continued driving normally.

Determine if the vehicle experienced an actual physical incident (spin-out, barrier/curb strike, rollover, or acute roadway loss of control leading to vehicle immobilization or damage) OR if this is benign behavior (controlled lane departure, safe shoulder pull-off, routine exit ramp curve).

Return ONLY valid JSON:
{
  \"physical_contact_observed\": true | false,
  \"structural_damage_or_deformation\": true | false,
  \"post_impact_displacement_or_rest\": true | false,
  \"benign_alternative\": \"none\" | \"normal_turn\" | \"queue_stop\" | \"lane_change\" | \"camera_artifact\",
  \"confidence\": 0.0 to 1.0,
  \"incident_type\": \"spin_out\" | \"barrier_strike\" | \"rollover\" | \"controlled_maneuver\" | \"benign_stop\",
  \"reasoning\": \"brief visual description\"
}
\"\"\"

MULTI_VEHICLE_PROMPT = """You are an expert Traffic Incident Verification Specialist in an Intelligent Transportation System.
Analyze the attached 3-panel fixed-spatial evidence card showing 3 sequential temporal stages across a fixed collision zone:
- Panel 1 (PRE-EVENT: T-5s): Vehicles entering and approaching the collision zone under normal roadway traffic flow (5.0s before impact).
- Panel 2 (TRIGGER / IMPACT: T): Physical contact, impact, or acute collision trajectory convergence.
- Panel 3 (POST-EVENT AFTERMATH: T+5s): 5.0 seconds post-impact. Answers the critical kinetic question: Did the vehicle(s) come to a complete kinetic rest / remain immobilized in the lane while adjacent traffic swerves around them, OR did they continue driving normally?

Determine if an actual physical collision occurred between the vehicles.

Return ONLY a strict JSON object with fields:
- physical_contact_observed (bool)
- structural_damage_or_deformation (bool)
- post_impact_displacement_or_rest (bool)
- benign_alternative (string: 'none', 'normal_turn', 'queue_stop', 'lane_change', or 'camera_artifact')
- confidence (float 0.0 - 1.0)
- collision_type (string: 'rear_end_collision', 'sideswipe', 't_bone_collision', or 'none')
- rationale (string explanation)
\"\"\"'''

assert target in content, "Target prompt block not found"
content = content.replace(target, repl, 1)

with open('modules/vlm_arbitrator.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated modules/vlm_arbitrator.py prompts successfully")
