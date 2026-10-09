MULTI_VEHICLE_PROMPT = """You are an expert Traffic Incident Verification Specialist in an Intelligent Transportation System.
Analyze the attached single-image composite filmstrip showing 3 sequential temporal stages:
determine if a collision occurred between the vehicles.

Return ONLY a strict JSON object with fields:
- physical_contact_observed (bool)
- structural_damage_or_deformation (bool)
- post_impact_displacement_or_rest (bool)
- benign_alternative (string: 'none', 'normal_turn', 'queue_stop', 'lane_change', or 'camera_artifact')
- confidence (float 0.0 - 1.0)
- collision_type (string: 'rear_end_collision', 'sideswipe', 't_bone_collision', or 'none')
- rationale (string explanation)
"""


