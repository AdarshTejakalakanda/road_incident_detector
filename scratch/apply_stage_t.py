with open("modules/vlm_arbitrator.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

helpers_code = [
    chr(10),
    'STAGE_T_VLM_PROMPT = """You are an expert Traffic Incident Verification Specialist in an Intelligent Transportation System.""" + chr(10)',
    'STAGE_T_VLM_PROMPT += "Analyze the attached composite image showing 3-stage temporal progression: Pre-event (t - 1.0s), Impact (t), Post-event (t + 1.5s)." + chr(10)',
    'STAGE_T_VLM_PROMPT += "Evaluate strictly against physical evidence:" + chr(10)',
    'STAGE_T_VLM_PROMPT += "1. physical_contact_observed: True ONLY if direct bumper/chassis/barrier contact is visibly observed." + chr(10)',
    'STAGE_T_VLM_PROMPT += "2. structural_damage_or_deformation: True ONLY if visible bumper crumpling, panel crushing, or hood buckling is observed." + chr(10)',
    'STAGE_T_VLM_PROMPT += "3. post_impact_displacement_or_rest: True ONLY if vehicle trajectory is abruptly halted, deflected off course, or brought to complete stop at impact." + chr(10)',
    'STAGE_T_VLM_PROMPT += "4. benign_alternative: Categorize what benign behavior occurred if this was not an incident (none, normal_turn, queue_stop, lane_change, camera_artifact)." + chr(10)',
    'STAGE_T_VLM_PROMPT += "Return ONLY valid JSON matching this schema: {physical_contact_observed: bool, structural_damage_or_deformation: bool, post_impact_displacement_or_rest: bool, benign_alternative: str, rationale: str}" + chr(10)',
    chr(10),
]

