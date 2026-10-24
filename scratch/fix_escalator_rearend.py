with open('modules/scene_escalator.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = """        disp = (
            SceneEscalatorDisposition.ESCALATED_PHYSICAL_INCIDENT
            if escalated
            else SceneEscalatorDisposition.SUPPRESSED_BENIGN_AFTERMATH
        )"""

replacement = """        # Physical Consequence Safeguard: Rear-end collisions (Type 1 & Type 2) require real kinetic impact
        # (either sustained velocity drop dv_sust >= 4.0 km/h or kinetic arrest tau_rest >= 0.25 s).
        # Cruising vehicles with dv_sust < 4.0 and tau_rest < 0.25 are passing/adjacent-lane flow.
        if ctype in ('Type 1', 'Type 2') and dv_sust < 4.0 and tau_rest < 0.25:
            escalated = False

        disp = (
            SceneEscalatorDisposition.ESCALATED_PHYSICAL_INCIDENT
            if escalated
            else SceneEscalatorDisposition.SUPPRESSED_BENIGN_AFTERMATH
        )"""

text = text.replace(target, replacement)
with open('modules/scene_escalator.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Updated scene_escalator.py!')
