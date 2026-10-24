with open('modules/scene_escalator.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace(
    'def __init__(self, tau_aftermath: float = 0.50):',
    'def __init__(self, tau_aftermath: float = 0.50, k_upstream: float = 3.0):'
)
text = text.replace(
    'self.tau = tau_aftermath',
    'self.tau = tau_aftermath\n        self.k_up = k_upstream'
)
text = text.replace(
    'phi_up = min(1.0, max(0.0, float(upstream_react)) / 1.00)',
    'phi_up = min(1.0, max(0.0, float(upstream_react)) / getattr(self, "k_up", 3.0))'
)

with open('modules/scene_escalator.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Done!')
