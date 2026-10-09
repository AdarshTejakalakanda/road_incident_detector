with open('modules/vlm_arbitrator.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_rest = 'has_unimputed_rest = bool(tau_rest_real >= 0.25 or (abs_accel >= 18.0 and dist <= 1.0))'
new_rest = 'has_unimputed_rest = bool(tau_rest_real >= 0.25)'

old_diss = 'has_finite_dissipation = bool(speed_drop >= 15.0 or s_dissip >= 0.70 or abs_accel >= 18.0)'
new_diss = 'has_finite_dissipation = bool(speed_drop >= 15.0 or s_dissip >= 0.70)'

assert old_rest in text, 'old_rest not found'
assert old_diss in text, 'old_diss not found'

text = text.replace(old_rest, new_rest)
text = text.replace(old_diss, new_diss)

with open('modules/vlm_arbitrator.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully enforced strict tau_rest and dissipation!')
