"""Empirical evaluation of persistence, net displacements, and visual characteristics.
"""

import json
import numpy as np

with open("scratch/path_c_forensic_results.json") as f:
    data = json.load(f)

n_persists = [d["n_persist"] for d in data]
dtheta_insts = [d["delta_theta_inst"] for d in data]
yaw_rates = [d["yaw_rate_inst"] for d in data]
dtheta_3fs = [d["delta_theta_3f"] for d in data]
dtheta_5fs = [d["delta_theta_5f"] for d in data]
dtheta_10fs = [d["delta_theta_10f"] for d in data]
a_lats = [d["a_lat"] for d in data]

print(f"N_persist distribution across 19 records:")
print(f"  Min: {min(n_persists)}")
print(f"  Median: {float(np.median(n_persists))}")
print(f"  Max: {max(n_persists)}")

print(f"\nDelta theta inst distribution:")
print(f"  Min: {min(dtheta_insts):.2f} deg, Median: {float(np.median(dtheta_insts)):.2f} deg, Max: {max(dtheta_insts):.2f} deg")

print(f"\nDelta theta 5f net distribution:")
print(f"  Min: {min(dtheta_5fs):.2f} deg, Median: {float(np.median(dtheta_5fs)):.2f} deg, Max: {max(dtheta_5fs):.2f} deg")

print(f"\nDelta theta 10f net distribution:")
print(f"  Min: {min(dtheta_10fs):.2f} deg, Median: {float(np.median(dtheta_10fs)):.2f} deg, Max: {max(dtheta_10fs):.2f} deg")

print(f"\na_lat distribution:")
print(f"  Min: {min(a_lats):.2f} m/s^2, Median: {float(np.median(a_lats)):.2f} m/s^2, Max: {max(a_lats):.2f} m/s^2")
