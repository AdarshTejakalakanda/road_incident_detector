"""Check exact values of Path C v3 on track 246 and track 229 at F835."""

# In extract_path_c_forensics:
# v_mps = 8.804 m/s (31.7 km/h)
# yaw_rate_inst was 64.54 deg/s (calculated from delta_theta_inst = 2.15 deg / (1/30) = 64.5 deg/s)
# in the forensic script, a_lat was calculated as v_curr * yaw_rate_inst_rad:
# 8.804 * radians(64.54) = 8.804 * 1.126 = 9.92 m/s^2.
# BUT in Path C v3:
# a_lat = v_t * (delta_theta_div / delta_t_5f)
# Here, delta_theta_div = 59.88 deg!
# delta_t_5f = 4 * 0.0333 = 0.1333 s!
# So (delta_theta_div / delta_t_5f) = 59.88 / 0.1333 = 449.1 deg/s = 7.839 rad/s!
# Therefore a_lat = 8.804 * 7.839 = 69.0 m/s^2 >> 10.0 m/s^2!
v_t = 8.804
delta_theta_5f = 59.88
dt_5f = 4.0 / 30.0
import math
yaw_rate_5f_rad = math.radians(delta_theta_5f) / dt_5f
a_lat_5f = v_t * yaw_rate_5f_rad
print(f"delta_theta_5f: {delta_theta_5f:.2f} deg")
print(f"yaw_rate_5f: {math.degrees(yaw_rate_5f_rad):.1f} deg/s")
print(f"a_lat_5f: {a_lat_5f:.2f} m/s^2")
print(f"Is a_lat_5f <= 10.0 m/s^2? {a_lat_5f <= 10.0}")
