# Update geometry/engine.py
with open('road_surveillance/geometry/engine.py', 'r', encoding='utf-8') as f:
    engine_content = f.read()

target_engine = '''                    else:
                        a_long = float((speed_mps - prev_k.speed_mps) / dt_sec)
                        a_lat = 0.0'''

repl_engine = '''                    else:
                        a_long = float((speed_mps - prev_k.speed_mps) / dt_sec)
                        a_lat = 0.0

                    # Physical Traction Constraint Gate:
                    # Positive forward acceleration is traction-bounded (<= 6.0 m/s^2).
                    # Sudden positive acceleration spikes across 1-2 frames are detector box flickers.
                    # Deceleration during collision impact can reach -45.0 m/s^2.
                    MAX_PHYSICAL_FORWARD_ACCEL = 6.0
                    if a_long > MAX_PHYSICAL_FORWARD_ACCEL:
                        a_long = MAX_PHYSICAL_FORWARD_ACCEL'''

assert target_engine in engine_content, "target_engine not found"
engine_content = engine_content.replace(target_engine, repl_engine, 1)

with open('road_surveillance/geometry/engine.py', 'w', encoding='utf-8') as f:
    f.write(engine_content)
print("Updated road_surveillance/geometry/engine.py successfully")


# Update geometry/spatial_metrics.py
with open('road_surveillance/geometry/spatial_metrics.py', 'r', encoding='utf-8') as f:
    metrics_content = f.read()

target_metrics = '''                # Clamp physically extreme single-frame pixel jitter (max 500 px/s^2)
                px_acc = float(np.clip(raw_px_acc, -500.0, 500.0))'''

repl_metrics = '''                # Clamp physically: impact decel up to -500 px/s^2, but positive acceleration
                # is traction-bounded (max ~80 px/s^2 for vehicles) to eliminate forward flicker spikes
                px_acc = float(np.clip(raw_px_acc, -500.0, 80.0))'''

assert target_metrics in metrics_content, "target_metrics not found"
metrics_content = metrics_content.replace(target_metrics, repl_metrics, 1)

with open('road_surveillance/geometry/spatial_metrics.py', 'w', encoding='utf-8') as f:
    f.write(metrics_content)
print("Updated road_surveillance/geometry/spatial_metrics.py successfully")


# Update events/candidates/single_vehicle.py
with open('road_surveillance/events/candidates/single_vehicle.py', 'r', encoding='utf-8') as f:
    single_content = f.read()

target_single = '''            # 3. Physically plausible bounds: clamp extreme sensor jitter rather than discarding frame
            clamped_a = float(np.clip(metric_a, -self.max_physical_decel, self.max_physical_decel))'''

repl_single = '''            # 3. Physically plausible bounds: positive forward acceleration is traction-bounded (<= 5.0 m/s^2),
            # while severe collision impact deceleration can reach -45.0 m/s^2.
            # Abnormal positive acceleration spikes across 1-2 frames are treated as detector flickers.
            MAX_FORWARD_ACCEL = 5.0
            clamped_a = float(np.clip(metric_a, -self.max_physical_decel, MAX_FORWARD_ACCEL))'''

assert target_single in single_content, "target_single not found"
single_content = single_content.replace(target_single, repl_single, 1)

with open('road_surveillance/events/candidates/single_vehicle.py', 'w', encoding='utf-8') as f:
    f.write(single_content)
print("Updated road_surveillance/events/candidates/single_vehicle.py successfully")


# Update run_pipeline.py for subfolder by video_id
with open('run_pipeline.py', 'r', encoding='utf-8') as f:
    pipe_content = f.read()

target_pipe = '''    os.makedirs("artifacts", exist_ok=True)
    roi_dir = os.path.join(PROJECT_ROOT, "output", "roi_crops")
    os.makedirs(roi_dir, exist_ok=True)'''

repl_pipe = '''    os.makedirs("artifacts", exist_ok=True)
    # Save individual run separately in subfolder named with video_id under roi_crops
    roi_dir = os.path.join(PROJECT_ROOT, "output", "roi_crops", video_id)
    os.makedirs(roi_dir, exist_ok=True)'''

assert target_pipe in pipe_content, "target_pipe not found"
pipe_content = pipe_content.replace(target_pipe, repl_pipe, 1)

with open('run_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(pipe_content)
print("Updated run_pipeline.py for per-video roi_crops subfolder successfully")
