import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import pickle
import numpy as np
import pandas as pd

from road_surveillance.config import CollisionConfig
from road_surveillance.tracking import Track
from road_surveillance.geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence
from scripts.benchmark_iteration_b_detector_recovery import compute_iou, cluster_episodes, evaluate_video_candidates_temporal

print("=== TESTING PROTOTYPE REFINEMENT RULES ===")

# Load cached data
with open("cache_videos_iter_b_split_in_distribution_15.pkl", "rb") as f:
    in_dist_vids = pickle.load(f)
with open("cache_videos_iter_b_split_geo_aware_15.pkl", "rb") as f:
    geo_vids = pickle.load(f)

print(f"Loaded {len(in_dist_vids)} in-dist, {len(geo_vids)} geo-aware videos.")
