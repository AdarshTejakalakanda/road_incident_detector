"""Synthetic Video Generator STRICTLY for Unit and Integration Testing.

NOTE: Per architectural constraints, synthetic data is isolated strictly to
unit/integration testing and must NEVER be mixed into real benchmark evaluation.
"""

import os
import cv2
import numpy as np


def create_synthetic_incident_video(
    output_path: str = "tests/test_data/synthetic_incident.mp4",
    num_frames: int = 120,
    fps: float = 30.0,
    width: int = 640,
    height: int = 480,
) -> str:
    """Generate a clean synthetic video simulating road roadway with vehicles for unit testing."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Road coordinates
    road_poly = np.array([[120, height], [520, height], [420, 150], [220, 150]], dtype=np.int32)

    # Vehicle 1: Moves forward, stops at frame 60
    # Vehicle 2: Moves toward vehicle 1
    v1_y = 400.0
    v2_y = 180.0

    for f_idx in range(num_frames):
        frame = np.full((height, width, 3), (60, 140, 60), dtype=np.uint8)  # Grass background

        # Draw road
        cv2.fillPoly(frame, [road_poly], (70, 70, 70))
        # Center lane dashed line
        for y_dash in range(160, height, 40):
            cv2.line(frame, (width // 2, y_dash), (width // 2, y_dash + 20), (255, 255, 255), 2)

        # Vehicle 1 (Car - Blue rectangle)
        if f_idx < 50:
            v1_y -= 2.0  # Moving forward
        # After frame 50, stationary!

        v1_x = 260
        cv2.rectangle(frame, (int(v1_x), int(v1_y)), (int(v1_x + 50), int(v1_y + 40)), (255, 0, 0), -1)
        cv2.putText(frame, "CAR_1", (v1_x, int(v1_y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Vehicle 2 (Car - Red rectangle) moving down (approaching)
        if f_idx > 30 and f_idx < 80:
            v2_y += 3.0

        v2_x = 270
        cv2.rectangle(frame, (int(v2_x), int(v2_y)), (int(v2_x + 50), int(v2_y + 40)), (0, 0, 255), -1)
        cv2.putText(frame, "CAR_2", (v2_x, int(v2_y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        writer.write(frame)

    writer.release()
    return os.path.abspath(output_path)


if __name__ == "__main__":
    out = create_synthetic_incident_video()
    print(f"Created synthetic test video: {out}")
