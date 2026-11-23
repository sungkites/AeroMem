"""Exercise the depth visibility rule with three synthetic observations."""

import numpy as np

from memory.phase6 import line_of_sight
from memory.store import Detection
from sim.backend import Frame
from sim.geometry import Box, Camera, Pose


def main():
    x, z = 1.25, 1.0
    points = tuple(
        (x, col * 0.023, z + row * 0.023)
        for col in range(-3, 4)
        for row in range(-2, 3)
    )
    target = Detection(
        "box",
        Box((1.21, -0.10, 0.92), (1.37, 0.10, 1.08)),
        "example",
        (1.0, 0.0),
        (x, 0.0, z),
        surface=points,
    )
    camera = Camera(64, 48, 140, 9)
    pose = Pose.look_at((0.0, 0.0, z), (1.0, 0.0, z))
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    for depth, expected in [
        (0.80, "occluded"),
        (1.25, "line_of_sight"),
        (float("nan"), "unknown"),
    ]:
        frame = Frame(camera, pose, rgb, np.full((48, 64), depth, dtype=np.float32))
        state = line_of_sight(target, frame)
        if state != expected:
            raise RuntimeError(f"Expected {expected}, got {state}")
        print(state)


if __name__ == "__main__":
    main()
