from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from .geometry import Box, Camera, Pose, sub

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class Frame:
    camera: Camera
    pose: Pose
    rgb: tuple[tuple[RGB, ...], ...]
    # Optical-axis Z in meters. None means there was no valid return.
    depth: tuple[tuple[float | None, ...], ...]


class Simulator(Protocol):
    def move(self, pose: Pose) -> bool: ...
    def observe(self) -> Frame: ...
    def place_object(self, key: str, box: Box, color: RGB) -> None: ...
    def relocate_object(self, key: str, box: Box) -> None: ...


@dataclass(frozen=True)
class SceneObject:
    box: Box
    color: RGB


class BoxSimulator:
    """Deterministic RGB-D raycasting and swept collision for scaffold checks."""

    def __init__(self, pose: Pose, camera: Camera = Camera(), radius_m: float = 0.12):
        if radius_m < 0:
            raise ValueError("radius_m must be nonnegative")
        self.pose, self.camera, self.radius_m = pose, camera, radius_m
        self.objects: dict[str, SceneObject] = {}

    def place_object(self, key: str, box: Box, color: RGB) -> None:
        if key in self.objects:
            raise ValueError(f"scene object already exists: {key}")
        self.objects[key] = SceneObject(box, color)

    def relocate_object(self, key: str, box: Box) -> None:
        self.objects[key] = replace(self.objects[key], box=box)

    def move(self, pose: Pose) -> bool:
        direction = sub(pose.position, self.pose.position)
        if any(obj.box.ray_interval(self.pose.position, direction, limit=1.0,
                                    pad=self.radius_m) is not None
               for obj in self.objects.values()):
            return False
        self.pose = pose
        return True

    def observe(self) -> Frame:
        rgb, depth = [], []
        for v in range(self.camera.height):
            rgb_row, depth_row = [], []
            for u in range(self.camera.width):
                direction = self.camera.direction(self.pose, u, v)
                hits = []
                for obj in self.objects.values():
                    interval = obj.box.ray_interval(self.pose.position, direction,
                                                    limit=self.camera.max_depth_m)
                    if interval is not None and interval[0] > 0:
                        hits.append((interval[0], obj.color))
                distance, color = min(hits, default=(None, (0, 0, 0)))
                rgb_row.append(color)
                depth_row.append(distance)
            rgb.append(tuple(rgb_row))
            depth.append(tuple(depth_row))
        return Frame(self.camera, self.pose, tuple(rgb), tuple(depth))
