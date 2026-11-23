"""Metric world coordinates (Z up) and right/down/forward camera axes."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, dist, isfinite, sin

Vec3 = tuple[float, float, float]


def add(a: Vec3, b: Vec3) -> Vec3:
    return tuple(x + y for x, y in zip(a, b))


def sub(a: Vec3, b: Vec3) -> Vec3:
    return tuple(x - y for x, y in zip(a, b))


def scale(vector: Vec3, factor: float) -> Vec3:
    return tuple(value * factor for value in vector)


def dot(a: Vec3, b: Vec3) -> float:
    return sum(x * y for x, y in zip(a, b))


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def unit(vector: Vec3) -> Vec3:
    length = dist(vector, (0, 0, 0))
    if length == 0:
        raise ValueError("cannot normalize a zero vector")
    return scale(vector, 1 / length)


@dataclass(frozen=True)
class Box:
    low: Vec3
    high: Vec3

    def __post_init__(self):
        if (len(self.low) != 3 or len(self.high) != 3
                or not all(isfinite(a) and isfinite(b) and a < b
                           for a, b in zip(self.low, self.high))):
            raise ValueError("a box needs three finite, strictly ordered bounds")

    @property
    def center(self) -> Vec3:
        return scale(add(self.low, self.high), 0.5)

    def contains(self, point: Vec3, pad: float = 0.0) -> bool:
        return all(low - pad <= x <= high + pad for x, low, high in zip(point, self.low, self.high))

    def iou(self, other: "Box") -> float:
        overlap = 1.0
        volume = other_volume = 1.0
        for low, high, other_low, other_high in zip(self.low, self.high, other.low, other.high):
            overlap *= max(0.0, min(high, other_high) - max(low, other_low))
            volume *= high - low
            other_volume *= other_high - other_low
        return overlap / (volume + other_volume - overlap)

    def ray_interval(self, origin: Vec3, direction: Vec3, limit=float("inf"), pad=0.0):
        """Return entry/exit parameters along origin + t * direction."""
        near, far = 0.0, limit
        for x, delta, low, high in zip(origin, direction, self.low, self.high):
            low, high = low - pad, high + pad
            if abs(delta) < 1e-12:
                if not low <= x <= high:
                    return None
                continue
            entry, exit_ = sorted(((low - x) / delta, (high - x) / delta))
            near, far = max(near, entry), min(far, exit_)
            if near > far:
                return None
        return near, far


@dataclass(frozen=True)
class Pose:
    position: Vec3
    right: Vec3
    down: Vec3
    forward: Vec3

    @classmethod
    def look_at(cls, position: Vec3, target: Vec3, roll_rad: float = 0.0) -> "Pose":
        forward = unit(sub(target, position))
        up = (0, 0, 1) if abs(forward[2]) < 0.9999 else (0, 1, 0)
        right = unit(cross(forward, up))
        down = cross(forward, right)
        return cls(position,
                   add(scale(right, cos(roll_rad)), scale(down, sin(roll_rad))),
                   add(scale(down, cos(roll_rad)), scale(right, -sin(roll_rad))),
                   forward)


@dataclass(frozen=True)
class Camera:
    width: int = 64
    height: int = 48
    focal_px: float = 40.0
    max_depth_m: float = 12.0

    def direction(self, pose: Pose, u: int, v: int) -> Vec3:
        x = (u - (self.width - 1) / 2) / self.focal_px
        y = (v - (self.height - 1) / 2) / self.focal_px
        return add(add(pose.forward, scale(pose.right, x)), scale(pose.down, y))

    def pixel(self, pose: Pose, world: Vec3) -> tuple[int, int] | None:
        delta = sub(world, pose.position)
        depth = dot(delta, pose.forward)
        if not 0 < depth <= self.max_depth_m:
            return None
        u = round(dot(delta, pose.right) * self.focal_px / depth + (self.width - 1) / 2)
        v = round(dot(delta, pose.down) * self.focal_px / depth + (self.height - 1) / 2)
        return (u, v) if 0 <= u < self.width and 0 <= v < self.height else None
