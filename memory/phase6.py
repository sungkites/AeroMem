"""CPU visibility, occupancy and local planning shared with Orin replay."""

from itertools import product
from math import dist, isfinite
from time import perf_counter

import numpy as np

from memory.phase3 import MemoryFastPath
from sim.geometry import dot, sub


OCCLUSION_DELTA_M = 0.065


def line_of_sight(old, frame):
    """Classify foreground obstruction, coverage, or insufficient evidence."""
    projected = {}
    for point in old.surface:
        pixel = frame.camera.pixel(frame.pose, point)
        if pixel is not None:
            depth = dot(sub(point, frame.pose.position), frame.pose.forward)
            projected[pixel] = min(depth, projected.get(pixel, depth))
    if len(projected) < 12:
        for fractions in product((0.2, 0.5, 0.8), repeat=3):
            point = tuple(a + f*(b-a) for a, b, f in zip(old.box.low, old.box.high, fractions))
            pixel = frame.camera.pixel(frame.pose, point)
            if pixel is None or pixel in projected:
                continue
            ray = frame.camera.direction(frame.pose, *pixel)
            interval = old.box.ray_interval(frame.pose.position, ray)
            if interval is not None:
                projected[pixel] = interval[0]
    residuals = []
    for (u, v), expected in projected.items():
        observed = frame.depth[v, u]
        if isfinite(observed) and 0 < observed <= frame.camera.max_depth_m:
            residuals.append(observed-expected)
    if len(residuals) < 6:
        return "unknown"
    blocked = sum(delta < -OCCLUSION_DELTA_M for delta in residuals) / len(residuals)
    if blocked >= 0.68:
        return "occluded"
    return "line_of_sight" if blocked <= 0.35 else "unknown"


class VisibilityFastPath(MemoryFastPath):
    def update(self, frame, query=(), goal=None, active=None, stride=16):
        start = perf_counter()
        self.grid.integrate(frame, stride=stride, limit=4.5 if stride >= 16 else 6.0)
        origin = self.grid.anchor(frame.pose.position)
        visible = [(key, obs) for key, obs in self.tracks
                   if frame.camera.pixel(frame.pose, obs.point) is not None]
        visible.sort(key=lambda pair: (pair[0] != active, dist(pair[1].point, frame.pose.position)))
        evidence = {key:line_of_sight(obs,frame) for key,obs in visible[:5]}
        retrieved = []
        if self.embeddings is not None and query and len(query) == self.embeddings.shape[1]:
            vector = np.asarray(query, dtype=np.float32)
            scores = (self.embeddings * vector).sum(axis=1) / max(np.linalg.norm(vector), 1e-8)
            retrieved = [self.keys[i] for i in np.argsort(-scores)[:5]]
        local = goal is not None and sum(abs(a-b) for a,b in zip(origin,goal)) <= 3
        route = self.grid.path(origin,goal) if local else None
        self.milliseconds.append((perf_counter()-start)*1000)
        return evidence,retrieved,route
