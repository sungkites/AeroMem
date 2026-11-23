"""Depth-only frame updates and appearance checks for relocated instances."""

from dataclasses import dataclass
from math import atan2, degrees, dist, hypot, log, prod
from time import perf_counter

import numpy as np

from memory.lift import decode_mask
from memory.store import cosine, revisit_evidence


class KeyframeGate:
    def __init__(self, distance=0.35, angle=18):
        self.previous = None
        self.distance, self.angle = distance, angle

    def trigger(self, pose, event=False):
        if self.previous is not None and not event:
            def angles(current):
                x, y, z = current.forward
                return degrees(atan2(y, x)), degrees(atan2(z, hypot(x, y)))
            yaw0, pitch0 = angles(self.previous)
            yaw1, pitch1 = angles(pose)
            turn = abs((yaw1-yaw0+180) % 360-180)
            if (dist(self.previous.position, pose.position) < self.distance
                    and max(turn, abs(pitch1-pitch0)) < self.angle):
                return False
        self.previous = pose
        return True


class FlightIO:
    """Navigation capability: synchronized camera frames, pose and checked motion."""
    __slots__ = ("_pose", "_observe", "_move")

    def __init__(self, simulator):
        self._pose = lambda: simulator.pose
        self._observe = simulator.observe
        self._move = simulator.move

    @property
    def pose(self):
        return self._pose()

    def observe(self):
        return self._observe()

    def move(self, pose):
        return self._move(pose)


class MemoryFastPath:
    def __init__(self, grid):
        self.grid = grid
        self.tracks = ()
        self.embeddings = None
        self.milliseconds = []

    def bind(self, tracks):
        self.tracks = tuple(tracks)
        embeddings = [(k, d.embedding) for k, d in self.tracks if d.embedding]
        if embeddings:
            self.keys, vectors = zip(*embeddings)
            array = np.asarray(vectors, dtype=np.float32)
            self.embeddings = array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)
        else:
            self.keys, self.embeddings = (), None

    def update(self, frame, query=(), goal=None, active=None, stride=16):
        start = perf_counter()
        self.grid.integrate(frame, stride=stride, limit=4.5 if stride >= 16 else 6.0)
        origin = self.grid.anchor(frame.pose.position)
        visible = [(key, obs) for key, obs in self.tracks
                   if frame.camera.pixel(frame.pose, obs.point) is not None]
        visible.sort(key=lambda pair: (pair[0] != active, dist(pair[1].point, frame.pose.position)))
        # A bounded local working set keeps the onboard update independent of global map size.
        evidence = {key: {"occluded": "Occluded", "clear": "Relocated",
                          "unknown": "Unknown", "confirmed": "Confirmed"}[surface_state(obs, frame)]
                    for key, obs in visible[:5]}
        retrieved = []
        if self.embeddings is not None and query and len(query) == self.embeddings.shape[1]:
            vector = np.asarray(query, dtype=np.float32)
            scores = (self.embeddings * vector).sum(axis=1) / max(np.linalg.norm(vector), 1e-8)
            retrieved = [self.keys[i] for i in np.argsort(-scores)[:5]]
        path = self.grid.path(origin, goal) if goal is not None and sum(abs(a-b) for a,b in zip(origin,goal)) <= 3 else None
        self.milliseconds.append((perf_counter() - start) * 1000)
        return evidence, retrieved, path


def surface_state(old, frame):
    evidence = revisit_evidence(old, frame)
    if evidence != "unknown" or not old.surface:
        return evidence
    samples = np.asarray(old.surface) - np.asarray(frame.pose.position)
    ray_depth = np.einsum("ij,j->i", samples, frame.pose.forward)
    supported = []
    for point, expected in zip(old.surface, ray_depth):
        pixel = frame.camera.pixel(frame.pose, point)
        if pixel is not None:
            u, v = pixel
            measured = frame.depth[v, u]
            supported.append(measured is not None and np.isfinite(measured) and abs(measured - expected) <= 0.065)
    return "confirmed" if len(supported) >= max(6, len(old.surface) * 0.55) and sum(supported)/len(supported) >= 0.70 else "unknown"


def dimensions(detection):
    return sorted(high - low for low, high in zip(detection.box.low, detection.box.high))


def compatible_scale(first, second):
    a, b = dimensions(first), dimensions(second)
    ratios = [y/x for x, y in zip(a, b)]
    return (all(0.46 <= r <= 2.0 for r in ratios) and 0.27 <= prod(ratios) <= 3.1
            and max(ratios)/min(ratios) <= 2.80)


@dataclass
class Appearance:
    detection: object
    pose: object
    crop: object
    frame: object = None


def appearance(frame, detection, masks):
    """Select the foreground mask supporting this lifted detection."""
    pixel = frame.camera.pixel(frame.pose, detection.point)
    if pixel is None:
        return None
    u, v = pixel
    matches = [item for item in masks if item["label"] == detection.label
               and item["box"][0] <= u <= item["box"][2] and item["box"][1] <= v <= item["box"][3]]
    if not matches:
        return None
    best = min(matches, key=lambda item: (dist((u, v), ((item["box"][0]+item["box"][2])*0.5,
                                                        (item["box"][1]+item["box"][3])*0.5)),
                                          -item["score"]))
    x0, y0, x1, y1 = np.rint(best["box"]).astype(int)
    x0, x1 = max(0, x0-4), min(frame.camera.width, x1+5)
    y0, y1 = max(0, y0-4), min(frame.camera.height, y1+5)
    mask = decode_mask(frame.depth.shape, best["mask"])[y0:y1, x0:x1]
    rgb = frame.rgb[y0:y1, x0:x1]
    if mask.sum() < 12:
        return None
    return Appearance(detection, frame.pose, np.where(mask[..., None], rgb, 111).astype(np.uint8), frame)


def independent_views(first, second):
    return dist(first.pose.position, second.pose.position) >= 0.23


def rank_instances(history, candidates):
    """Require supporting RGB-D in two distinct camera positions before relinking."""
    ranked = []
    for group in candidates:
        valid = [a for a in group if any(compatible_scale(h.detection, a.detection) for h in history)]
        for i, first in enumerate(valid):
            for second in valid[i+1:]:
                if not independent_views(first, second) or not compatible_scale(first.detection, second.detection):
                    continue
                view_scores = []
                reference_scores = {}
                for sample in (first, second):
                    scores = [(cosine(h.detection.embedding, sample.detection.embedding), j)
                              for j, h in enumerate(history) if compatible_scale(h.detection, sample.detection)]
                    scores = [(score, j) for score, j in scores if score is not None]
                    if not scores:
                        break
                    score, index = max(scores)
                    reference_scores[index] = reference_scores.get(index, 0) + score
                    view_scores.append(score)
                if len(view_scores) != 2 or min(view_scores) < 0.58:
                    continue
                if (cosine(first.detection.embedding, second.detection.embedding) or 0) < 0.68:
                    continue
                known = history[max(reference_scores, key=reference_scores.get)]
                sample = (first, second)[int(view_scores[1] >= view_scores[0])]
                scale_error = sum(abs(log(a/b)) for a,b in zip(dimensions(known.detection), dimensions(sample.detection)))
                confidence = sum(view_scores)/2 - 0.06 * scale_error
                ranked.append((confidence, group, known, sample))
    ranked.sort(key=lambda row: -row[0])
    best = {}
    for row in ranked:
        best.setdefault(id(row[1]), row)
    return list(best.values())


def pick_instance(ranked, verifier, force_verify=False):
    if not ranked:
        return None
    score, _, historical, current = ranked[0]
    ambiguous = (force_verify or score < 0.83 or
                 (len(ranked) > 1 and score - ranked[1][0] < 0.10))
    return (current if verifier(historical, current) else None) if ambiguous else current
