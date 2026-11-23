from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import product
from math import isfinite, sqrt
from pathlib import Path
from tempfile import NamedTemporaryFile

from sim.backend import Frame
from sim.geometry import Box, Vec3, dot, sub, unit


@dataclass(frozen=True)
class Detection:
    label: str
    box: Box
    region: str
    embedding: tuple[float, ...] = ()
    centroid: Vec3 | None = None
    view_normal: Vec3 | None = None
    surface: tuple[Vec3, ...] = ()

    @property
    def point(self) -> Vec3:
        return self.centroid if self.centroid is not None else self.box.center


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float | None:
    if a and len(a) == len(b):
        norm_product = sqrt(sum(x * x for x in a) * sum(y * y for y in b))
        if norm_product > 0:
            return sum(x * y for x, y in zip(a, b)) / norm_product
    return None


def similarity(a: Detection, b: Detection) -> float:
    score = cosine(a.embedding, b.embedding)
    if score is not None:
        return score
    return float(a.label.strip().casefold() == b.label.strip().casefold())


def depth_evidence(box: Box, frame: Frame, tolerance_m: float = 0.06,
                   clear_fraction: float = 0.8) -> str:
    """Classify multiple rays that actually intersect the remembered volume."""
    pixels = set()
    for fractions in product((0.2, 0.5, 0.8), repeat=3):
        point = tuple(a + (b - a) * f for a, b, f in zip(box.low, box.high, fractions))
        pixel = frame.camera.pixel(frame.pose, point)
        if pixel is not None:
            pixels.add(pixel)
    blocked = clear = unknown = 0
    for u, v in pixels:
        direction = frame.camera.direction(frame.pose, u, v)
        interval = box.ray_interval(frame.pose.position, direction)
        if interval is None:
            continue
        near, far = interval
        depth = frame.depth[v][u]
        if depth is None or not isfinite(depth) or not 0 < depth <= frame.camera.max_depth_m:
            unknown += 1
        elif depth < near - tolerance_m:
            blocked += 1
        elif depth > far + tolerance_m:
            clear += 1
        else:
            unknown += 1
    total = blocked + clear + unknown
    if total >= 3 and blocked / total >= 0.8:
        return "occluded"
    full_volume_in_view = all(frame.camera.pixel(frame.pose, corner) is not None
                              for corner in product(*zip(box.low, box.high)))
    if total >= 3 and clear / total >= clear_fraction and full_volume_in_view:
        return "clear"
    return "unknown"


def revisit_evidence(old: Detection, frame: Frame) -> str:
    """Compare remembered foreground-surface samples to a real revisit frame."""
    old_view = old.view_normal
    new_view = sub(frame.pose.position, old.point)
    if old_view is not None and dot(new_view, new_view) > 0 and dot(old_view, unit(new_view)) >= 0.7:
        projected = {}
        for point in old.surface:
            pixel = frame.camera.pixel(frame.pose, point)
            if pixel is not None:
                expected = dot(sub(point, frame.pose.position), frame.pose.forward)
                projected[pixel] = min(expected, projected.get(pixel, expected))
        residuals = []
        for (u, v), expected in projected.items():
            depth = frame.depth[v][u]
            if depth is not None and isfinite(depth) and 0 < depth <= frame.camera.max_depth_m:
                residuals.append(depth - expected)
        if len(residuals) >= max(6, len(old.surface) * 0.55):
            if sum(value < -0.075 for value in residuals) / len(residuals) >= 0.8:
                return "occluded"
            if sum(value > 0.095 for value in residuals) / len(residuals) >= 0.72:
                return "clear"
    return depth_evidence(old.box, frame, tolerance_m=0.055, clear_fraction=2/3)


@dataclass(frozen=True)
class Event:
    sortie: str
    viewpoint: Vec3
    state: str
    evidence: str
    detection: Detection | None = None


@dataclass
class Track:
    key: int
    observation: Detection
    history: list[Event] = field(default_factory=list)

    @property
    def state(self) -> str:
        return self.history[-1].state


class SpatialMemory:
    """Site → region → track → sortie evidence. One writer per snapshot file."""

    def __init__(self, site: str):
        self.site = site
        self.tracks: dict[int, Track] = {}
        self.next_key = 1

    def add_track(self, sortie: str, viewpoint: Vec3, observation: Detection,
                  evidence: str = "new_track") -> Track:
        key = self.next_key
        self.next_key += 1
        track = Track(key, observation, [Event(sortie, viewpoint, "observed", evidence, observation)])
        self.tracks[key] = track
        return track

    def retrieve(self, label: str, region: str | None = None) -> list[Track]:
        return [track for track in self.tracks.values()
                if track.observation.label.strip().casefold() == label.strip().casefold()
                and (region is None or track.observation.region == region)]

    def semantic_lookup(self, query: tuple[float, ...], region: str | None = None
                        ) -> list[tuple[float, Track]]:
        matches = []
        for track in self.tracks.values():
            if region is None or track.observation.region == region:
                score = cosine(query, track.observation.embedding)
                if score is not None:
                    matches.append((score, track))
        return sorted(matches, key=lambda match: (-match[0], match[1].key))

    def update(self, sortie: str, frame: Frame, detections: list[Detection]) -> None:
        unmatched = set(self.tracks)
        available = set(range(len(detections)))

        def match(key, index, state, evidence):
            track = self.tracks[key]
            track.observation = detections[index]
            track.history.append(Event(sortie, frame.pose.position, state, evidence, detections[index]))
            unmatched.remove(key)
            available.remove(index)

        pairs = []
        for key in sorted(unmatched):
            old = self.tracks[key].observation
            for index, new in enumerate(detections):
                overlap, semantic = old.box.iou(new.box), similarity(old, new)
                if overlap >= 0.25 and semantic >= 0.75:
                    pairs.append((-overlap, -semantic, key, index))
        for _, _, key, index in sorted(pairs):
            if key in unmatched and index in available:
                match(key, index, "observed", "iou_semantic")

        evidence = {key: depth_evidence(self.tracks[key].observation.box, frame)
                    for key in sorted(unmatched)}
        reid = []
        for key in sorted(unmatched):
            old = self.tracks[key].observation
            for index in sorted(available):
                new = detections[index]
                if old.box.iou(new.box) < 0.25 and similarity(old, new) >= 0.85:
                    reid.append((key, index))
        track_options = Counter(key for key, _ in reid)
        detection_options = Counter(index for _, index in reid)
        for key, index in reid:
            if evidence[key] == "clear" and track_options[key] == detection_options[index] == 1:
                match(key, index, "relocated", "clear_and_unique_semantic_match")

        for key in sorted(unmatched):
            ray_state = evidence[key]
            state = "missing" if ray_state == "clear" else ray_state
            self.tracks[key].history.append(Event(sortie, frame.pose.position, state, ray_state))
        for index in sorted(available):
            self.add_track(sortie, frame.pose.position, detections[index])

    def save(self, path: Path) -> None:
        payload = {"schema_version": 1, "site": self.site, "next_key": self.next_key,
                   "tracks": [asdict(track) for track in self.tracks.values()]}
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temp:
            pending = Path(temp.name)
            try:
                json.dump(payload, temp, ensure_ascii=False, allow_nan=False)
                temp.flush()
                os.fsync(temp.fileno())
                temp.close()
                os.replace(pending, path)
            finally:
                temp.close()
                pending.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> "SpatialMemory":
        def detection(data):
            box = data["box"]
            return Detection(data["label"], Box(tuple(box["low"]), tuple(box["high"])),
                             data["region"], tuple(data["embedding"]),
                             tuple(data["centroid"]) if data.get("centroid") is not None else None,
                             tuple(data["view_normal"]) if data.get("view_normal") is not None else None,
                             tuple(tuple(point) for point in data.get("surface", ())))

        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["schema_version"] != 1:
            raise ValueError("unsupported memory schema")
        memory = cls(payload["site"])
        for item in payload["tracks"]:
            events = [Event(event["sortie"], tuple(event["viewpoint"]), event["state"],
                            event["evidence"], detection(event["detection"])
                            if event["detection"] is not None else None)
                      for event in item["history"]]
            track = Track(item["key"], detection(item["observation"]), events)
            memory.tracks[track.key] = track
        memory.next_key = max(payload.get("next_key", 1), max(memory.tracks, default=0) + 1)
        return memory
