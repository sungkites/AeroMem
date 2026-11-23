"""Turn foreground masks and optical-axis depth into world-frame observations."""

from dataclasses import dataclass
from itertools import product
from time import monotonic

import numpy as np

from memory.store import Detection
from sim.geometry import Box, sub, unit
from vlm.wire import request


def rays(frame, u, v):
    camera = frame.camera
    x = (u - (camera.width - 1) / 2) / camera.focal_px
    y = (v - (camera.height - 1) / 2) / camera.focal_px
    return (np.asarray(frame.pose.forward) + x[..., None] * np.asarray(frame.pose.right)
            + y[..., None] * np.asarray(frame.pose.down))


def decode_mask(shape, rows):
    mask = np.zeros(shape, dtype=bool)
    for row, edges in rows:
        for a, b in zip(edges[::2], edges[1::2]):
            mask[row, a:b] = True
    return mask


def erode(mask):
    border = np.pad(mask, 1, constant_values=False)
    h, w = mask.shape
    return np.logical_and.reduce([border[y:y + h, x:x + w] for y in range(3) for x in range(3)])


def lift(frame, record, region):
    mask = erode(decode_mask(frame.depth.shape, record["mask"]))
    valid = mask & np.isfinite(frame.depth) & (frame.depth > 0)
    v, u = np.nonzero(valid)
    if len(u) < 12:
        return None
    depth = frame.depth[v, u]
    median = np.median(depth)
    spread = np.median(np.abs(depth - median))
    inliers = np.abs(depth - median) <= max(0.10, min(0.45, 3.5 * float(spread)))
    u, v, depth = u[inliers], v[inliers], depth[inliers]
    if len(u) < 12:
        return None
    cloud = np.asarray(frame.pose.position) + rays(frame, u, v) * depth[:, None]
    lo, hi = np.quantile(cloud, [0.05, 0.95], axis=0)
    center = tuple(np.median(cloud, axis=0).tolist())
    box = Box(tuple((lo - 0.035).tolist()), tuple((hi + 0.035).tolist()))
    samples = cloud[np.linspace(0, len(cloud) - 1, min(len(cloud), 36), dtype=int)]
    return Detection(record["label"], box, region, tuple(record["embedding"]),
                     center, unit(sub(frame.pose.position, center)),
                     tuple(tuple(p) for p in samples.tolist()))


@dataclass
class Perception:
    observations: list
    masks: list
    latency_ms: float
    worker_ms: float
    cuda: dict


def memory_roi(frame, box):
    pixels = [frame.camera.pixel(frame.pose, corner) for corner in product(*zip(box.low, box.high))]
    if any(pixel is None for pixel in pixels):
        return None
    columns, rows = zip(*pixels)
    u0, u1, v0, v1 = min(columns), max(columns), min(rows), max(rows)
    pad = max(22, round(0.28 * max(u1 - u0, v1 - v0)))
    return (max(0, u0 - pad), max(0, v0 - pad),
            min(frame.camera.width, u1 + pad + 1), min(frame.camera.height, v1 + pad + 1))


def perceive(socket_path, frame, vocabulary, region, box=None, zoom=False):
    start = monotonic()
    roi = memory_roi(frame, box) if box is not None else None
    if roi is None and zoom:
        w, h = frame.camera.width, frame.camera.height
        roi = (w // 5, h // 5, 4 * w // 5, 4 * h // 5)
    x0, y0, x1, y1 = roi or (0, 0, frame.camera.width, frame.camera.height)
    response = request(socket_path, "detect", frame.rgb[y0:y1, x0:x1], vocabulary=vocabulary)
    points = []
    for record in response["detections"]:
        if roi is not None:
            record["box"] = [coordinate + shift for coordinate, shift in
                             zip(record["box"], (x0, y0, x0, y0))]
            record["mask"] = [[row + y0, [edge + x0 for edge in edges]]
                              for row, edges in record["mask"]]
        detection = lift(frame, record, region)
        if detection is not None:
            points.append(detection)
    return Perception(points, response["detections"], (monotonic() - start) * 1000,
                      response["worker_ms"], response["cuda"])
