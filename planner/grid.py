from __future__ import annotations

from heapq import heappop, heappush
from math import ceil, dist, floor, isfinite

from sim.backend import Frame
from sim.geometry import Vec3, add, scale, sub

Cell = tuple[int, int, int]
NEIGHBORS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


class OccupancyGrid:
    def __init__(self, voxel_m: float = 0.5):
        if not isfinite(voxel_m) or voxel_m <= 0:
            raise ValueError("voxel size must be finite and positive")
        self.voxel_m = voxel_m
        self.free: set[Cell] = set()
        self.occupied: set[Cell] = set()

    def cell(self, point: Vec3) -> Cell:
        return tuple(floor(value / self.voxel_m) for value in point)

    def center(self, cell: Cell) -> Vec3:
        return tuple((index + 0.5) * self.voxel_m for index in cell)

    def segment_cells(self, start: Vec3, end: Vec3) -> set[Cell]:
        steps = max(1, ceil(4 * dist(start, end) / self.voxel_m))
        return {self.cell(add(start, scale(sub(end, start), step / steps)))
                for step in range(steps + 1)}

    def integrate(self, frame: Frame, stride: int = 2) -> None:
        if stride < 1:
            raise ValueError("stride must be positive")
        origin = frame.pose.position
        occupied, free = set(), {self.cell(origin)}
        for v in range(0, frame.camera.height, stride):
            for u in range(0, frame.camera.width, stride):
                depth = frame.depth[v][u]
                if depth is None or not isfinite(depth) or not 0 < depth <= frame.camera.max_depth_m:
                    continue
                endpoint = add(origin, scale(frame.camera.direction(frame.pose, u, v), depth))
                occupied.add(self.cell(endpoint))
                free.update(self.segment_cells(origin, endpoint))
        # Keep the latest observed state. Surface hits win within a frame.
        self.occupied.difference_update(free - occupied)
        self.occupied.update(occupied)
        self.free.update(free - occupied)
        self.free.difference_update(occupied)

    def path(self, start: Cell, goal: Cell) -> list[Cell] | None:
        if start not in self.free or goal not in self.free:
            return None
        frontier = [(0, 0, start)]
        best = {start: 0}
        parent = {}
        while frontier:
            _, cost, current = heappop(frontier)
            if cost != best[current]:
                continue
            if current == goal:
                route = [goal]
                while route[-1] != start:
                    route.append(parent[route[-1]])
                return route[::-1]
            for offset in NEIGHBORS:
                neighbor = tuple(i + j for i, j in zip(current, offset))
                if neighbor in self.free and cost + 1 < best.get(neighbor, float("inf")):
                    best[neighbor] = cost + 1
                    parent[neighbor] = current
                    heuristic = sum(abs(i - j) for i, j in zip(neighbor, goal))
                    heappush(frontier, (cost + 1 + heuristic, cost + 1, neighbor))
        return None
