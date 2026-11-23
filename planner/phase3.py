"""Compact RGB-D view planning and a terminal approach to an observed object."""

from collections import deque
from math import dist

import numpy as np

from planner.flight_map import FlightMap
from planner.grid import NEIGHBORS


class Phase3Grid(FlightMap):
    def _cells(self, positions):
        """Deduplicate identical integer XYZ cells with a lexicographic scalar key."""
        if not positions.size:
            return np.empty((0, 3), dtype=np.int32)
        cells = np.floor(positions / self.voxel_m).astype(np.int64)
        minimum = cells.min(axis=0)
        offset = cells - minimum
        shape = tuple(offset.max(axis=0) + 1)
        unique = np.unique(np.ravel_multi_index(offset.T, shape))
        return np.column_stack(np.unravel_index(unique, shape)) + minimum


def reachable(grid, origin, max_steps):
    start = grid.anchor(origin)
    queue = deque([(start, 0)])
    costs = {start: 0}
    while queue:
        cell, cost = queue.popleft()
        if cost >= max_steps:
            continue
        for dx, dy, dz in NEIGHBORS:
            neighbor = cell[0]+dx, cell[1]+dy, cell[2]+dz
            if neighbor in grid.free and neighbor not in costs:
                costs[neighbor] = cost+1
                queue.append((neighbor, cost+1))
    return costs


def line_clear(grid, viewpoint, target):
    center = target.point
    samples = [center]
    samples.extend(target.surface[i] for i in range(0, len(target.surface), 12))
    visible = sum(not any(not target.box.contains(grid.center(cell), pad=0.10)
                          for cell in grid.segment_cells(viewpoint, sample) & grid.occupied)
                  for sample in samples)
    return visible * 2 >= len(samples)


def compact_goals(grid, origin, target, *, radius=1.85, avoided=(), approach=False, height=(0.8, 2.2)):
    """Return viable terminal cells in shortest-increment order."""
    costs = reachable(grid, origin, 26)
    candidates = []
    for cell, length in costs.items():
        point = grid.center(cell)
        separation = dist(point, target.point)
        if not 0.48 <= separation <= radius or not height[0] <= point[2] <= height[1]:
            continue
        if not approach and (length == 0 or dist(point, origin) < 0.25 or
                             any(dist(point, seen) < 0.35 for seen in avoided)):
            continue
        candidates.append((length, -point[2] if not approach else separation, cell))
    candidates.sort()
    # A* executes the selected route; evaluate visibility only for reachable candidates.
    return [cell for _, _, cell in candidates
            if approach or line_clear(grid, grid.center(cell), target)][:30]
