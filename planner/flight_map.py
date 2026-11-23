"""RGB-D occupancy mapping for the near-term flight envelope."""

import numpy as np
from scipy.ndimage import distance_transform_edt

from memory.lift import rays
from planner.grid import OccupancyGrid
from sim.geometry import Vec3


class FlightMap(OccupancyGrid):
    def __init__(self, voxel_m=0.24):
        super().__init__(voxel_m)
        self._signed_field = None

    def integrate(self, frame, stride=6, limit=6.0):
        v, u = np.mgrid[0:frame.camera.height:stride, 0:frame.camera.width:stride]
        depth = frame.depth[v, u]
        valid = np.isfinite(depth) & (depth > 0) & (depth < frame.camera.max_depth_m)
        depth, u, v = depth[valid], u[valid], v[valid]
        origin = np.asarray(frame.pose.position)
        directions = rays(frame, u, v)
        scale = self.voxel_m
        surfaces = origin + (directions * depth[:, None])
        occupied = self._cells(surfaces)
        z_centers = lambda cells: (cells[:, 2] + 0.5) * scale
        occupied = set(map(tuple, occupied[(z_centers(occupied) >= 0.7)
                                           & (z_centers(occupied) <= 2.3)].tolist()))

        interval = scale / 3
        steps = np.arange(interval, limit, interval)
        visible = steps[None, :] < (depth[:, None] - scale * 0.42)
        samples = origin + (directions[:, None, :] * steps[None, :, None])[visible]
        cells = self._cells(samples)
        free = set(map(tuple, cells[(z_centers(cells) >= 0.8) & (z_centers(cells) <= 2.2)].tolist()))
        free.add(self.cell(frame.pose.position))
        self.occupied.difference_update(free - occupied)
        self.occupied.update(occupied)
        self.free.update(free - occupied)
        self.free.difference_update(occupied)
        self._signed_field = None

    def _cells(self, positions):
        return np.unique(np.floor(positions / self.voxel_m).astype(np.int32), axis=0)

    def anchor(self, point: Vec3):
        # The executed collision check validates these short links to the grid center.
        cell = self.cell(point)
        if cell in self.occupied:
            self.occupied.remove(cell)
            self._signed_field = None
        if cell not in self.free:
            self._signed_field = None
        self.free.add(cell)
        return cell

    def mark_blocked(self, cell):
        self.free.discard(cell)
        self.occupied.add(cell)
        self._signed_field = None

    def clearance(self, cell):
        """Signed distance in meters to the closest mapped occupied/free voxel."""
        if not self.occupied:
            return float("inf")
        if cell not in self.free and cell not in self.occupied:
            return float("-inf")
        if self._signed_field is None:
            known = np.asarray(list(self.free | self.occupied))
            lower = known.min(axis=0) - 1
            upper = known.max(axis=0) + 2
            obstacles = np.zeros(tuple(upper - lower), dtype=bool)
            obstacles[tuple((np.asarray(list(self.occupied)) - lower).T)] = True
            outside = distance_transform_edt(~obstacles)
            inside = distance_transform_edt(obstacles)
            self._signed_field = lower, (outside - inside) * self.voxel_m
        lower, field = self._signed_field
        return float(field[tuple(np.asarray(cell) - lower)])
