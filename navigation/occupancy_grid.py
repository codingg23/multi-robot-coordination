"""
occupancy_grid.py

Simple 2D occupancy grid for mapping the environment.

Each cell is either FREE, OCCUPIED, or UNKNOWN.
Robots update their own grid and push diffs to the state server,
where all updates get merged into a shared map.

Grid resolution is 5cm per cell. A 6x6m room = 120x120 cells = fine.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import math

# Cell states
UNKNOWN = 0
FREE = 1
OCCUPIED = 2

CELL_SIZE_M = 0.05  # 5cm per cell


@dataclass
class GridConfig:
    width_m: float = 6.0
    height_m: float = 6.0
    cell_size_m: float = CELL_SIZE_M
    origin_x_m: float = 0.0   # world coords of grid origin (bottom-left)
    origin_y_m: float = 0.0


class OccupancyGrid:
    """
    2D occupancy grid. Stores cell states and hit/miss counts for
    probabilistic updates (prevents single noisy readings from dominating).
    """

    def __init__(self, config: GridConfig):
        self.config = config
        self.cols = int(config.width_m / config.cell_size_m)
        self.rows = int(config.height_m / config.cell_size_m)

        self.cells = np.full((self.rows, self.cols), UNKNOWN, dtype=np.int8)
        self._hit_counts = np.zeros((self.rows, self.cols), dtype=np.int16)
        self._miss_counts = np.zeros((self.rows, self.cols), dtype=np.int16)

    def world_to_grid(self, x_m: float, y_m: float) -> Optional[tuple[int, int]]:
        """Convert world coordinates to grid cell indices."""
        col = int((x_m - self.config.origin_x_m) / self.config.cell_size_m)
        row = int((y_m - self.config.origin_y_m) / self.config.cell_size_m)
        if 0 <= col < self.cols and 0 <= row < self.rows:
            return (row, col)
        return None

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Convert grid cell indices to world coordinates (cell centre)."""
        x = self.config.origin_x_m + (col + 0.5) * self.config.cell_size_m
        y = self.config.origin_y_m + (row + 0.5) * self.config.cell_size_m
        return (x, y)

    def mark_occupied(self, x_m: float, y_m: float):
        """Mark a cell as occupied (obstacle hit)."""
        cell = self.world_to_grid(x_m, y_m)
        if cell:
            row, col = cell
            self._hit_counts[row, col] += 1
            self._recompute_cell(row, col)

    def mark_free(self, x_m: float, y_m: float):
        """Mark a cell as free (ray passed through with no obstacle)."""
        cell = self.world_to_grid(x_m, y_m)
        if cell:
            row, col = cell
            self._miss_counts[row, col] += 1
            self._recompute_cell(row, col)

    def _recompute_cell(self, row: int, col: int):
        """Update cell state based on hit/miss ratio."""
        hits = self._hit_counts[row, col]
        misses = self._miss_counts[row, col]
        total = hits + misses
        if total < 3:
            return  # not enough observations yet
        ratio = hits / total
        if ratio > 0.6:
            self.cells[row, col] = OCCUPIED
        elif ratio < 0.3:
            self.cells[row, col] = FREE

    def raycast(self, robot_x: float, robot_y: float, angle_rad: float, max_range_m: float, step_m: float = 0.04):
        """
        Trace a ray from robot position in the given direction.
        Marks cells along the ray as free, and the endpoint as occupied.

        Used to update the grid from ultrasonic sensor readings.
        """
        dx = math.cos(angle_rad) * step_m
        dy = math.sin(angle_rad) * step_m
        steps = int(max_range_m / step_m)

        x, y = robot_x, robot_y
        for i in range(steps):
            x += dx
            y += dy
            if i < steps - 1:
                self.mark_free(x, y)
            else:
                self.mark_occupied(x, y)

    def merge(self, other: "OccupancyGrid"):
        """
        Merge another grid into this one.
        Simply adds hit/miss counts and recomputes.
        """
        if other.cells.shape != self.cells.shape:
            raise ValueError("Grid dimensions must match for merge")

        self._hit_counts += other._hit_counts
        self._miss_counts += other._miss_counts

        # recompute all cells that changed
        changed = np.where(
            (other._hit_counts > 0) | (other._miss_counts > 0)
        )
        for row, col in zip(changed[0], changed[1]):
            self._recompute_cell(row, col)

    def get_unknown_cells(self) -> list[tuple[int, int]]:
        """Return all cells that are still UNKNOWN."""
        return list(zip(*np.where(self.cells == UNKNOWN)))

    def get_frontier_cells(self) -> list[tuple[float, float]]:
        """
        Return world coordinates of frontier cells - free cells adjacent to unknown cells.
        These are the best candidates for exploration.
        """
        frontiers = []
        free_mask = self.cells == FREE
        unknown_mask = self.cells == UNKNOWN

        for row in range(1, self.rows - 1):
            for col in range(1, self.cols - 1):
                if not free_mask[row, col]:
                    continue
                neighbours = [
                    unknown_mask[row-1, col], unknown_mask[row+1, col],
                    unknown_mask[row, col-1], unknown_mask[row, col+1],
                ]
                if any(neighbours):
                    frontiers.append(self.grid_to_world(row, col))

        return frontiers

    def coverage_fraction(self) -> float:
        """Fraction of cells that are not UNKNOWN."""
        total = self.rows * self.cols
        mapped = np.sum(self.cells != UNKNOWN)
        return float(mapped) / total

    def to_dict(self) -> dict:
        """Serialise for sending over network."""
        return {
            "cells": self.cells.tolist(),
            "hits": self._hit_counts.tolist(),
            "misses": self._miss_counts.tolist(),
            "config": {
                "width_m": self.config.width_m,
                "height_m": self.config.height_m,
                "cell_size_m": self.config.cell_size_m,
                "origin_x_m": self.config.origin_x_m,
                "origin_y_m": self.config.origin_y_m,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OccupancyGrid":
        cfg = GridConfig(**d["config"])
        grid = cls(cfg)
        grid.cells = np.array(d["cells"], dtype=np.int8)
        grid._hit_counts = np.array(d["hits"], dtype=np.int16)
        grid._miss_counts = np.array(d["misses"], dtype=np.int16)
        return grid


def grid_stats(grid: OccupancyGrid) -> dict:
    """Return basic statistics about the grid state."""
    total = grid.rows * grid.cols
    return {
        "total_cells": total,
        "free": int(np.sum(grid.cells == FREE)),
        "occupied": int(np.sum(grid.cells == OCCUPIED)),
        "unknown": int(np.sum(grid.cells == UNKNOWN)),
        "coverage": grid.coverage_fraction(),
    }

RAYCAST_STEP_M = 0.04  # default step size for raycast
