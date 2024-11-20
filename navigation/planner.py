"""
planner.py

Path planning on top of the occupancy grid.

A* to get from current position to a target cell.
The planner runs on each robot independently using the merged shared grid.

This is not anything fancy. Just A* with a simple heuristic and
a small penalty for cells near obstacles (so paths prefer the middle of corridors).
"""

import heapq
import math
import numpy as np
from typing import Optional
from .occupancy_grid import OccupancyGrid, FREE, OCCUPIED, UNKNOWN


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    # Euclidean distance
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _neighbours(row: int, col: int, grid: OccupancyGrid) -> list[tuple[int, int]]:
    """8-connected neighbours that are not occupied."""
    candidates = [
        (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1),
        (row - 1, col - 1), (row - 1, col + 1), (row + 1, col - 1), (row + 1, col + 1),
    ]
    out = []
    for r, c in candidates:
        if 0 <= r < grid.rows and 0 <= c < grid.cols:
            if grid.cells[r, c] != OCCUPIED:
                out.append((r, c))
    return out


def _obstacle_penalty(row: int, col: int, grid: OccupancyGrid, radius: int = 2) -> float:
    """Small penalty for cells near obstacles. Encourages paths away from walls."""
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = row + dr, col + dc
            if 0 <= r < grid.rows and 0 <= c < grid.cols:
                if grid.cells[r, c] == OCCUPIED:
                    dist = math.sqrt(dr**2 + dc**2)
                    if dist < radius:
                        return 0.5 * (radius - dist) / radius
    return 0.0


def astar(
    grid: OccupancyGrid,
    start_world: tuple[float, float],
    goal_world: tuple[float, float],
) -> Optional[list[tuple[float, float]]]:
    """
    Find a path from start to goal. Returns world coordinates or None if no path.

    The path is a list of waypoints. On the robot, we drive toward each waypoint
    in sequence, with obstacle avoidance handling minor deviations.
    """
    start_cell = grid.world_to_grid(*start_world)
    goal_cell = grid.world_to_grid(*goal_world)

    if start_cell is None or goal_cell is None:
        return None

    # snap goal to nearest free cell if it's occupied
    if grid.cells[goal_cell] == OCCUPIED:
        goal_cell = _nearest_free(grid, goal_cell)
    if goal_cell is None:
        return None

    open_heap = []
    heapq.heappush(open_heap, (0.0, start_cell))

    came_from = {}
    g_score = {start_cell: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal_cell:
            return _reconstruct_path(came_from, current, grid)

        for nbr in _neighbours(*current, grid):
            # diagonal movement costs more
            dr, dc = abs(nbr[0] - current[0]), abs(nbr[1] - current[1])
            move_cost = 1.414 if (dr == 1 and dc == 1) else 1.0
            move_cost += _obstacle_penalty(*nbr, grid)

            # unknown cells cost a bit more to discourage driving through them
            if grid.cells[nbr] == UNKNOWN:
                move_cost += 0.3

            tentative_g = g_score[current] + move_cost
            if tentative_g < g_score.get(nbr, float("inf")):
                came_from[nbr] = current
                g_score[nbr] = tentative_g
                f = tentative_g + _heuristic(nbr, goal_cell)
                heapq.heappush(open_heap, (f, nbr))

    return None  # no path found


def _reconstruct_path(came_from: dict, current: tuple, grid: OccupancyGrid) -> list:
    path_cells = [current]
    while current in came_from:
        current = came_from[current]
        path_cells.append(current)
    path_cells.reverse()
    return [grid.grid_to_world(*c) for c in path_cells]


def _nearest_free(grid: OccupancyGrid, cell: tuple[int, int], max_radius: int = 5) -> Optional[tuple]:
    row, col = cell
    for r in range(1, max_radius + 1):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid.rows and 0 <= nc < grid.cols:
                    if grid.cells[nr, nc] == FREE:
                        return (nr, nc)
    return None


def select_frontier_target(
    grid: OccupancyGrid,
    robot_pos: tuple[float, float],
    other_robot_targets: list[tuple[float, float]],
    min_separation_m: float = 1.0,
) -> Optional[tuple[float, float]]:
    """
    Pick the best frontier to explore next.

    Prefers frontiers that are:
    - Reachable
    - Not already targeted by another robot
    - A reasonable distance away (not too close)
    """
    frontiers = grid.get_frontier_cells()
    if not frontiers:
        return None

    # filter out cells too close to other robots' targets
    available = []
    for fx, fy in frontiers:
        too_close = any(
            math.sqrt((fx - tx)**2 + (fy - ty)**2) < min_separation_m
            for tx, ty in other_robot_targets
        )
        if not too_close:
            available.append((fx, fy))

    if not available:
        available = frontiers  # if all are close, just pick one anyway

    # score by distance from robot (prefer moderate distance, not too near or far)
    rx, ry = robot_pos
    def score(f):
        d = math.sqrt((f[0] - rx)**2 + (f[1] - ry)**2)
        # sweet spot around 1-2m away
        return abs(d - 1.5)

    available.sort(key=score)
    return available[0] if available else None

MAX_PLAN_ATTEMPTS = 3  # retry limit if first path fails

MIN_ROBOT_SEPARATION_M = 1.0  # avoid sending two robots to same frontier
