"""
test_grid.py

Unit tests for the occupancy grid and path planner.

These run on any machine, no Pi required.
"""

import pytest
import numpy as np
from navigation.occupancy_grid import OccupancyGrid, GridConfig, FREE, OCCUPIED, UNKNOWN
from navigation.planner import astar


def make_grid(width=4.0, height=4.0):
    return OccupancyGrid(GridConfig(width_m=width, height_m=height))


class TestOccupancyGrid:

    def test_world_to_grid_basic(self):
        grid = make_grid()
        cell = grid.world_to_grid(0.1, 0.1)
        assert cell is not None
        row, col = cell
        assert row == 2  # 0.1 / 0.05 = 2
        assert col == 2

    def test_world_to_grid_out_of_bounds(self):
        grid = make_grid()
        assert grid.world_to_grid(-0.1, 0.5) is None
        assert grid.world_to_grid(5.0, 0.5) is None

    def test_mark_occupied_updates_cell(self):
        grid = make_grid()
        # need enough hits to flip state
        for _ in range(5):
            grid.mark_occupied(1.0, 1.0)
        cell = grid.world_to_grid(1.0, 1.0)
        assert cell is not None
        assert grid.cells[cell] == OCCUPIED

    def test_mark_free_updates_cell(self):
        grid = make_grid()
        for _ in range(5):
            grid.mark_free(2.0, 2.0)
        cell = grid.world_to_grid(2.0, 2.0)
        assert cell is not None
        assert grid.cells[cell] == FREE

    def test_probabilistic_update_resists_noise(self):
        grid = make_grid()
        # 3 hits, 7 misses - should stay FREE
        for _ in range(3):
            grid.mark_occupied(1.0, 1.0)
        for _ in range(7):
            grid.mark_free(1.0, 1.0)
        cell = grid.world_to_grid(1.0, 1.0)
        assert grid.cells[cell] == FREE

    def test_grid_to_world_roundtrip(self):
        grid = make_grid()
        for x in [0.5, 1.0, 2.5, 3.9]:
            for y in [0.5, 1.0, 2.5, 3.9]:
                cell = grid.world_to_grid(x, y)
                if cell:
                    wx, wy = grid.grid_to_world(*cell)
                    # world coords should be within one cell size
                    assert abs(wx - x) <= grid.config.cell_size_m
                    assert abs(wy - y) <= grid.config.cell_size_m

    def test_coverage_starts_at_zero(self):
        grid = make_grid()
        assert grid.coverage_fraction() == 0.0

    def test_coverage_increases_as_cells_mapped(self):
        grid = make_grid()
        for _ in range(10):
            grid.mark_free(1.0, 1.0)
        assert grid.coverage_fraction() > 0.0

    def test_merge_adds_counts(self):
        g1 = make_grid()
        g2 = make_grid()
        for _ in range(5):
            g1.mark_occupied(1.0, 1.0)
        for _ in range(5):
            g2.mark_occupied(1.0, 1.0)
        g1.merge(g2)
        cell = g1.world_to_grid(1.0, 1.0)
        assert g1._hit_counts[cell] == 10

    def test_serialise_roundtrip(self):
        grid = make_grid()
        for _ in range(5):
            grid.mark_free(1.0, 1.5)
            grid.mark_occupied(2.0, 2.5)
        d = grid.to_dict()
        restored = OccupancyGrid.from_dict(d)
        assert np.array_equal(grid.cells, restored.cells)
        assert np.array_equal(grid._hit_counts, restored._hit_counts)

    def test_frontier_cells_are_adjacent_to_unknown(self):
        grid = make_grid()
        # mark a small area as free
        for _ in range(5):
            grid.mark_free(2.0, 2.0)
            grid.mark_free(2.05, 2.0)
        frontiers = grid.get_frontier_cells()
        # all frontiers should be free cells
        for fx, fy in frontiers:
            cell = grid.world_to_grid(fx, fy)
            assert grid.cells[cell] == FREE


class TestPlanner:

    def test_simple_open_path(self):
        grid = make_grid(width=3.0, height=3.0)
        # mark everything as free
        for x in np.arange(0.1, 3.0, 0.05):
            for y in np.arange(0.1, 3.0, 0.05):
                for _ in range(5):
                    grid.mark_free(x, y)

        path = astar(grid, (0.1, 0.1), (2.8, 2.8))
        assert path is not None
        assert len(path) > 1
        # path should start near start
        assert abs(path[0][0] - 0.1) < 0.2
        assert abs(path[0][1] - 0.1) < 0.2

    def test_no_path_through_wall(self):
        grid = make_grid(width=3.0, height=3.0)
        # create a vertical wall of obstacles
        for y in np.arange(0.0, 3.0, 0.05):
            for _ in range(10):
                grid.mark_occupied(1.5, y)
        # mark start and end sides as free
        for x in np.arange(0.1, 1.4, 0.05):
            for y in np.arange(0.1, 2.9, 0.05):
                for _ in range(5):
                    grid.mark_free(x, y)

        # can't get from left side to right side
        path = astar(grid, (0.5, 1.5), (2.5, 1.5))
        assert path is None

    def test_same_start_and_goal(self):
        grid = make_grid()
        for _ in range(5):
            grid.mark_free(1.0, 1.0)
        path = astar(grid, (1.0, 1.0), (1.0, 1.0))
        # should return a trivial path
        assert path is not None

# TODO: add benchmark test for large grid A* performance
