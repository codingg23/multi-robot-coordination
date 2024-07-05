"""
sim_env.py

Simulation environment for training the multi-robot coordination policy.

Uses Gymnasium (the maintained fork of OpenAI Gym).
Runs faster than real time so we can train offline.

The observation space:
  - Flattened local occupancy grid (40x40 cells around the robot)
  - Relative positions of other robots (dx, dy for each)
  - Known object locations (dx, dy, confidence for up to 3 objects)
  - Own velocity and heading

The action space:
  - Discrete: 8 directions + stop = 9 actions
  - Speed is fixed per action (easier to train than continuous speed)

Reward:
  - +1 per new cell explored (exploration bonus)
  - +10 for finding the target object
  - -0.5 per timestep (time pressure)
  - -5 for collision

This is a multi-agent env but I train all robots with a shared policy.
Means the policy needs to be useful regardless of which robot ID runs it,
which seemed to work okay in practice.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional
import math
import random

# Action encoding: 8 directions + stop
ACTIONS = [
    (1.0, 0.0),    # 0: forward
    (1.0, 0.785),  # 1: forward-right
    (0.0, 1.571),  # 2: right
    (-1.0, 0.785), # 3: back-right
    (-1.0, 0.0),   # 4: back
    (-1.0,-0.785), # 5: back-left
    (0.0,-1.571),  # 6: left
    (1.0,-0.785),  # 7: forward-left
    (0.0, 0.0),    # 8: stop
]
SPEED_M_PER_S = 0.3
TIMESTEP_S = 0.25
LOCAL_GRID_SIZE = 40


class SimRoom:
    """Simple rectangular room with randomly placed obstacles."""

    def __init__(self, width_m: float = 6.0, height_m: float = 6.0, n_obstacles: int = 5, seed: Optional[int] = None):
        self.width = width_m
        self.height = height_m
        self.rng = np.random.default_rng(seed)

        # obstacles are axis-aligned rectangles
        self.obstacles = []
        for _ in range(n_obstacles):
            ox = self.rng.uniform(0.5, width_m - 1.0)
            oy = self.rng.uniform(0.5, height_m - 1.0)
            ow = self.rng.uniform(0.2, 0.8)
            oh = self.rng.uniform(0.2, 0.8)
            self.obstacles.append((ox, oy, ow, oh))

    def is_free(self, x: float, y: float, radius: float = 0.15) -> bool:
        """Check if a position is free (not colliding with walls or obstacles)."""
        if x < radius or x > self.width - radius or y < radius or y > self.height - radius:
            return False
        for ox, oy, ow, oh in self.obstacles:
            if abs(x - ox) < (ow / 2 + radius) and abs(y - oy) < (oh / 2 + radius):
                return False
        return True

    def random_free_pos(self) -> tuple[float, float]:
        for _ in range(100):
            x = self.rng.uniform(0.2, self.width - 0.2)
            y = self.rng.uniform(0.2, self.height - 0.2)
            if self.is_free(x, y):
                return (float(x), float(y))
        return (self.width / 2, self.height / 2)


class MultiRobotEnv(gym.Env):
    """
    Multi-robot exploration environment.

    Each robot gets its own observation and produces its own action.
    The env step takes all robot actions simultaneously.

    I'm training with a shared policy so all robots use the same weights.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        n_robots: int = 3,
        room_width_m: float = 6.0,
        room_height_m: float = 6.0,
        n_obstacles: int = 5,
        max_steps: int = 1000,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_robots = n_robots
        self.room_width = room_width_m
        self.room_height = room_height_m
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps

        # observation for one robot:
        # local grid (40*40) + other robots (n_robots-1)*2 + object locations 3*3 + own state 3
        grid_obs = LOCAL_GRID_SIZE * LOCAL_GRID_SIZE
        other_robots_obs = (n_robots - 1) * 2
        objects_obs = 3 * 3
        own_state_obs = 3  # vx, vy, heading
        obs_size = grid_obs + other_robots_obs + objects_obs + own_state_obs

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_robots, obs_size), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([9] * n_robots)

        self._rng = np.random.default_rng(seed)
        self.room: Optional[SimRoom] = None
        self.robot_positions: list[list[float]] = []
        self.robot_headings: list[float] = []
        self.explored_cells: set = set()
        self.target_pos: Optional[tuple[float, float]] = None
        self.target_found_by: set = set()
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.room = SimRoom(
            self.room_width, self.room_height, self.n_obstacles,
            seed=int(self._rng.integers(0, 100000))
        )
        self.robot_positions = [list(self.room.random_free_pos()) for _ in range(self.n_robots)]
        self.robot_headings = [self._rng.uniform(0, 2 * math.pi) for _ in range(self.n_robots)]
        self.explored_cells = set()
        self.target_pos = self.room.random_free_pos()
        self.target_found_by = set()
        self.step_count = 0

        for i, pos in enumerate(self.robot_positions):
            self._update_explored(pos[0], pos[1], i)

        return self._get_obs(), {}

    def step(self, actions):
        rewards = np.zeros(self.n_robots)
        self.step_count += 1

        for i, action in enumerate(actions):
            speed, d_heading = ACTIONS[action]
            new_heading = (self.robot_headings[i] + d_heading) % (2 * math.pi)
            dx = speed * SPEED_M_PER_S * TIMESTEP_S * math.cos(new_heading)
            dy = speed * SPEED_M_PER_S * TIMESTEP_S * math.sin(new_heading)
            nx = self.robot_positions[i][0] + dx
            ny = self.robot_positions[i][1] + dy

            if self.room.is_free(nx, ny):
                self.robot_positions[i] = [nx, ny]
                self.robot_headings[i] = new_heading
                new_cells = self._update_explored(nx, ny, i)
                rewards[i] += new_cells * 1.0  # exploration reward
            else:
                rewards[i] -= 5.0  # collision penalty

        # check target discovery
        if self.target_pos:
            for i in range(self.n_robots):
                if i not in self.target_found_by:
                    rx, ry = self.robot_positions[i]
                    tx, ty = self.target_pos
                    dist = math.sqrt((rx - tx)**2 + (ry - ty)**2)
                    if dist < 0.5:  # within 50cm = "found"
                        self.target_found_by.add(i)
                        rewards[i] += 10.0

        # time penalty
        rewards -= 0.5

        terminated = len(self.target_found_by) == self.n_robots
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), rewards.sum(), terminated, truncated, {}

    def _update_explored(self, x: float, y: float, robot_id: int) -> int:
        """Mark cells around robot as explored. Returns number of new cells."""
        cell_x = int(x / 0.1)
        cell_y = int(y / 0.1)
        radius = 3  # cells visible around robot
        new_count = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                cell = (cell_x + dx, cell_y + dy)
                if cell not in self.explored_cells:
                    self.explored_cells.add(cell)
                    new_count += 1
        return new_count

    def _get_obs(self) -> np.ndarray:
        """Build observation array for all robots."""
        obs = np.zeros((self.n_robots, self.observation_space.shape[1]), dtype=np.float32)

        for i in range(self.n_robots):
            rx, ry = self.robot_positions[i]
            heading = self.robot_headings[i]
            offset = 0

            # local occupancy grid
            cell_size = self.room_width / LOCAL_GRID_SIZE
            for gr in range(LOCAL_GRID_SIZE):
                for gc in range(LOCAL_GRID_SIZE):
                    wx = gc * cell_size
                    wy = gr * cell_size
                    if self.room.is_free(wx, wy):
                        obs[i, offset] = 1.0
                    else:
                        obs[i, offset] = -1.0
                    offset += 1

            # other robot positions (relative)
            for j in range(self.n_robots):
                if j == i:
                    continue
                dx = (self.robot_positions[j][0] - rx) / self.room_width
                dy = (self.robot_positions[j][1] - ry) / self.room_height
                obs[i, offset] = np.clip(dx, -1, 1)
                obs[i, offset + 1] = np.clip(dy, -1, 1)
                offset += 2

            # target location (if found by this robot)
            if self.target_pos and i in self.target_found_by:
                tx, ty = self.target_pos
                obs[i, offset] = np.clip((tx - rx) / self.room_width, -1, 1)
                obs[i, offset + 1] = np.clip((ty - ry) / self.room_height, -1, 1)
                obs[i, offset + 2] = 1.0
            offset += 3 * 3  # 3 object slots

            # own state
            obs[i, offset] = math.cos(heading)
            obs[i, offset + 1] = math.sin(heading)
            obs[i, offset + 2] = self.step_count / self.max_steps

        return obs

    def render(self):
        pass  # could add pygame visualisation here - didn't get to it

FRONTIER_BONUS = 1.0  # reward per new cell explored
