"""
state_server.py

Central state server that aggregates grid updates and robot positions.

Robots push their local grid diffs and position updates here.
The server merges everything into a shared world state and serves it back.

Uses ZeroMQ PUB/SUB so robots can subscribe to state updates without
needing to poll. The robots push updates every 2 seconds, or immediately
when they detect something interesting (obstacle, target object).

This is the single point of failure in the current design - if this
goes down all robots stop dead. A peer-to-peer approach would be better
but I didn't have time to implement it.
"""

import zmq
import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from ..navigation.occupancy_grid import OccupancyGrid, GridConfig

logger = logging.getLogger(__name__)


@dataclass
class RobotState:
    robot_id: int
    x_m: float
    y_m: float
    heading_rad: float
    current_target: Optional[tuple[float, float]]
    mode: str  # "explore", "retrieve", "idle"
    battery_pct: float
    last_updated: float = field(default_factory=time.time)


@dataclass
class ObjectSighting:
    object_label: str
    colour: str
    world_x: float
    world_y: float
    confidence: float
    seen_by: int     # robot_id
    timestamp: float


class StateServer:
    """
    Runs on a central machine (or a dedicated Pi).
    Receives updates from all robots and maintains the global world state.
    """

    def __init__(self, push_port: int = 5555, pub_port: int = 5556, grid_config: Optional[GridConfig] = None):
        self.push_port = push_port
        self.pub_port = pub_port

        cfg = grid_config or GridConfig()
        self.shared_grid = OccupancyGrid(cfg)
        self.robot_states: dict[int, RobotState] = {}
        self.object_sightings: list[ObjectSighting] = []

        self._lock = threading.Lock()
        self._running = False

        self.context = zmq.Context()
        # robots push updates to this socket
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://*:{push_port}")
        # server publishes merged state to all robots
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://*:{pub_port}")

        logger.info(f"StateServer bound on ports {push_port} (PULL) and {pub_port} (PUB)")

    def run(self):
        """Main loop. Call this in a thread."""
        self._running = True
        broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        broadcast_thread.start()

        while self._running:
            try:
                msg = self.pull_socket.recv_json(flags=zmq.NOBLOCK)
                self._handle_message(msg)
            except zmq.Again:
                time.sleep(0.01)
            except Exception as e:
                logger.exception(f"Error handling message: {e}")

    def stop(self):
        self._running = False

    def _handle_message(self, msg: dict):
        msg_type = msg.get("type")
        robot_id = msg.get("robot_id")

        with self._lock:
            if msg_type == "position_update":
                self.robot_states[robot_id] = RobotState(
                    robot_id=robot_id,
                    x_m=msg["x_m"],
                    y_m=msg["y_m"],
                    heading_rad=msg["heading_rad"],
                    current_target=msg.get("current_target"),
                    mode=msg.get("mode", "explore"),
                    battery_pct=msg.get("battery_pct", 100.0),
                )
                logger.debug(f"Robot {robot_id} position: ({msg['x_m']:.2f}, {msg['y_m']:.2f})")

            elif msg_type == "grid_update":
                try:
                    update_grid = OccupancyGrid.from_dict(msg["grid"])
                    self.shared_grid.merge(update_grid)
                    coverage = self.shared_grid.coverage_fraction()
                    logger.debug(f"Grid update from robot {robot_id}, coverage: {coverage:.1%}")
                except Exception as e:
                    logger.warning(f"Failed to merge grid from robot {robot_id}: {e}")

            elif msg_type == "object_sighting":
                sighting = ObjectSighting(
                    object_label=msg["label"],
                    colour=msg.get("colour", "unknown"),
                    world_x=msg["world_x"],
                    world_y=msg["world_y"],
                    confidence=msg["confidence"],
                    seen_by=robot_id,
                    timestamp=time.time(),
                )
                self.object_sightings.append(sighting)
                logger.info(f"Object sighting: {sighting.colour} {sighting.object_label} at "
                            f"({sighting.world_x:.2f}, {sighting.world_y:.2f}) by robot {robot_id}")

    def _broadcast_loop(self):
        """Broadcast merged state to all robots every 2 seconds."""
        while self._running:
            time.sleep(2.0)
            with self._lock:
                state = {
                    "type": "world_state",
                    "timestamp": time.time(),
                    "grid": self.shared_grid.to_dict(),
                    "robots": {
                        rid: {
                            "x_m": rs.x_m,
                            "y_m": rs.y_m,
                            "heading_rad": rs.heading_rad,
                            "current_target": rs.current_target,
                            "mode": rs.mode,
                        }
                        for rid, rs in self.robot_states.items()
                    },
                    "sightings": [
                        {
                            "label": s.object_label,
                            "colour": s.colour,
                            "x": s.world_x,
                            "y": s.world_y,
                            "confidence": s.confidence,
                            "t": s.timestamp,
                        }
                        for s in self.object_sightings[-20:]  # last 20 sightings
                    ],
                    "coverage": self.shared_grid.coverage_fraction(),
                }
            try:
                self.pub_socket.send_json(state)
            except Exception as e:
                logger.warning(f"Broadcast failed: {e}")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--push-port", type=int, default=5555)
    parser.add_argument("--pub-port", type=int, default=5556)
    parser.add_argument("--room-size", type=float, default=6.0)
    args = parser.parse_args()

    cfg = GridConfig(width_m=args.room_size, height_m=args.room_size)
    server = StateServer(args.push_port, args.pub_port, cfg)
    print(f"State server running on push:{args.push_port}, pub:{args.pub_port}")
    server.run()
