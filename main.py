"""
main.py

Entry point for each robot.

Starts all threads:
  - Sensor thread: reads ultrasonic sensors at 10Hz
  - Vision thread: runs camera + object detection at 5Hz
  - State sync thread: receives updates from server, pushes own updates
  - Planning thread: computes next target and updates motor commands
  - Obstacle avoidance: reactive layer, runs at 20Hz, overrides planner if obstacle close

Usage:
    ROBOT_ID=0 python main.py --server 192.168.1.100:5555 --mode explore
"""

import os
import sys
import time
import math
import logging
import threading
import argparse
from typing import Optional

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        level=level,
    )


class RobotRunner:
    """
    Main robot controller. Coordinates all subsystems.
    """

    def __init__(
        self,
        robot_id: int,
        server_host: str,
        server_push_port: int,
        server_sub_port: int,
        mode: str = "explore",
        target_colour: str = "red",
    ):
        self.robot_id = robot_id
        self.mode = mode
        self.target_colour = target_colour

        # position estimate (updated via odometry - very rough without proper SLAM)
        self.x_m = 0.5 + robot_id * 0.5  # stagger start positions
        self.y_m = 0.5
        self.heading_rad = 0.0

        self._running = False
        self._lock = threading.Lock()

        # sensor readings
        self._sensor_readings = {"front": None, "left": None, "right": None}

        # imported lazily so this file loads even without Pi hardware
        self._motor = None
        self._sensors = None
        self._vision = None
        self._state_client = None

        self._init_hardware()
        self._init_comms(server_host, server_push_port, server_sub_port)

    def _init_hardware(self):
        try:
            from utils.motor_control import MotorController
            from utils.ultrasonic import SensorArray
            from vision.detector import ObjectDetector, ColourBlobDetector

            self._motor = MotorController()
            self._sensors = SensorArray(
                front_pins=(24, 25),
                left_pins=(5, 6),
                right_pins=(13, 19),
            )
            self._yolo = ObjectDetector(target_label="sports ball")
            self._colour_detector = ColourBlobDetector()
            logger.info(f"Robot {self.robot_id}: hardware initialised")
        except Exception as e:
            logger.warning(f"Hardware init failed (running in sim mode?): {e}")

    def _init_comms(self, host: str, push_port: int, sub_port: int):
        try:
            import zmq
            ctx = zmq.Context()
            self._push_socket = ctx.socket(zmq.PUSH)
            self._push_socket.connect(f"tcp://{host}:{push_port}")
            self._sub_socket = ctx.socket(zmq.SUB)
            self._sub_socket.connect(f"tcp://{host}:{sub_port}")
            self._sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
            logger.info(f"Robot {self.robot_id}: connected to server {host}:{push_port}")
        except Exception as e:
            logger.warning(f"Comms init failed: {e}")
            self._push_socket = None
            self._sub_socket = None

    def _sensor_loop(self):
        """Read ultrasonic sensors at 10Hz."""
        while self._running:
            if self._sensors:
                readings = self._sensors.read_all()
                with self._lock:
                    self._sensor_readings = readings
            time.sleep(0.1)

    def _obstacle_avoidance_loop(self):
        """Reactive obstacle avoidance. Overrides planner commands if obstacle close."""
        while self._running:
            with self._lock:
                front = self._sensor_readings.get("front")
                left = self._sensor_readings.get("left")
                right = self._sensor_readings.get("right")

            if front is not None and front < 0.3:
                # obstacle close ahead - turn away
                if left is None or (right is not None and right > left):
                    # more space to the right
                    if self._motor:
                        self._motor.drive(0.0, 0.5)  # turn right
                else:
                    if self._motor:
                        self._motor.drive(0.0, -0.5)  # turn left
                time.sleep(0.3)
            time.sleep(0.05)

    def _push_state(self):
        """Push position and grid updates to server."""
        if not self._push_socket:
            return
        msg = {
            "type": "position_update",
            "robot_id": self.robot_id,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "heading_rad": self.heading_rad,
            "mode": self.mode,
            "battery_pct": 85.0,  # TODO: read actual battery level
        }
        try:
            self._push_socket.send_json(msg)
        except Exception as e:
            logger.debug(f"Push failed: {e}")

    def run(self):
        """Start all threads and run the main loop."""
        self._running = True

        threads = [
            threading.Thread(target=self._sensor_loop, daemon=True, name="sensors"),
            threading.Thread(target=self._obstacle_avoidance_loop, daemon=True, name="avoidance"),
        ]
        for t in threads:
            t.start()

        logger.info(f"Robot {self.robot_id} running in mode '{self.mode}'")

        try:
            while self._running:
                self._push_state()
                time.sleep(2.0)
        except KeyboardInterrupt:
            logger.info("Shutting down")
        finally:
            self._running = False
            if self._motor:
                self._motor.stop()
                self._motor.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Multi-robot coordination node")
    parser.add_argument("--server", default="localhost:5555", help="State server host:port")
    parser.add_argument("--mode", default="explore", choices=["explore", "retrieve", "idle"])
    parser.add_argument("--target-colour", default="red")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging(args.debug)

    robot_id = int(os.environ.get("ROBOT_ID", "0"))
    host, push_port_str = args.server.rsplit(":", 1)
    push_port = int(push_port_str)
    sub_port = push_port + 1

    runner = RobotRunner(
        robot_id=robot_id,
        server_host=host,
        server_push_port=push_port,
        server_sub_port=sub_port,
        mode=args.mode,
        target_colour=args.target_colour,
    )
    runner.run()


if __name__ == "__main__":
    main()

STATE_PUSH_INTERVAL_S = 2.0  # how often to push position to server
