"""
ultrasonic.py

HC-SR04 ultrasonic distance sensor driver.

Measures distance by timing how long a sound pulse takes to return.
Distance = (pulse_duration * speed_of_sound) / 2

I'm running 3 sensors per robot (front, left, right).
They can't all trigger at the same time or they'll interfere with each other,
so they're triggered in sequence with small gaps.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SPEED_OF_SOUND_M_PER_S = 343.0
MAX_RANGE_M = 4.0
MIN_RANGE_M = 0.02


class UltrasonicSensor:
    """Single HC-SR04 sensor."""

    def __init__(self, trig_pin: int, echo_pin: int, timeout_s: float = 0.05):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.timeout_s = timeout_s
        self._gpio = None
        self._setup()

    def _setup(self):
        try:
            import RPi.GPIO as GPIO
            self._gpio = GPIO
            GPIO.setup(self.trig_pin, GPIO.OUT)
            GPIO.setup(self.echo_pin, GPIO.IN)
            GPIO.output(self.trig_pin, GPIO.LOW)
            time.sleep(0.002)  # settle time
        except ImportError:
            pass  # not on a Pi

    def read_distance_m(self) -> Optional[float]:
        """Read distance in metres. Returns None on timeout or error."""
        if self._gpio is None:
            return None

        GPIO = self._gpio
        # 10 microsecond trigger pulse
        GPIO.output(self.trig_pin, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.trig_pin, GPIO.LOW)

        # wait for echo start
        pulse_start = time.time()
        timeout = pulse_start + self.timeout_s
        while GPIO.input(self.echo_pin) == GPIO.LOW:
            pulse_start = time.time()
            if pulse_start > timeout:
                return None

        # wait for echo end
        pulse_end = time.time()
        timeout = pulse_end + self.timeout_s
        while GPIO.input(self.echo_pin) == GPIO.HIGH:
            pulse_end = time.time()
            if pulse_end > timeout:
                return None

        duration = pulse_end - pulse_start
        distance = (duration * SPEED_OF_SOUND_M_PER_S) / 2.0

        if distance < MIN_RANGE_M or distance > MAX_RANGE_M:
            return None

        return distance


class SensorArray:
    """
    Three ultrasonic sensors: front, left-front, right-front.
    Triggered in sequence to avoid interference.
    """

    def __init__(self, front_pins: tuple[int, int], left_pins: tuple[int, int], right_pins: tuple[int, int]):
        self.front = UltrasonicSensor(*front_pins)
        self.left = UltrasonicSensor(*left_pins)
        self.right = UltrasonicSensor(*right_pins)
        self._trigger_gap_s = 0.015  # 15ms between triggers

    def read_all(self) -> dict[str, Optional[float]]:
        """Read all three sensors. Returns dict with distances in metres."""
        results = {}
        for name, sensor in [("front", self.front), ("left", self.left), ("right", self.right)]:
            results[name] = sensor.read_distance_m()
            time.sleep(self._trigger_gap_s)
        return results

    def obstacle_ahead(self, threshold_m: float = 0.4) -> bool:
        """Quick check if there's anything close in front."""
        d = self.front.read_distance_m()
        return d is not None and d < threshold_m

    def clear_to_turn_left(self, threshold_m: float = 0.25) -> bool:
        d = self.left.read_distance_m()
        return d is None or d > threshold_m

    def clear_to_turn_right(self, threshold_m: float = 0.25) -> bool:
        d = self.right.read_distance_m()
        return d is None or d > threshold_m
