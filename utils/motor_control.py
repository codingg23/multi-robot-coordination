"""
motor_control.py

L298N motor driver interface for Raspberry Pi.

The L298N has two H-bridges. Each controls one motor.
Speed is set via PWM, direction via two GPIO pins per motor.

I'm using RPi.GPIO in BCM mode. Pin numbers below are BCM numbers.
"""

import time
import logging

logger = logging.getLogger(__name__)

# Default GPIO pins (BCM)
DEFAULT_LEFT_IN1 = 17
DEFAULT_LEFT_IN2 = 18
DEFAULT_LEFT_ENA = 12   # PWM capable
DEFAULT_RIGHT_IN3 = 22
DEFAULT_RIGHT_IN4 = 23
DEFAULT_RIGHT_ENB = 13  # PWM capable

PWM_FREQ = 100  # Hz


class MotorController:
    """
    Controls left and right motors via L298N.

    Speed is -1.0 to 1.0 (negative = reverse).
    """

    def __init__(
        self,
        left_in1=DEFAULT_LEFT_IN1, left_in2=DEFAULT_LEFT_IN2, left_ena=DEFAULT_LEFT_ENA,
        right_in3=DEFAULT_RIGHT_IN3, right_in4=DEFAULT_RIGHT_IN4, right_enb=DEFAULT_RIGHT_ENB,
    ):
        self._pins = {
            "left_in1": left_in1, "left_in2": left_in2, "left_ena": left_ena,
            "right_in3": right_in3, "right_in4": right_in4, "right_enb": right_enb,
        }
        self._pwm_left = None
        self._pwm_right = None
        self._initialized = False
        self._setup()

    def _setup(self):
        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for pin in self._pins.values():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)

            self._pwm_left = GPIO.PWM(self._pins["left_ena"], PWM_FREQ)
            self._pwm_right = GPIO.PWM(self._pins["right_enb"], PWM_FREQ)
            self._pwm_left.start(0)
            self._pwm_right.start(0)
            self._initialized = True
            logger.info("Motor controller initialised")
        except ImportError:
            logger.warning("RPi.GPIO not available (not on a Pi?), motor control disabled")
        except Exception as e:
            logger.error(f"Motor init failed: {e}")

    def set_speeds(self, left: float, right: float):
        """
        Set motor speeds. Values from -1.0 (full reverse) to 1.0 (full forward).
        """
        if not self._initialized:
            return
        self._set_motor(
            self._pins["left_in1"], self._pins["left_in2"], self._pwm_left, left
        )
        self._set_motor(
            self._pins["right_in3"], self._pins["right_in4"], self._pwm_right, right
        )

    def _set_motor(self, in1: int, in2: int, pwm, speed: float):
        GPIO = self._GPIO
        speed = max(-1.0, min(1.0, speed))
        duty = abs(speed) * 100

        if speed > 0:
            GPIO.output(in1, GPIO.HIGH)
            GPIO.output(in2, GPIO.LOW)
        elif speed < 0:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.HIGH)
        else:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.LOW)

        pwm.ChangeDutyCycle(duty)

    def drive(self, speed: float, turn: float):
        """
        Higher-level drive command.
        speed: forward/backward (-1 to 1)
        turn: left/right (-1 = hard left, 1 = hard right)
        """
        left = speed - turn
        right = speed + turn
        max_val = max(abs(left), abs(right), 1.0)
        self.set_speeds(left / max_val, right / max_val)

    def stop(self):
        self.set_speeds(0, 0)

    def cleanup(self):
        if self._initialized:
            self.stop()
            if self._pwm_left:
                self._pwm_left.stop()
            if self._pwm_right:
                self._pwm_right.stop()
            self._GPIO.cleanup()
            logger.info("Motor controller cleaned up")
