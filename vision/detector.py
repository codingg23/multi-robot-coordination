"""
detector.py

Object detection and landmark recognition using YOLOv8-nano.

YOLOv8-nano was the only model that runs at acceptable FPS on a Pi 4.
Even then it's ~5.5 FPS at 320x320, which is enough for our purposes.

For landmarks (used for loop closure in the grid) we use a simpler
colour-blob detector - no neural net needed for brightly coloured markers.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional
import time

logger = logging.getLogger(__name__)

# Target colours for the search-and-retrieve task
# HSV ranges work better than RGB for colour detection under varying lighting
TARGET_COLOURS = {
    "red": [(0, 120, 70), (10, 255, 255)],      # red wraps around in HSV
    "red2": [(170, 120, 70), (180, 255, 255)],  # second red range
    "blue": [(100, 150, 50), (130, 255, 255)],
    "green": [(40, 60, 40), (80, 255, 255)],
    "yellow": [(20, 100, 100), (30, 255, 255)],
}

# Landmark colours (bright markers placed in the environment)
LANDMARK_COLOURS = {
    "pink": [(140, 100, 100), (170, 255, 255)],
    "orange": [(10, 150, 100), (25, 255, 255)],
}


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 in pixels
    centre_px: tuple[int, int]
    distance_est_m: Optional[float] = None  # estimated from bbox height


@dataclass
class LandmarkDetection:
    landmark_id: str    # colour name
    centre_px: tuple[int, int]
    area_px: int


class ObjectDetector:
    """
    Wraps YOLOv8-nano for object detection on the Pi camera feed.

    On Pi 4 with 4GB RAM, inference at 320x320 takes ~180ms per frame.
    I tried 640x640 but that was too slow (~450ms), not usable for real-time.
    """

    def __init__(self, model_path: str = "yolov8n.pt", target_label: str = "sports ball"):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model_loaded = True
            logger.info(f"YOLOv8 model loaded: {model_path}")
        except ImportError:
            logger.warning("ultralytics not installed, detection disabled")
            self.model = None
            self.model_loaded = False

        self.target_label = target_label
        self.inference_size = 320
        self._last_inference_ms = 0.0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a frame. Returns list of detections."""
        if not self.model_loaded:
            return []

        t0 = time.time()
        results = self.model(frame, imgsz=self.inference_size, verbose=False)
        self._last_inference_ms = (time.time() - t0) * 1000

        detections = []
        for r in results:
            for box in r.boxes:
                label = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if conf < 0.45:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                bbox_h = y2 - y1

                dist_est = None
                if label == self.target_label and bbox_h > 0:
                    # rough distance estimate using known object size
                    # calibrated for a ~7cm diameter ball: focal_px * real_h / bbox_h
                    FOCAL_PX = 280  # empirically calibrated
                    REAL_H_M = 0.07
                    dist_est = (FOCAL_PX * REAL_H_M) / bbox_h

                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    centre_px=(cx, cy),
                    distance_est_m=dist_est,
                ))

        return detections

    @property
    def fps_estimate(self) -> float:
        if self._last_inference_ms > 0:
            return 1000 / self._last_inference_ms
        return 0.0


class ColourBlobDetector:
    """
    Simple HSV colour blob detector for landmarks.

    Much faster than YOLO (~1ms per frame). Used for detecting coloured
    markers placed around the environment for loop closure.
    """

    def __init__(self, min_area_px: int = 300):
        self.min_area_px = min_area_px

    def detect_landmarks(self, frame: np.ndarray) -> list[LandmarkDetection]:
        """Find coloured landmark markers in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        landmarks = []

        for name, (lower, upper) in LANDMARK_COLOURS.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area_px:
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                landmarks.append(LandmarkDetection(
                    landmark_id=name,
                    centre_px=(cx, cy),
                    area_px=int(area),
                ))

        return landmarks

    def detect_target(self, frame: np.ndarray, colour: str = "red") -> Optional[tuple[int, int, int]]:
        """
        Detect a single target colour. Returns (cx, cy, area) or None.
        Used when we already know what colour we're looking for.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if colour == "red":
            # red spans two HSV ranges
            r1 = TARGET_COLOURS["red"]
            r2 = TARGET_COLOURS["red2"]
            mask = cv2.bitwise_or(
                cv2.inRange(hsv, np.array(r1[0]), np.array(r1[1])),
                cv2.inRange(hsv, np.array(r2[0]), np.array(r2[1])),
            )
        elif colour in TARGET_COLOURS:
            r = TARGET_COLOURS[colour]
            mask = cv2.inRange(hsv, np.array(r[0]), np.array(r[1]))
        else:
            return None

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 400:
            return None

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy, int(area))

DEFAULT_MIN_BLOB_AREA = 300  # minimum pixel area to count as a detection
