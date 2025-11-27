"""
Optional object detector helpers that refine object-centric tours using YOLO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .explorer import GaussianScene, ObjectCandidate
from .path_planner import CameraPose

try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO
except Exception:  # pragma: no cover - degrade gracefully.
    YOLO = None


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2


class ObjectDetector:
    """Thin wrapper around Ultralytics YOLO models."""

    def __init__(self, model_name: str = "yolov8n.pt", device: str | None = None):
        self.model = None
        if YOLO:
            try:
                self.model = YOLO(model_name)
                if device:
                    self.model.to(device)
            except Exception:
                self.model = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.model is None:
            return []
        results = self.model.predict(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            xyxy = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    label=self.model.names[int(box.cls)],
                    confidence=float(box.conf),
                    bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                )
            )
        return detections

    def select_objects_from_frames(
        self,
        scene: GaussianScene,
        candidates: Sequence[ObjectCandidate],
        frames: Sequence[np.ndarray],
        poses: Sequence[CameraPose],
        resolution: Tuple[int, int],
    ) -> List[ObjectCandidate]:
        """Boost candidates that overlap with detections on preview frames."""

        if not candidates:
            return []
        scores = np.array([cand.score for cand in candidates], dtype=np.float32)
        for frame, pose in zip(frames, poses):
            detections = self.detect(frame)
            if not detections:
                continue
            for idx, cand in enumerate(candidates):
                pixel = self._project_point(pose, cand.position, resolution)
                if pixel is None:
                    continue
                for det in detections:
                    if self._pixel_in_bbox(pixel, det.bbox):
                        scores[idx] += det.confidence * 0.5
        ranked = [
            ObjectCandidate(name=c.name, position=c.position, score=float(s))
            for c, s in zip(candidates, scores)
        ]
        ranked.sort(key=lambda c: c.score, reverse=True)
        return ranked

    def _project_point(
        self,
        pose: CameraPose,
        point: np.ndarray,
        resolution: Tuple[int, int],
    ) -> Tuple[float, float] | None:
        width, height = resolution
        forward = pose.forward / np.linalg.norm(pose.forward)
        right = np.cross(forward, pose.up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        rel = point - pose.position
        cam_x = np.dot(right, rel)
        cam_y = np.dot(up, rel)
        cam_z = np.dot(-forward, rel)
        if cam_z <= 0:
            return None
        fx = fy = 0.5 * width / np.tan(np.radians(pose.fov * 0.5))
        cx = width / 2.0
        cy = height / 2.0
        px = fx * (cam_x / cam_z) + cx
        py = fy * (cam_y / cam_z) + cy
        if px < 0 or px >= width or py < 0 or py >= height:
            return None
        return px, py

    @staticmethod
    def _pixel_in_bbox(pixel: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> bool:
        px, py = pixel
        x1, y1, x2, y2 = bbox
        return x1 <= px <= x2 and y1 <= py <= y2


__all__ = ["ObjectDetector", "Detection"]
