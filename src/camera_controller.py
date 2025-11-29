from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .explorer import GaussianScene


def _lerp(current: float, target: float, alpha: float) -> float:
    alpha = max(0.0, min(1.0, alpha))
    return current + (target - current) * alpha


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


@dataclass
class CameraController:
    """Lightweight first-person camera controller with safety guards."""

    position: np.ndarray
    scene: Optional[GaussianScene] = None
    yaw: float = 0.0
    pitch: float = 0.0
    movement_speed: float = 1.0
    height_lock: float = 1.0
    rotation_damping: float = 6.0
    collision_radius: float = 0.35
    collision_probe_step: float = 0.25
    max_pitch: float = math.radians(85.0)

    target_yaw: float = field(init=False)
    target_pitch: float = field(init=False)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float32)
        self.position[1] = self.height_lock
        self.target_yaw = self.yaw
        self.target_pitch = float(np.clip(self.pitch, -self.max_pitch, self.max_pitch))

    # ------------------------------------------------------------------ #
    # Update loop
    # ------------------------------------------------------------------ #
    def update(
        self,
        forward_input: float,
        yaw_input: float,
        pitch_input: float,
        delta_time: float,
    ) -> None:
        delta_time = max(delta_time, 1e-4)
        self._update_rotation(yaw_input, pitch_input, delta_time)
        self._apply_movement(forward_input, delta_time)
        self.position[1] = self.height_lock

    # ------------------------------------------------------------------ #
    # Rotation handling
    # ------------------------------------------------------------------ #
    def _update_rotation(self, yaw_input: float, pitch_input: float, delta_time: float) -> None:
        self.target_yaw += yaw_input
        self.target_pitch = float(
            np.clip(self.target_pitch + pitch_input, -self.max_pitch, self.max_pitch)
        )
        smoothing = 1.0 - math.exp(-self.rotation_damping * delta_time)
        self.yaw = _lerp(self.yaw, self.target_yaw, smoothing)
        self.pitch = _lerp(self.pitch, self.target_pitch, smoothing)

    # ------------------------------------------------------------------ #
    # Movement & collision
    # ------------------------------------------------------------------ #
    def _apply_movement(self, forward_input: float, delta_time: float) -> None:
        forward = self.forward_vector
        move_amount = max(0.0, forward_input) * self.movement_speed * delta_time
        if move_amount <= 0.0:
            return
        if self._is_blocked(forward, move_amount):
            return
        self.position += forward * move_amount

    def _is_blocked(self, direction: np.ndarray, move_distance: float) -> bool:
        if self.scene is None or getattr(self.scene, "kd_tree", None) is None:
            return False
        max_distance = move_distance + self.collision_radius
        steps = max(1, int(math.ceil(max_distance / self.collision_probe_step)))
        for idx in range(1, steps + 1):
            probe_distance = min(max_distance, idx * self.collision_probe_step)
            probe_point = self.position + direction * probe_distance
            probe_point[1] = self.height_lock
            nearby = self.scene.kd_tree.query_ball_point(probe_point, r=self.collision_radius)
            if nearby:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Derived properties
    # ------------------------------------------------------------------ #
    @property
    def forward_vector(self) -> np.ndarray:
        cos_pitch = math.cos(self.pitch)
        forward = np.array(
            [
                math.cos(self.yaw) * cos_pitch,
                math.sin(self.pitch),
                math.sin(self.yaw) * cos_pitch,
            ],
            dtype=np.float32,
        )
        return _normalize(forward)

    @property
    def right_vector(self) -> np.ndarray:
        forward = self.forward_vector
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up)
        return _normalize(right)

    @property
    def up_vector(self) -> np.ndarray:
        forward = self.forward_vector
        right = self.right_vector
        up = np.cross(right, forward)
        return _normalize(up)
