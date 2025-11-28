"""
Camera path planning for cinematic Gaussian Splatting fly-throughs.

The planner:
- Receives coverage waypoints from `SceneExplorer`.
- Builds smooth, spline-based camera trajectories with ease-in/out motion.
- Enforces collision safety through local density sampling.
- Exposes helper utilities for object-focused highlight tours.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field
from typing import List, Optional, Sequence

import numpy as np

from .explorer import GaussianScene, ObjectCandidate, SceneExplorer, SceneType


@dataclass
class CameraPose:
    position: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    timestamp: float
    fov: float = 55.0

    @property
    def look_at(self) -> np.ndarray:
        return self.position + self.forward


@dataclass
class CameraPath:
    poses: List[CameraPose]
    duration: float
    description: str


@dataclass
class PlannerSettings:
    fps: int = 30
    min_clearance: float = 0.6
    max_acceleration: float = 3.0
    look_ahead: float = 2.0
    up_vector: np.ndarray = dataclass_field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32))


class CameraPathPlanner:
    """Spline-based cinematic trajectory planner."""

    def __init__(
        self,
        explorer: SceneExplorer,
        settings: Optional[PlannerSettings] = None,
        rng_seed: int = 21,
    ) -> None:
        self.explorer = explorer
        self.settings = settings or PlannerSettings()
        self.rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def plan_cinematic_path(
        self,
        scene: GaussianScene,
        waypoints: np.ndarray,
        duration: float = 90.0,
        description: str = "exploratory",
    ) -> CameraPath:
        samples = max(2, int(duration * self.settings.fps))
        base_path = self._build_catmull_rom_path(waypoints, samples * 4)
        positions = self._reparameterize(base_path, samples)
        safe_positions = self._ensure_clearance(scene, positions)
        grounded_positions = self._apply_grounding(scene, safe_positions)
        forwards = self._compute_forward_vectors(grounded_positions)
        poses = self._poses_from_vectors(
            grounded_positions,
            forwards,
            duration,
            bias=scene.scene_type,
        )
        return CameraPath(poses=poses, duration=duration, description=description)

    def plan_object_focus_path(
        self,
        scene: GaussianScene,
        objects: Sequence[ObjectCandidate],
        duration: float = 60.0,
    ) -> CameraPath:
        if not objects:
            raise ValueError("No object candidates provided.")
        waypoints = []
        for idx, obj in enumerate(objects):
            offset_dir = self._sample_unit_sphere()
            offset_dir[1] = max(0.1, offset_dir[1])
            distance = 2.0 if scene.scene_type == SceneType.INDOOR else 5.0
            waypoint = obj.position + offset_dir * distance
            waypoint[1] = np.clip(
                waypoint[1],
                scene.bounds_min[1] + 0.2,
                scene.bounds_max[1] - 0.2,
            )
            waypoints.append(waypoint)
            # Insert intermediate anchor for smoother transitions.
            if idx < len(objects) - 1:
                mid = (waypoint + objects[idx + 1].position) / 2.0
                waypoints.append(mid)
        return self.plan_cinematic_path(
            scene,
            waypoints=np.asarray(waypoints, dtype=np.float32),
            duration=duration,
            description="object_focus",
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_catmull_rom_path(
        self, waypoints: np.ndarray, samples: int
    ) -> np.ndarray:
        points = np.asarray(waypoints, dtype=np.float32)
        if len(points) < 4:
            raise ValueError("Need at least 4 waypoints for Catmull-Rom spline.")
        padded = np.vstack([points[0], points, points[-1]])
        segments = len(points) - 1
        samples_per_segment = max(4, samples // segments)
        trajectory = []
        for i in range(segments):
            p0, p1, p2, p3 = padded[i : i + 4]
            for u in np.linspace(0, 1, samples_per_segment, endpoint=False):
                point = self._catmull_rom(p0, p1, p2, p3, u)
                trajectory.append(point)
        trajectory.append(points[-1])
        return np.asarray(trajectory)

    def _catmull_rom(
        self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float
    ) -> np.ndarray:
        """Standard Catmull-Rom spline interpolation."""

        a = 2 * p1
        b = p2 - p0
        c = 2 * p0 - 5 * p1 + 4 * p2 - p3
        d = -p0 + 3 * p1 - 3 * p2 + p3
        t2 = t * t
        t3 = t2 * t
        return 0.5 * (a + b * t + c * t2 + d * t3)

    def _reparameterize(self, samples: np.ndarray, target_len: int) -> np.ndarray:
        distances = np.linalg.norm(np.diff(samples, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(distances)])
        total_length = cumulative[-1]
        progress = np.linspace(0, 1, target_len)
        eased = progress * progress * (3 - 2 * progress)  # smoothstep
        target_dist = eased * total_length
        interp = np.empty((target_len, 3), dtype=np.float32)
        for axis in range(3):
            interp[:, axis] = np.interp(target_dist, cumulative, samples[:, axis])
        return interp

    def _ensure_clearance(
        self,
        scene: GaussianScene,
        positions: np.ndarray,
    ) -> np.ndarray:
        safe = positions.copy()
        for idx, point in enumerate(positions):
            clearance = self.explorer.estimate_clearance(scene, point)
            density = self.explorer.sample_density(scene, point)
            desired = self.settings.min_clearance + density * 0.4
            if clearance + 1e-4 < desired:
                normal = self._estimate_density_gradient(scene, point)
                if np.linalg.norm(normal) < 1e-3:
                    normal = self._sample_unit_sphere()
                normal = normal / np.linalg.norm(normal)
                safe[idx] = point + normal * (desired - clearance + 1e-2)
            safe[idx] = np.clip(safe[idx], scene.bounds_min + 0.1, scene.bounds_max - 0.1)
        return safe

    def _apply_grounding(self, scene: GaussianScene, positions: np.ndarray) -> np.ndarray:
        """
        Bias camera heights towards a human-held viewpoint and smooth
        vertical movement to prevent large up/down swings.
        """

        grounded = positions.copy()
        floor = float(scene.bounds_min[1]) + 0.2
        ceiling = float(scene.bounds_max[1]) - 0.2
        headroom = 1.8 if scene.scene_type == SceneType.INDOOR else 2.3
        target_top = min(floor + headroom, ceiling)
        min_height = min(floor + 0.4, target_top)
        grounded[:, 1] = np.clip(grounded[:, 1], min_height, target_top)
        grounded[:, 1] = self._smooth_signal(grounded[:, 1], window=19)
        grounded[:, 1] = np.clip(grounded[:, 1], min_height, target_top)
        return grounded

    def _smooth_signal(self, values: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return values
        kernel = np.ones(window, dtype=np.float32) / window
        padded = np.pad(values, (window // 2,), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed.astype(np.float32)

    def _estimate_density_gradient(
        self, scene: GaussianScene, point: np.ndarray, epsilon: float = 0.15
    ) -> np.ndarray:
        gradients = []
        for axis in range(3):
            offset = np.zeros(3, dtype=np.float32)
            offset[axis] = epsilon
            high = self.explorer.sample_density(scene, point + offset)
            low = self.explorer.sample_density(scene, point - offset)
            gradients.append((high - low) / (2 * epsilon))
        return np.asarray(gradients, dtype=np.float32)

    def _compute_forward_vectors(self, positions: np.ndarray) -> np.ndarray:
        forwards = []
        for i, pos in enumerate(positions):
            if i < len(positions) - 1:
                direction = positions[i + 1] - pos
            else:
                direction = pos - positions[i - 1]
            norm = np.linalg.norm(direction)
            if norm < 1e-5:
                direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                norm = 1.0
            forwards.append(direction / norm)
        return np.asarray(forwards)

    def _poses_from_vectors(
        self,
        positions: np.ndarray,
        forwards: np.ndarray,
        duration: float,
        bias: SceneType,
    ) -> List[CameraPose]:
        times = np.linspace(0, duration, len(positions))
        poses = []
        roll_bias = 0.0 if bias == SceneType.INDOOR else 0.05
        for idx, (pos, fwd) in enumerate(zip(positions, forwards)):
            up = self.settings.up_vector.copy()
            if bias == SceneType.OUTDOOR:
                up = self._blend_vectors(up, np.array([0.0, 0.8, 0.2]))
            if roll_bias != 0.0:
                up = self._apply_roll(up, fwd, roll_bias * math.sin(idx / 20.0))
            poses.append(
                CameraPose(
                    position=pos,
                    forward=fwd,
                    up=up / np.linalg.norm(up),
                    timestamp=float(times[idx]),
                    fov=60.0 if bias == SceneType.OUTDOOR else 50.0,
                )
            )
        return poses

    def _apply_roll(self, up: np.ndarray, forward: np.ndarray, angle: float) -> np.ndarray:
        axis = forward / np.linalg.norm(forward)
        sin = math.sin(angle)
        cos = math.cos(angle)
        return (
            up * cos
            + np.cross(axis, up) * sin
            + axis * np.dot(axis, up) * (1 - cos)
        )

    def _blend_vectors(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        vec = 0.7 * a + 0.3 * b
        return vec / np.linalg.norm(vec)

    def _sample_unit_sphere(self) -> np.ndarray:
        vec = self.rng.normal(size=3)
        vec[1] = abs(vec[1])
        return vec / np.linalg.norm(vec)


__all__ = ["CameraPathPlanner", "CameraPose", "CameraPath", "PlannerSettings"]
