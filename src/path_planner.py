"""Minimal, forward-only camera path planner for Gaussian splat scenes.

The previous navigation-field planner has been replaced with a simpler,
robust approach tailored for cinematic interior fly-throughs:

- Detect a safe interior corridor using scene bounds percentiles.
- Align the camera motion to the dominant horizontal axis (PCA on XZ).
- Keep the camera inside the scene bounds with generous margins to avoid
  wall clipping and exits.
- Move only forward with gently varying lateral offsets for cinematic
  parallax while maintaining smooth motion and constant height.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .explorer import GaussianScene


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
    duration: float = 75.0
    margin_ratio: float = 0.08
    lateral_ratio: float = 0.12
    min_travel_distance: float = 2.5
    min_extent: float = 1.0
    min_clearance: float = 0.35
    up_vector: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )


class CameraPathPlanner:
    """Generates smooth, forward-only interior camera paths."""

    def __init__(self, settings: PlannerSettings | None = None) -> None:
        self.settings = settings or PlannerSettings()

    def plan_cinematic_path(self, scene: GaussianScene) -> CameraPath:
        safe_min, safe_max = self._safe_bounds(scene)
        direction = self._principal_direction(scene)
        center = (safe_min + safe_max) / 2.0

        travel_extent, offset_start = self._travel_extents(direction, safe_min, safe_max)
        perp = self._perpendicular(direction)
        amplitude = self._lateral_amplitude(safe_min, safe_max)

        num_frames = int(self.settings.duration * self.settings.fps)
        times = np.linspace(0.0, self.settings.duration, num_frames)
        positions = []
        for t in np.linspace(0.0, 1.0, num_frames):
            forward_progress = offset_start + t * travel_extent
            base = center + forward_progress * direction
            sway = amplitude * np.sin(t * np.pi * 2.0) * perp
            pos = base + sway
            pos = np.clip(pos, safe_min, safe_max)
            pos[1] = self._eye_height(scene, safe_min, safe_max)
            positions.append(pos)
        positions = np.asarray(positions, dtype=np.float32)
        positions = self._smooth_positions(positions, window=7)
        positions = self._apply_clearance(scene, positions)
        positions = self._smooth_positions(positions, window=5)

        forwards = self._forward_vectors(positions)
        poses = [
            CameraPose(
                position=pos,
                forward=fwd,
                up=self.settings.up_vector,
                timestamp=float(times[idx]),
                fov=55.0,
            )
            for idx, (pos, fwd) in enumerate(zip(positions, forwards))
        ]
        return CameraPath(
            poses=poses,
            duration=self.settings.duration,
            description="exploratory",
        )

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def _safe_bounds(self, scene: GaussianScene) -> tuple[np.ndarray, np.ndarray]:
        margin = max(scene.diagonal_length * self.settings.margin_ratio, 0.35)
        lower = np.percentile(scene.positions, 5, axis=0)
        upper = np.percentile(scene.positions, 95, axis=0)
        safe_min = np.maximum(lower + margin, scene.bounds_min)
        safe_max = np.minimum(upper - margin, scene.bounds_max)
        if np.any(safe_max <= safe_min):
            center = (scene.bounds_min + scene.bounds_max) / 2.0
            fallback_extent = max(
                self.settings.min_extent, 0.08 * scene.diagonal_length, 0.5
            )
            half = fallback_extent * 0.5
            safe_min = np.maximum(center - half, scene.bounds_min)
            safe_max = np.minimum(center + half, scene.bounds_max)
            if np.any(safe_max <= safe_min):
                epsilon = max(scene.diagonal_length * 0.01, 0.1)
                safe_min = center - epsilon
                safe_max = center + epsilon
        return safe_min.astype(np.float32), safe_max.astype(np.float32)

    def _principal_direction(self, scene: GaussianScene) -> np.ndarray:
        xz = scene.positions[:, [0, 2]] - scene.positions[:, [0, 2]].mean(axis=0)
        cov = np.cov(xz.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, int(np.argmax(eigvals))]
        direction = np.array([principal[0], 0.0, principal[1]], dtype=np.float32)
        norm = np.linalg.norm(direction[[0, 2]])
        if norm < 1e-5:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            direction /= norm
        return direction

    def _perpendicular(self, direction: np.ndarray) -> np.ndarray:
        perp = np.array([-direction[2], 0.0, direction[0]], dtype=np.float32)
        norm = np.linalg.norm(perp[[0, 2]])
        return perp / max(norm, 1e-5)

    def _travel_extents(
        self, direction: np.ndarray, safe_min: np.ndarray, safe_max: np.ndarray
    ) -> tuple[float, float]:
        corners = [
            np.array([x, 0.0, z], dtype=np.float32)
            for x in (safe_min[0], safe_max[0])
            for z in (safe_min[2], safe_max[2])
        ]
        projections = [float(np.dot(corner, direction)) for corner in corners]
        proj_min, proj_max = min(projections), max(projections)
        span = proj_max - proj_min
        travel_extent = max(span * 0.9, self.settings.min_travel_distance)
        start_offset = proj_min + (span - travel_extent) / 2.0 - np.dot(
            (safe_min + safe_max) / 2.0, direction
        )
        return travel_extent, start_offset

    def _lateral_amplitude(
        self, safe_min: np.ndarray, safe_max: np.ndarray
    ) -> float:
        width = min(safe_max[0] - safe_min[0], safe_max[2] - safe_min[2])
        return max(0.05, width * self.settings.lateral_ratio)

    def _eye_height(
        self, scene: GaussianScene, safe_min: np.ndarray, safe_max: np.ndarray
    ) -> float:
        center = (scene.bounds_min + scene.bounds_max) / 2.0
        span = scene.bounds_max - scene.bounds_min
        xz_mask = (
            np.abs(scene.positions[:, 0] - center[0]) <= 0.35 * span[0]
        ) & (
            np.abs(scene.positions[:, 2] - center[2]) <= 0.35 * span[2]
        )
        band = scene.positions[xz_mask][:, 1] if np.any(xz_mask) else scene.positions[:, 1]
        floor = float(np.percentile(band, 10))
        mid = float(np.percentile(band, 70))
        height = floor + 0.25 * max(mid - floor, 0.0)
        return float(np.clip(height, safe_min[1] + 0.15, safe_max[1] - 0.35))

    def _smooth_positions(self, positions: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return positions
        padded = np.pad(positions, ((window // 2, window // 2), (0, 0)), mode="edge")
        kernel = np.ones(window, dtype=np.float32) / window
        smoothed = np.empty_like(positions)
        for axis in range(positions.shape[1]):
            smoothed[:, axis] = np.convolve(padded[:, axis], kernel, mode="valid")
        return smoothed

    def _forward_vectors(self, positions: np.ndarray) -> np.ndarray:
        """Derive stable forward directions, reusing last valid when steps are tiny."""
        diffs = np.diff(positions, axis=0, prepend=positions[[0]])
        diffs[0] = diffs[1]
        diffs[:, 1] = 0.0  # keep horizontal forward motion

        forwards = np.zeros_like(diffs, dtype=np.float32)
        last_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        for i, diff in enumerate(diffs):
            norm = np.linalg.norm(diff)
            if norm > 1e-4 and np.isfinite(norm):
                last_dir = (diff / norm).astype(np.float32)
            forwards[i] = last_dir
        return forwards

    def _apply_clearance(self, scene: GaussianScene, positions: np.ndarray) -> np.ndarray:
        """Push poses away from dense splats to avoid collisions."""
        positions = positions.copy()
        for _ in range(3):  # iterative relax to escape walls
            distances, idx = scene.kd_tree.query(positions, k=1)
            too_close = distances < self.settings.min_clearance
            if not np.any(too_close):
                break
            nearest = scene.positions[idx[too_close]]
            direction = positions[too_close] - nearest
            norms = np.linalg.norm(direction, axis=1, keepdims=True)
            direction = direction / np.clip(norms, 1e-3, None)
            offsets = (self.settings.min_clearance - distances[too_close])[:, None] * direction
            positions[too_close] += offsets
            positions = np.clip(positions, scene.bounds_min, scene.bounds_max)
        return positions


__all__ = ["CameraPathPlanner", "CameraPose", "CameraPath", "PlannerSettings"]
