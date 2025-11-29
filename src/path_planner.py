"""
Camera path planning for cinematic Gaussian Splatting fly-throughs.

This overhaul focuses on robust indoor navigation:
- Build a 2D navigation field from the scene density grid to stay inside the
  captured space (no more "black void" starts).
- Plan purely horizontal camera motion with smooth yaw-only rotations.
- Use collision-aware grid navigation (A*) plus Catmull–Rom smoothing for
  graceful travel between anchors.
- Keep camera height steady at a comfortable eye level, avoiding vertical
  bounces.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field as dataclass_field
from typing import List, Optional, Sequence

import numpy as np
from scipy import ndimage

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
    camera_radius: float = 0.35
    max_acceleration: float = 3.0
    look_ahead: float = 2.0
    up_vector: np.ndarray = dataclass_field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32))


class CameraPathPlanner:
    """Navigation-field-based cinematic trajectory planner."""

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
        nav_field = self._build_navigation_field(scene)
        start = self._choose_start(nav_field)
        anchor_points = self._project_anchors(nav_field, waypoints)
        ordered_targets = self._order_targets(start, anchor_points)
        route = self._stitch_route(nav_field, start, ordered_targets)
        smooth_route = self._smooth_route(route, samples=int(duration * 2))
        samples = max(2, int(duration * self.settings.fps))
        positions = self._reparameterize(smooth_route, samples)
        positions = self._lift_to_world(nav_field, positions)
        forwards = self._compute_horizontal_forwards(positions)
        forwards = self._smooth_vectors(forwards, window=9)
        poses = self._poses_from_vectors(
            positions,
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
        anchors = []
        for obj in objects:
            offset = self._sample_unit_circle() * (
                1.6 if scene.scene_type == SceneType.INDOOR else 3.5
            )
            anchor = obj.position.copy()
            anchor[0] += offset[0]
            anchor[2] += offset[1]
            anchors.append(anchor)
        return self.plan_cinematic_path(scene, np.asarray(anchors), duration, "object_focus")

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    # --- Navigation field construction -------------------------------------------------

    def _build_navigation_field(self, scene: GaussianScene) -> "NavigationField":
        density_xz = scene.density_grid.max(axis=1)
        blurred = ndimage.gaussian_filter(density_xz, sigma=1.2)

        density_threshold = max(0.1, float(np.percentile(blurred, 80)))
        filled_mask = blurred > 0.02
        support_limit = max(4, int(scene.density_grid.shape[0] * 0.12))
        support_distance = ndimage.distance_transform_edt(~filled_mask)
        support_mask = support_distance <= support_limit

        occupancy = blurred >= density_threshold
        safety_cells = self._safety_cells(scene.grid_spacing)
        grown_obstacles = ndimage.binary_dilation(occupancy, iterations=safety_cells)
        free_mask = support_mask & ~grown_obstacles

        floor_height = float(np.percentile(scene.positions[:, 1], 5))
        ceiling = float(np.percentile(scene.positions[:, 1], 95))
        eye_height = np.clip(floor_height + 1.5, floor_height + 0.5, ceiling - 0.3)

        return NavigationField(
            free_mask=free_mask,
            support_mask=support_mask,
            origin=scene.grid_origin,
            spacing=scene.grid_spacing,
            eye_height=eye_height,
        )

    def _safety_cells(self, spacing: np.ndarray) -> int:
        horizontal_spacing = float(min(spacing[0], spacing[2]))
        clearance = self.settings.camera_radius + self.settings.min_clearance
        return max(1, int(math.ceil(clearance / max(horizontal_spacing, 1e-4))))

    def _choose_start(self, nav: "NavigationField") -> np.ndarray:
        if not np.any(nav.free_mask):
            center = (np.array(nav.free_mask.shape) // 2).astype(int)
            return center
        distance = ndimage.distance_transform_edt(nav.free_mask)
        idx = np.unravel_index(np.argmax(distance), distance.shape)
        return np.array(idx, dtype=int)

    def _project_anchors(self, nav: "NavigationField", waypoints: np.ndarray) -> np.ndarray:
        anchors = []
        for point in np.asarray(waypoints, dtype=np.float32):
            cell = self._world_to_cell(nav, point)
            snapped = self._snap_to_free(nav, cell)
            anchors.append(snapped)
        if not anchors:
            anchors.append(self._choose_start(nav))
        return np.asarray(anchors, dtype=int)

    def _order_targets(self, start: np.ndarray, anchors: np.ndarray) -> List[np.ndarray]:
        remaining = anchors.tolist()
        ordered: List[np.ndarray] = []
        current = np.array(start, dtype=int)
        while remaining:
            distances = [np.linalg.norm(np.array(cell) - current) for cell in remaining]
            idx = int(np.argmax(distances))
            current = np.array(remaining.pop(idx), dtype=int)
            ordered.append(current)
        return ordered

    def _stitch_route(
        self, nav: "NavigationField", start: np.ndarray, targets: Sequence[np.ndarray]
    ) -> np.ndarray:
        cells = [start]
        current = start
        for target in targets:
            path = self._astar(nav.free_mask, current, target)
            if len(path) == 0:
                break
            cells.extend(path[1:])
            current = target
        xz_world = [self._cell_to_xz(nav, cell) for cell in cells]
        return np.asarray(xz_world, dtype=np.float32)

    def _smooth_route(self, route: np.ndarray, samples: int) -> np.ndarray:
        if len(route) < 4:
            return self._reparameterize(route, samples)
        padded = np.vstack([route[0], route, route[-1]])
        segments = len(route) - 1
        samples_per_segment = max(4, samples // segments)
        smoothed = []
        for i in range(segments):
            p0, p1, p2, p3 = padded[i : i + 4]
            for t in np.linspace(0, 1, samples_per_segment, endpoint=False):
                smoothed.append(self._catmull_rom(p0, p1, p2, p3, t))
        smoothed.append(route[-1])
        return np.asarray(smoothed, dtype=np.float32)

    # --- Geometry helpers -------------------------------------------------------------

    def _catmull_rom(
        self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float
    ) -> np.ndarray:
        a = 2 * p1
        b = p2 - p0
        c = 2 * p0 - 5 * p1 + 4 * p2 - p3
        d = -p0 + 3 * p1 - 3 * p2 + p3
        t2 = t * t
        t3 = t2 * t
        return 0.5 * (a + b * t + c * t2 + d * t3)

    def _reparameterize(self, samples: np.ndarray, target_len: int) -> np.ndarray:
        if len(samples) == 1:
            return np.repeat(samples, target_len, axis=0)
        distances = np.linalg.norm(np.diff(samples, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(distances)])
        total_length = max(cumulative[-1], 1e-6)
        progress = np.linspace(0, 1, target_len)
        eased = progress * progress * (3 - 2 * progress)
        target_dist = eased * total_length
        interp = np.empty((target_len, samples.shape[1]), dtype=np.float32)
        for axis in range(samples.shape[1]):
            interp[:, axis] = np.interp(target_dist, cumulative, samples[:, axis])
        return interp

    def _smooth_signal(self, values: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return values
        kernel = np.ones(window, dtype=np.float32) / window
        padded = np.pad(values, (window // 2,), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed.astype(np.float32)

    def _smooth_positions(self, positions: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return positions
        smoothed = positions.copy()
        for axis in range(positions.shape[1]):
            smoothed[:, axis] = self._smooth_signal(positions[:, axis], window)
        return smoothed

    def _smooth_vectors(self, vectors: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return vectors
        smoothed = self._smooth_positions(vectors, window)
        norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-6, None)
        return smoothed / norms

    # --- Coordinate conversions -------------------------------------------------------

    def _world_to_cell(self, nav: "NavigationField", point: np.ndarray) -> np.ndarray:
        rel = (point[[0, 2]] - nav.origin[[0, 2]]) / nav.spacing[[0, 2]]
        cell = np.round(rel).astype(int)
        cell = np.clip(cell, 0, np.array(nav.free_mask.shape) - 1)
        return cell

    def _cell_to_xz(self, nav: "NavigationField", cell: np.ndarray) -> np.ndarray:
        return nav.origin[[0, 2]] + (np.asarray(cell) + 0.5) * nav.spacing[[0, 2]]

    def _lift_to_world(self, nav: "NavigationField", positions: np.ndarray) -> np.ndarray:
        lifted = np.zeros((len(positions), 3), dtype=np.float32)
        lifted[:, 0] = positions[:, 0]
        lifted[:, 2] = positions[:, 1]
        lifted[:, 1] = nav.eye_height
        return lifted

    # --- Grid navigation --------------------------------------------------------------

    def _snap_to_free(self, nav: "NavigationField", cell: np.ndarray) -> np.ndarray:
        if nav.free_mask[tuple(cell)]:
            return cell
        free_cells = np.argwhere(nav.free_mask)
        if len(free_cells) == 0:
            return np.clip(cell, 0, np.array(nav.free_mask.shape) - 1)
        distances = np.linalg.norm(free_cells - cell, axis=1)
        idx = int(np.argmin(distances))
        return free_cells[idx]

    def _astar(self, free_mask: np.ndarray, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        start_t = tuple(start)
        goal_t = tuple(goal)
        if start_t == goal_t:
            return [start]
        height, width = free_mask.shape
        open_set: List[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, start_t))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = {start_t: 0.0}

        neighbors = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_t:
                return self._reconstruct_path(came_from, current)
            for dx, dz in neighbors:
                nx, nz = current[0] + dx, current[1] + dz
                if nx < 0 or nz < 0 or nx >= height or nz >= width:
                    continue
                if not free_mask[nx, nz]:
                    continue
                tentative = g_score[current] + math.hypot(dx, dz)
                neighbor = (nx, nz)
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score = tentative + math.hypot(goal_t[0] - nx, goal_t[1] - nz)
                    heapq.heappush(open_set, (f_score, neighbor))
        return []

    def _reconstruct_path(
        self, came_from: dict[tuple[int, int], tuple[int, int]], current: tuple[int, int]
    ) -> List[np.ndarray]:
        path = [np.array(current, dtype=int)]
        while current in came_from:
            current = came_from[current]
            path.append(np.array(current, dtype=int))
        path.reverse()
        return path

    def _poses_from_vectors(
        self,
        positions: np.ndarray,
        forwards: np.ndarray,
        duration: float,
        bias: SceneType,
    ) -> List[CameraPose]:
        times = np.linspace(0, duration, len(positions))
        poses = []
        for idx, (pos, fwd) in enumerate(zip(positions, forwards)):
            poses.append(
                CameraPose(
                    position=pos,
                    forward=fwd,
                    up=self.settings.up_vector,
                    timestamp=float(times[idx]),
                    fov=55.0,
                )
            )
        return poses

    def _compute_horizontal_forwards(self, positions: np.ndarray) -> np.ndarray:
        forwards = []
        for i, pos in enumerate(positions):
            if i < len(positions) - 1:
                direction = positions[i + 1] - pos
            else:
                direction = pos - positions[i - 1]
            direction[1] = 0.0
            norm = np.linalg.norm(direction[[0, 2]])
            if norm < 1e-5:
                direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                direction = direction / norm
            forwards.append(direction)
        return np.asarray(forwards, dtype=np.float32)

    def _sample_unit_circle(self) -> np.ndarray:
        angle = float(self.rng.uniform(0, 2 * math.pi))
        return np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)


@dataclass
class NavigationField:
    free_mask: np.ndarray
    support_mask: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    eye_height: float


__all__ = ["CameraPathPlanner", "CameraPose", "CameraPath", "PlannerSettings"]
