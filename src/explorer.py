import math
from dataclasses import dataclass
from typing import List

import numpy as np

from .scene_loader import SceneStats


@dataclass
class Waypoint:
    position: np.ndarray
    look_at: np.ndarray


def _principal_axis(stats: SceneStats) -> np.ndarray:
    extent = stats.bbox_max - stats.bbox_min
    axis = np.zeros(3, dtype=np.float32)
    idx = int(np.argmax(np.abs(extent)))
    axis[idx] = 1.0
    return axis


def generate_exploration_waypoints(
    stats: SceneStats,
    num_orbits: int = 2,
    points_per_orbit: int = 60,
    arc_elevation: float = 0.25,
) -> List[Waypoint]:
    """
    Produce a set of coarse camera waypoints covering the scene.

    The strategy mixes wide orbits with a forward sweep across the scene's
    dominant axis to get parallax and coverage.
    """
    center = stats.center
    base_radius = max(stats.radius * 1.35, stats.scale * 0.75 + 1e-3)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    waypoints: List[Waypoint] = []

    for orbit_id in range(num_orbits):
        radius = base_radius * (1.0 - 0.12 * orbit_id)
        height = stats.radius * (0.15 + 0.1 * orbit_id)
        for i in range(points_per_orbit):
            theta = 2 * math.pi * (i / points_per_orbit) + orbit_id * 0.6
            vertical = math.sin(theta * 0.5) * arc_elevation * stats.radius
            pos = center + np.array(
                [
                    math.cos(theta) * radius,
                    height + vertical,
                    math.sin(theta) * radius,
                ],
                dtype=np.float32,
            )
            waypoints.append(Waypoint(position=pos, look_at=center.copy()))

    # Add a dolly sweep along the largest extent to peek across the interior.
    sweep_axis = _principal_axis(stats)
    sweep_dir = sweep_axis if np.linalg.norm(sweep_axis) > 0 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sweep_dir = sweep_dir / (np.linalg.norm(sweep_dir) + 1e-8)
    sweep_span = base_radius * 1.4
    sweep_height = stats.radius * 0.12
    sweep_start = center + sweep_dir * sweep_span + up * sweep_height
    sweep_end = center - sweep_dir * sweep_span + up * sweep_height * 0.5
    for alpha in np.linspace(0.0, 1.0, num=24):
        pos = sweep_start * (1 - alpha) + sweep_end * alpha
        look_at = center + sweep_dir * (0.15 * base_radius) * (0.5 - alpha)
        waypoints.append(Waypoint(position=pos, look_at=look_at))

    # Finish on a slightly elevated hero shot.
    final_pos = center + np.array([0.0, stats.radius * 0.3, base_radius * 0.8], dtype=np.float32)
    waypoints.append(Waypoint(position=final_pos, look_at=center.copy()))
    return waypoints
