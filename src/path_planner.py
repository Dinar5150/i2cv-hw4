import numpy as np
from dataclasses import dataclass
from typing import List, Sequence, Optional

from scipy.interpolate import CubicSpline

from .explorer import Waypoint
from .scene_loader import SceneStats


@dataclass
class CameraPose:
    position: np.ndarray
    rotation: np.ndarray
    look_at: np.ndarray
    fov: float


def _smoothstep(x: np.ndarray) -> np.ndarray:
    return x * x * (3 - 2 * x)


def _look_at_rotation(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        forward_norm = 1.0
    forward = forward / forward_norm
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right_norm = 1.0
    right = right / right_norm
    true_up = np.cross(right, forward)
    rot = np.stack([right, true_up, -forward], axis=1)
    return rot


def _estimate_path_length(points: np.ndarray) -> float:
    deltas = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def _sample_spline(points: np.ndarray, num: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, points.shape[0])
    t_query = _smoothstep(np.linspace(0.0, 1.0, num))
    splines = [CubicSpline(t, points[:, i]) for i in range(3)]
    samples = np.stack([splines[i](t_query) for i in range(3)], axis=1)
    return samples


def smooth_camera_path(
    waypoints: Sequence[Waypoint],
    stats: SceneStats,
    fps: int = 24,
    nominal_speed: float = 0.8,
    fov: float = 60.0,
    hold_seconds: float = 0.6,
    forced_duration: Optional[float] = None,
) -> List[CameraPose]:
    """
    Turn coarse waypoints into a dense, smooth camera path with orientations.

    Waypoints drive both position and aim; we spline-interpolate positions and
    look-at targets separately, then build look-at rotation matrices. A short
    hold is added at the start and end for cinematic ease-in/out.
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least two waypoints to build a path.")

    positions_wp = np.stack([w.position for w in waypoints], axis=0)
    look_at_wp = np.stack([w.look_at for w in waypoints], axis=0)

    # Estimate total frames from path length and desired speed.
    if forced_duration is not None:
        duration = forced_duration
    else:
        coarse_length = _estimate_path_length(positions_wp)
        duration = max(coarse_length / max(nominal_speed, 1e-3), 4.0)

    base_frames = int(duration * fps)
    num_frames = max(base_frames, len(waypoints) * 8)

    positions = _sample_spline(positions_wp, num_frames)
    look_targets = _sample_spline(look_at_wp, num_frames)
    
    # Clamp the interpolated positions to the safe bounds
    # Re-calculate safe bounds (same logic as explorer.py)
    bbox_size = stats.bbox_max - stats.bbox_min
    safe_margin = 0.15
    safe_min = stats.bbox_min + bbox_size * safe_margin
    safe_max = stats.bbox_max - bbox_size * safe_margin
    
    positions = np.maximum(positions, safe_min)
    positions = np.minimum(positions, safe_max)

    # Use path derivative to bias the look-at direction for forward motion.
    t = np.linspace(0.0, 1.0, positions_wp.shape[0])
    splines = [CubicSpline(t, positions_wp[:, i]) for i in range(3)]
    derivatives = np.stack([splines[i](np.linspace(0, 1, num_frames), 1) for i in range(3)], axis=1)

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    poses: List[CameraPose] = []
    for pos, look, deriv in zip(positions, look_targets, derivatives):
        forward_hint = look - pos + 0.35 * deriv
        aim = pos + forward_hint
        rot = _look_at_rotation(pos, aim, up)
        poses.append(CameraPose(position=pos.astype(np.float32), rotation=rot.astype(np.float32), look_at=aim.astype(np.float32), fov=float(fov)))

    # Add holds at the beginning and end.
    hold_frames = int(hold_seconds * fps)
    if hold_frames > 0 and poses:
        poses = [poses[0]] * hold_frames + poses + [poses[-1]] * hold_frames
    return poses
