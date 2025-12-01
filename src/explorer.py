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
    Generates a cinematic flyby path.
    """
    center = stats.center
    waypoints: List[Waypoint] = []

    # Determine the principal axis of the scene (longest dimension)
    extent = stats.bbox_max - stats.bbox_min
    principal_idx = int(np.argmax(extent))
    
    # Create a path that flies through the scene along the principal axis
    # We'll use a sine wave pattern for lateral movement to make it more interesting
    
    # Start from one end of the bounding box
    axis_len = extent[principal_idx]
    start_val = stats.bbox_min[principal_idx] + axis_len * 0.1
    end_val = stats.bbox_max[principal_idx] - axis_len * 0.1
    
    num_steps = 100
    
    # Secondary axes indices
    other_indices = [i for i in range(3) if i != principal_idx]
    idx1, idx2 = other_indices[0], other_indices[1]
    
    # Amplitude for the sine wave motion
    amp1 = extent[idx1] * 0.2
    amp2 = extent[idx2] * 0.1
    
    for i in range(num_steps):
        t = i / (num_steps - 1)
        
        # Position along principal axis
        curr_val = start_val + (end_val - start_val) * t
        
        # Lateral motion
        offset1 = math.sin(t * math.pi * 2) * amp1
        offset2 = math.cos(t * math.pi * 1.5) * amp2
        
        pos = np.zeros(3, dtype=np.float32)
        pos[principal_idx] = curr_val
        pos[idx1] = center[idx1] + offset1
        pos[idx2] = center[idx2] + offset2
        
        # Look slightly ahead on the path
        look_t = min(1.0, t + 0.1)
        look_val = start_val + (end_val - start_val) * look_t
        look_at = np.zeros(3, dtype=np.float32)
        look_at[principal_idx] = look_val
        look_at[idx1] = center[idx1] + math.sin(look_t * math.pi * 2) * amp1 * 0.5
        look_at[idx2] = center[idx2] + math.cos(look_t * math.pi * 1.5) * amp2 * 0.5
        
        waypoints.append(Waypoint(position=pos, look_at=look_at))

    return waypoints
