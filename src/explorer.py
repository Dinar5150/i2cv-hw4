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
    Generates a cinematic flyby path starting from the center.
    """
    center = stats.center
    waypoints: List[Waypoint] = []

    # Determine the principal axis of the scene (longest dimension)
    extent = stats.bbox_max - stats.bbox_min
    principal_idx = int(np.argmax(extent))
    
    # Start exactly at the center (0,0,0)
    # Then move outwards along the principal axis in a figure-8 or spiral
    
    # Secondary axes indices
    other_indices = [i for i in range(3) if i != principal_idx]
    idx1, idx2 = other_indices[0], other_indices[1]
    
    # Amplitude for the motion (stay well within bounds)
    amp_principal = extent[principal_idx] * 0.4
    amp1 = extent[idx1] * 0.3
    amp2 = extent[idx2] * 0.15
    
    num_steps = 120
    
    for i in range(num_steps):
        t = i / (num_steps - 1)
        # Map t to a phase that starts at 0
        phase = t * math.pi * 2.0
        
        # Lissajous-like path starting at (0,0,0)
        # sin(t) starts at 0.
        val_principal = math.sin(phase) * amp_principal
        val_1 = math.sin(phase * 2.0) * amp1
        val_2 = math.cos(phase) * amp2 - amp2 # Start at 0 if cos(0)=1, so subtract amp2? 
        # Better: use sin for all to ensure 0 start
        val_2 = math.sin(phase * 3.0) * amp2

        pos = np.zeros(3, dtype=np.float32)
        pos[principal_idx] = center[principal_idx] + val_principal
        pos[idx1] = center[idx1] + val_1
        pos[idx2] = center[idx2] + val_2
        
        # Look slightly ahead on the path
        # To move "forward", we look at where we will be in the future.
        # If the camera feels like it's moving backward, it means the look_at is "behind" the motion vector.
        # Let's look further ahead along the path.
        look_phase = phase + 0.1  # Small positive delta looks "forward" in time
        
        look_principal = math.sin(look_phase) * amp_principal
        look_1 = math.sin(look_phase * 2.0) * amp1
        look_2 = math.sin(look_phase * 3.0) * amp2
        
        look_at = np.zeros(3, dtype=np.float32)
        look_at[principal_idx] = center[principal_idx] + look_principal
        look_at[idx1] = center[idx1] + look_1
        look_at[idx2] = center[idx2] + look_2
        
        waypoints.append(Waypoint(position=pos, look_at=look_at))

    return waypoints
