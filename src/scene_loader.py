import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from plyfile import PlyData


@dataclass
class SceneData:
    positions: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    shs: Optional[torch.Tensor]
    colors_precomp: Optional[torch.Tensor]
    device: torch.device


@dataclass
class SceneStats:
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    center: np.ndarray
    radius: float
    scale: float


@dataclass
class SceneTransform:
    world_from_scene: np.ndarray
    scene_from_world: np.ndarray
    scale_factor: float
    translation: np.ndarray


def load_ply(path: str, device: torch.device):
    plydata = PlyData.read(path)
    v = plydata.elements[0]

    # Positions
    positions = np.stack((v["x"], v["y"], v["z"]), axis=1)

    # Opacity: Standard 3DGS PLY stores logits, we need sigmoid for gsplat v1.0+
    # If the PLY is uncompressed/raw, it might already be in [0,1] or logit.
    # Standard 3DGS convention is logits.
    opacities = v["opacity"][..., None]
    opacities = 1.0 / (1.0 + np.exp(-opacities))

    # Scales: Standard 3DGS PLY stores log-scales, we need exp
    scale_names = [p.name for p in v.properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.stack([v[n] for n in scale_names], axis=1)
    scales = np.exp(scales)

    # Rotations: (N, 4)
    rot_names = [p.name for p in v.properties if p.name.startswith("rot_")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rotations = np.stack([v[n] for n in rot_names], axis=1)
    # Normalize quaternions
    norm = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotations = rotations / (norm + 1e-8)

    # Spherical Harmonics
    # DC
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).reshape(-1, 1, 3)
    
    # Rest
    f_rest_names = [p.name for p in v.properties if p.name.startswith("f_rest_")]
    f_rest_names = sorted(f_rest_names, key=lambda x: int(x.split('_')[-1]))
    
    if f_rest_names:
        f_rest = np.stack([v[n] for n in f_rest_names], axis=1).reshape(-1, len(f_rest_names) // 3, 3)
        shs = np.concatenate([f_dc, f_rest], axis=1)
    else:
        shs = f_dc
    
    # Flatten to (N, K*3) if your SceneData expects that, or keep as (N, K, 3)
    # Based on renderer.py, it handles flattened or structured, but let's flatten to match typical gsplat.io behavior
    shs = shs.reshape(shs.shape[0], -1)

    return (
        torch.from_numpy(positions).float().to(device),
        torch.from_numpy(rotations).float().to(device),
        torch.from_numpy(scales).float().to(device),
        torch.from_numpy(opacities).float().to(device),
        torch.from_numpy(shs).float().to(device),
    )


def _compute_stats(positions: torch.Tensor) -> SceneStats:
    pos_np = positions.detach().cpu().numpy()
    # Use percentiles to ignore outliers (e.g. floaters far away)
    bbox_min = np.percentile(pos_np, 15.0, axis=0)
    bbox_max = np.percentile(pos_np, 85.0, axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    radius = float(np.linalg.norm(bbox_max - bbox_min)) * 0.5
    scale = float(np.max(bbox_max - bbox_min))
    return SceneStats(bbox_min=bbox_min, bbox_max=bbox_max, center=center, radius=radius, scale=scale)


def _build_transform(stats: SceneStats, normalize: bool, target_radius: float) -> SceneTransform:
    if not normalize or stats.radius < 1e-6:
        return SceneTransform(
            world_from_scene=np.eye(4, dtype=np.float32),
            scene_from_world=np.eye(4, dtype=np.float32),
            scale_factor=1.0,
            translation=np.zeros(3, dtype=np.float32),
        )

    scale_factor = target_radius / stats.radius
    translation = -stats.center
    world_from_scene = np.eye(4, dtype=np.float32)
    world_from_scene[:3, :3] *= scale_factor
    world_from_scene[:3, 3] = translation * scale_factor

    scene_from_world = np.eye(4, dtype=np.float32)
    scene_from_world[:3, :3] *= 1.0 / scale_factor
    scene_from_world[:3, 3] = -translation
    return SceneTransform(
        world_from_scene=world_from_scene,
        scene_from_world=scene_from_world,
        scale_factor=scale_factor,
        translation=translation.astype(np.float32),
    )


def load_scene(
    ply_path: str,
    device: Optional[torch.device] = None,
    normalize: bool = True,
    target_radius: float = 1.5,
) -> Tuple[SceneData, SceneStats, SceneTransform]:
    """
    Load a Gaussian Splatting scene from a .ply file and center/scale it.

    Returns the raw tensors, simple scene statistics, and the transform applied.
    """
    path = Path(ply_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing scene PLY: {path}")

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    # Load using custom PLY loader
    positions, rotations, scales, opacities, shs = load_ply(str(path), device)
    colors_precomp = None

    stats = _compute_stats(positions)
    transform = _build_transform(stats, normalize=normalize, target_radius=target_radius)

    if normalize:
        # Apply the world_from_scene transform to center and scale points
        ones = torch.ones((positions.shape[0], 1), device=positions.device, dtype=positions.dtype)
        hom_pos = torch.cat([positions, ones], dim=1)
        tf = torch.from_numpy(transform.world_from_scene).to(device=device, dtype=positions.dtype)
        positions = (tf @ hom_pos.T).T[:, :3]
        scales = scales * transform.scale_factor
        stats = _compute_stats(positions)

    scene = SceneData(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        shs=shs,
        colors_precomp=colors_precomp,
        device=device,
    )
    return scene, stats, transform
