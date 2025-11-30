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


def _first_existing(mapping, keys, default=None):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _load_with_gsplat(ply_path: Path):
    try:
        from gsplat.io import load_ply
    except Exception:
        try:
            from gsplat.io.utils import load_ply  # type: ignore
        except Exception as exc:  # pragma: no cover - import is environment specific
            raise ImportError(
                "gsplat is required to load the Gaussian splat PLY. "
                "Install with `pip install gsplat` or from source."
            ) from exc
    return load_ply(str(ply_path))


def _fallback_load_ply(ply_path: Path):
    """Minimal PLY reader when gsplat's loader is unavailable."""
    logging.warning("Falling back to plain PLY parsing; install gsplat for optimal loading.")
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"].data
    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    opacities = vertex["opacity"] if "opacity" in vertex.dtype.names else np.ones(len(vertex))
    scales = np.stack(
        [
            vertex["scale_0"] if "scale_0" in vertex.dtype.names else np.ones(len(vertex)),
            vertex["scale_1"] if "scale_1" in vertex.dtype.names else np.ones(len(vertex)),
            vertex["scale_2"] if "scale_2" in vertex.dtype.names else np.ones(len(vertex)),
        ],
        axis=1,
    )
    rotations = np.stack(
        [
            vertex["rot_0"] if "rot_0" in vertex.dtype.names else np.ones(len(vertex)),
            vertex["rot_1"] if "rot_1" in vertex.dtype.names else np.zeros(len(vertex)),
            vertex["rot_2"] if "rot_2" in vertex.dtype.names else np.zeros(len(vertex)),
            vertex["rot_3"] if "rot_3" in vertex.dtype.names else np.zeros(len(vertex)),
        ],
        axis=1,
    )
    colors = None
    if "red" in vertex.dtype.names:
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1) / 255.0
    return {
        "positions": torch.from_numpy(positions).float(),
        "scales": torch.from_numpy(scales).float(),
        "rotations": torch.from_numpy(rotations).float(),
        "opacities": torch.from_numpy(opacities).float().unsqueeze(-1),
        "colors_precomp": torch.from_numpy(colors).float() if colors is not None else None,
        "shs": None,
    }


def _compute_stats(positions: torch.Tensor) -> SceneStats:
    pos_np = positions.detach().cpu().numpy()
    bbox_min = pos_np.min(axis=0)
    bbox_max = pos_np.max(axis=0)
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
    try:
        payload = _load_with_gsplat(path)
    except Exception:
        payload = _fallback_load_ply(path)

    positions = _first_existing(payload, ["positions", "xyz", "means3D", "means"])
    if positions is None:
        raise ValueError("PLY payload did not contain positions/xyz data.")
    scales = _first_existing(payload, ["scales", "scale"])
    rotations = _first_existing(payload, ["rotations", "rotation", "rot"])
    opacities = _first_existing(payload, ["opacities", "opacity"])

    def _to_tensor(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.float().to(device)
        return torch.from_numpy(np.asarray(value)).float().to(device)

    positions = _to_tensor(positions)
    scales = _to_tensor(scales)
    rotations = _to_tensor(rotations)
    opacities = _to_tensor(opacities)
    if scales is None:
        scales = torch.ones((positions.shape[0], 3), device=device, dtype=positions.dtype)
    if rotations is None:
        rot_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=positions.dtype)
        rotations = rot_id.repeat(positions.shape[0], 1)
    if opacities is None:
        opacities = torch.ones((positions.shape[0], 1), device=device, dtype=positions.dtype)
    if opacities.ndim == 1:
        opacities = opacities.unsqueeze(-1)

    shs = payload.get("shs") or payload.get("features") or payload.get("features_dc")
    colors_precomp = payload.get("colors_precomp")
    shs = _to_tensor(shs)
    colors_precomp = _to_tensor(colors_precomp)

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
        shs=shs if isinstance(shs, torch.Tensor) else None,
        colors_precomp=colors_precomp if isinstance(colors_precomp, torch.Tensor) else None,
        device=device,
    )
    return scene, stats, transform
