import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from .path_planner import CameraPose
from .scene_loader import SceneData, SceneStats


@dataclass
class RenderConfig:
    width: int = 1280
    height: int = 720
    fps: int = 24
    fov: float = 60.0
    background: tuple = (0.0, 0.0, 0.0)
    scale_modifier: float = 1.0
    near: Optional[float] = None
    far: Optional[float] = None


def _perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fov_y / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj


def _view_matrix(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation.T
    view[:3, 3] = -rotation.T @ position
    return view


def _sh_degree(shs: Optional[torch.Tensor]) -> int:
    if shs is None:
        return 0
    # Features are laid out as (degree + 1)^2 * 3
    degree = int(math.sqrt(shs.shape[1] // 3) - 1)
    return max(degree, 0)


def _get_raster_components():
    import importlib

    settings_candidates = [
        ("gsplat.render", "GSplatRasterizationSettings"),
        ("gsplat.render", "RasterizationSettings"),
        ("gsplat.render", "GaussianRasterizationSettings"),
        ("gsplat.cuda", "GSplatRasterizationSettings"),
        ("gsplat.cuda", "RasterizationSettings"),
        ("gsplat.cuda", "GaussianRasterizationSettings"),
        ("gsplat.cuda._C", "GSplatRasterizationSettings"),
        ("gsplat.cuda._C", "RasterizationSettings"),
        ("gsplat.cuda._C", "GaussianRasterizationSettings"),
    ]
    raster_candidates = [
        ("gsplat.render", "rasterization"),
        ("gsplat.cuda", "rasterization"),
        ("gsplat.cuda._C", "rasterization"),
        ("gsplat.cuda._C", "rasterize"),
    ]

    Settings = None
    rasterization = None

    for mod_name, attr in settings_candidates:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, attr):
                Settings = getattr(mod, attr)
                break
        except Exception:
            continue

    for mod_name, attr in raster_candidates:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, attr):
                rasterization = getattr(mod, attr)
                break
        except Exception:
            continue

    return Settings, rasterization


def _render_with_gsplat(
    scene: SceneData, pose: CameraPose, config: RenderConfig, stats: SceneStats
) -> np.ndarray:
    Settings, rasterization = _get_raster_components()
    if Settings is None or rasterization is None:
        raise ImportError("gsplat rendering components were not found.")

    device = scene.device
    aspect = config.width / config.height
    fov_y = math.radians(config.fov)
    fov_x = 2 * math.atan(math.tan(fov_y / 2) * aspect)

    near = config.near or max(0.05, stats.radius * 0.05)
    far = config.far or max(50.0, stats.radius * 6.0)
    view = torch.from_numpy(_view_matrix(pose.rotation, pose.position)).to(device)
    proj = torch.from_numpy(_perspective(fov_y, aspect, near, far)).to(device)
    bg = torch.tensor(config.background, device=device, dtype=torch.float32)

    colors = scene.colors_precomp
    if colors is None:
        if scene.shs is not None:
            # SH coefficient 0 maps to base color with a constant factor.
            colors = scene.shs[:, :3] * 0.28209479177387814
        else:
            colors = torch.ones_like(scene.positions, device=device)

    sh_degree = _sh_degree(scene.shs)
    out = rasterization(
        means3D=scene.positions,
        shs=scene.shs,
        colors_precomp=colors,
        opacities=scene.opacities,
        scales=scene.scales,
        rotations=scene.rotations,
        cov3D_precomp=None,
        raster_settings=Settings(
            image_height=config.height,
            image_width=config.width,
            tanfovx=float(math.tan(fov_x * 0.5)),
            tanfovy=float(math.tan(fov_y * 0.5)),
            bg=bg,
            scale_modifier=config.scale_modifier,
            viewmatrix=view,
            projmatrix=proj,
            sh_degree=sh_degree,
            camera_center=torch.from_numpy(pose.position).to(device),
        ),
    )
    image = out["render"] if isinstance(out, dict) and "render" in out else out
    if torch.is_tensor(image):
        image = image.permute(1, 2, 0).clamp(0.0, 1.0).detach().cpu().numpy()
    return image.astype(np.float32)


def _placeholder_frame(pose: CameraPose, config: RenderConfig, stats: SceneStats) -> np.ndarray:
    # Simple radial gradient placeholder to keep pipeline running without gsplat.
    h, w = config.height, config.width
    y, x = np.ogrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    dist = dist / (np.max(dist) + 1e-6)
    base = np.clip(1.0 - dist[..., None], 0.0, 1.0)
    tint = np.array(
        [
            0.2 + 0.6 * (pose.position[0] / max(stats.radius, 1e-3)),
            0.3 + 0.5 * (pose.position[1] / max(stats.radius, 1e-3)),
            0.4 + 0.4 * (pose.position[2] / max(stats.radius, 1e-3)),
        ],
        dtype=np.float32,
    )
    frame = np.clip(base * tint, 0.0, 1.0)
    return frame.astype(np.float32)


def render_video_frames(
    scene: SceneData,
    stats: SceneStats,
    camera_poses: Iterable[CameraPose],
    config: RenderConfig,
) -> List[np.ndarray]:
    poses = list(camera_poses)
    frames: List[np.ndarray] = []
    use_placeholder = False
    for pose in tqdm(poses, desc="Rendering", unit="frame"):
        if not use_placeholder:
            try:
                frame = _render_with_gsplat(scene, pose, config, stats)
            except Exception as exc:  # pragma: no cover - runtime guard
                logging.warning("gsplat rendering failed (%s); falling back to placeholder frames.", exc)
                use_placeholder = True
                frame = _placeholder_frame(pose, config, stats)
        else:
            frame = _placeholder_frame(pose, config, stats)
        frames.append(frame)
    return frames


def save_video(frames: Iterable[np.ndarray], output_path: Path, fps: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps, codec="h264", quality=8) as writer:
        for frame in frames:
            frame_uint8 = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
            writer.append_data(frame_uint8)
    return output_path
