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


def _render_with_gsplat(
    scene: SceneData, pose: CameraPose, config: RenderConfig, stats: SceneStats
) -> np.ndarray:
    try:
        from gsplat import rasterization
    except Exception as exc:
        raise ImportError("gsplat rendering components were not found.") from exc

    device = scene.device
    aspect = config.width / config.height
    fov_y = math.radians(config.fov)
    fov_x = 2 * math.atan(math.tan(fov_y / 2) * aspect)

    near = config.near or max(0.05, stats.radius * 0.05)
    far = config.far or max(50.0, stats.radius * 6.0)
    view = torch.from_numpy(_view_matrix(pose.rotation, pose.position)).to(device)

    # Intrinsics matrix from FOV.
    fx = (config.width * 0.5) / math.tan(fov_x * 0.5)
    fy = (config.height * 0.5) / math.tan(fov_y * 0.5)
    cx = config.width * 0.5
    cy = config.height * 0.5
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device, dtype=torch.float32).unsqueeze(0)

    colors = scene.colors_precomp
    if colors is None:
        if scene.shs is not None:
            colors = scene.shs[:, :3] * 0.28209479177387814
        else:
            colors = torch.ones_like(scene.positions, device=device)
    # If SHs are provided as flattened bands, reshape to [N, K, 3].
    if scene.shs is not None and scene.shs.dim() == 2 and scene.shs.shape[1] % 3 == 0:
        colors = scene.shs.view(scene.shs.shape[0], -1, 3)

    sh_degree = _sh_degree(scene.shs)

    # Modern gsplat.rendering.rasterization interface (batch size 1).
    viewmats = view.unsqueeze(0)
    bg = torch.tensor(config.background, device=device, dtype=torch.float32)

    out = rasterization(
        means=scene.positions,
        quats=scene.rotations,
        scales=scene.scales,
        opacities=scene.opacities.squeeze(-1) if scene.opacities.ndim == 2 else scene.opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=K,
        width=config.width,
        height=config.height,
        sh_degree=sh_degree,
        render_mode="RGB",
        backgrounds=bg.view(1, 1, 3),
        near_plane=near,
        far_plane=far,
        radius_clip=1e-4,
        packed=False,
    )

    # gsplat returns (B, H, W, 3) or (colors, alphas, info)
    if isinstance(out, tuple):
        image = out[0]
    elif isinstance(out, dict) and "rgb" in out:
        image = out["rgb"]
    else:
        image = out

    if torch.is_tensor(image):
        if image.ndim == 4:
            image = image[0]
        image = image.clamp(0.0, 1.0).detach().cpu().numpy()
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
