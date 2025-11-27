"""
Renderer utilities that transform planned camera poses into RGB frames and
assemble them into cinematic MP4 videos.

Two rendering modes are supported:
- `gsplat`: Uses the CUDA accelerated `gsplat` Gaussian splatting renderer.
- `lite`: A CPU-based fallback that draws a subset of Gaussians into a pinhole
  camera with Gaussian weighting, useful for previews or when CUDA is
  unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .explorer import GaussianScene
from .path_planner import CameraPath, CameraPose

try:  # pragma: no cover - optional dependency
    import imageio
except ImportError:  # pragma: no cover - handled at runtime
    imageio = None


class GaussianRenderer:
    """High level API that hides backend selection and video export."""

    def __init__(
        self,
        backend: str = "auto",
        device: str | None = None,
        max_preview_points: int = 120_000,
    ) -> None:
        """
        Args:
            backend: "auto", "gsplat", or "lite".
            device: Preferred torch device for gsplat backend.
            max_preview_points: Number of Gaussians for lite backend.
        """

        self.backend_name = backend
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self.max_preview_points = max_preview_points
        self._backend_impl: _RendererBackend | None = None

    def prepare(self, scene: GaussianScene) -> None:
        self._backend_impl = self._select_backend(scene)

    def render_path(
        self,
        scene: GaussianScene,
        camera_path: CameraPath,
        resolution: Tuple[int, int] = (1920, 1080),
        progress_hook: callable | None = None,
    ) -> List[np.ndarray]:
        if self._backend_impl is None:
            self.prepare(scene)
        impl = self._backend_impl
        frames = []
        total = len(camera_path.poses)
        for idx, pose in enumerate(camera_path.poses):
            frame = impl.render(scene, pose, resolution)
            frames.append(frame)
            if progress_hook:
                progress_hook(idx + 1, total)
        return frames

    def save_video(
        self,
        frames: Sequence[np.ndarray],
        output_path: Path | str,
        fps: int,
    ) -> Path:
        if imageio is None:  # pragma: no cover - runtime guard.
            raise RuntimeError(
                "imageio is not installed. Install imageio[ffmpeg] to enable video export."
            )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=9) as writer:
            for frame in frames:
                writer.append_data(frame)
        return path

    # ------------------------------------------------------------------ #
    def _select_backend(self, scene: GaussianScene) -> "_RendererBackend":
        backend = self.backend_name
        if backend in ("auto", "gsplat"):
            try:
                return _GsplatBackend(scene, device=self.device)
            except Exception:  # pragma: no cover - falls back on preview renderer.
                if backend == "gsplat":
                    raise
        return _LiteGaussianBackend(max_points=self.max_preview_points)

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:  # pragma: no cover - torch optional
            return False


class _RendererBackend:
    """Interface each backend implements."""

    def render(
        self,
        scene: GaussianScene,
        pose: CameraPose,
        resolution: Tuple[int, int],
    ) -> np.ndarray:
        raise NotImplementedError


class _GsplatBackend(_RendererBackend):  # pragma: no cover - requires CUDA runtime.
    """Wraps the modern `gsplat` API using the rasterization helper."""

    def __init__(self, scene: GaussianScene, device: str = "cuda") -> None:
        try:
            import torch
            from gsplat.rendering import rasterization
        except Exception as exc:
            raise RuntimeError(
                "gsplat backend requested, but the gsplat package (and its dependencies) "
                "is not installed. Install via `pip install gsplat torch --extra-index-url "
                "https://download.pytorch.org/whl/cu121`."
            ) from exc

        self.torch = torch
        self.rasterize = rasterization
        self.device = torch.device(device)
        self._prepare_scene_tensors(scene)

    def _prepare_scene_tensors(self, scene: GaussianScene) -> None:
        torch = self.torch
        self.means = torch.from_numpy(scene.positions).to(self.device, dtype=torch.float32)
        self.scales = torch.from_numpy(scene.scales).to(self.device, dtype=torch.float32).clamp(
            min=1e-4
        )
        self.opacities = torch.from_numpy(scene.opacity).to(self.device, dtype=torch.float32).clamp(
            0.01, 1.0
        )
        self.colors = torch.from_numpy(scene.colors).to(self.device, dtype=torch.float32).clamp(
            0.0, 1.0
        )
        rotations = getattr(scene, "rotations", None)
        if rotations is not None:
            quats = torch.from_numpy(rotations).to(self.device, dtype=torch.float32)
        else:
            quats = torch.zeros((len(scene.positions), 4), dtype=torch.float32, device=self.device)
            quats[:, 3] = 1.0  # identity quaternion
        self.quats = quats
        self.background_color = torch.zeros(3, dtype=torch.float32, device=self.device)

    def render(
        self,
        scene: GaussianScene,
        pose: CameraPose,
        resolution: Tuple[int, int],
    ) -> np.ndarray:
        width, height = resolution
        view, K = self._camera_matrices(pose, width, height)
        render_colors, _, _ = self.rasterize(
            self.means,
            self.quats,
            self.scales,
            self.opacities,
            self.colors,
            view,
            K,
            width,
            height,
            backgrounds=self._background_tensor(height, width),
        )
        frame = (
            render_colors[0]
            .clamp(0.0, 1.0)
            .mul(255.0)
            .byte()
            .cpu()
            .numpy()
        )
        return frame

    def _camera_matrices(
        self, pose: CameraPose, width: int, height: int
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        torch = self.torch
        fx = fy = 0.5 * width / math.tan(math.radians(pose.fov * 0.5))
        cx = width / 2.0
        cy = height / 2.0
        position = torch.tensor(pose.position, device=self.device, dtype=torch.float32)
        forward = torch.tensor(pose.forward, device=self.device, dtype=torch.float32)
        up = torch.tensor(pose.up, device=self.device, dtype=torch.float32)
        right = torch.linalg.cross(forward, up)
        right = torch.nn.functional.normalize(right, dim=0)
        up_vec = torch.linalg.cross(right, forward)
        up_vec = torch.nn.functional.normalize(up_vec, dim=0)
        view = torch.eye(4, device=self.device, dtype=torch.float32)
        view[0, :3] = right
        view[1, :3] = up_vec
        view[2, :3] = -forward
        view[:3, 3] = -view[:3, :3] @ position
        K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            device=self.device,
            dtype=torch.float32,
        )
        return view, K

    def _background_tensor(self, height: int, width: int):
        torch = self.torch
        bg_color = self.background_color.view(1, 1, 3)
        bg = torch.ones(
            (height, width, 3),
            device=self.device,
            dtype=torch.float32,
        ).mul(bg_color)
        return bg.contiguous()


class _LiteGaussianBackend(_RendererBackend):
    """CPU fallback that draws a subset of Gaussians for previews."""

    def __init__(self, max_points: int = 120_000, seed: int = 19) -> None:
        self.max_points = max_points
        self.rng = np.random.default_rng(seed)

    def render(
        self,
        scene: GaussianScene,
        pose: CameraPose,
        resolution: Tuple[int, int],
    ) -> np.ndarray:
        width, height = resolution
        frame = np.zeros((height, width, 3), dtype=np.float32)
        depth = np.full((height, width), np.inf, dtype=np.float32)
        idx = self._subsample(len(scene.positions))
        points = scene.positions[idx]
        colors = scene.colors[idx]
        scales = scene.scales[idx]
        opacity = scene.opacity[idx]

        view = self._view_matrix(pose)
        pts_cam = (view[:3, :3] @ points.T + view[:3, 3:4]).T
        mask = pts_cam[:, 2] > 0.1
        pts_cam = pts_cam[mask]
        colors = colors[mask]
        scales = scales[mask]
        opacity = opacity[mask]

        fx = fy = 0.5 * width / math.tan(math.radians(pose.fov * 0.5))
        cx = width / 2.0
        cy = height / 2.0
        pixels = np.empty((len(pts_cam), 2), dtype=np.int32)
        pixels[:, 0] = (fx * (pts_cam[:, 0] / pts_cam[:, 2]) + cx).astype(np.int32)
        pixels[:, 1] = (fy * (pts_cam[:, 1] / pts_cam[:, 2]) + cy).astype(np.int32)
        valid = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        pts_cam = pts_cam[valid]
        colors = colors[valid]
        scales = scales[valid]
        opacity = opacity[valid]
        pixels = pixels[valid]
        sigma = np.clip(
            np.mean(scales, axis=1) / pts_cam[:, 2] * fx,
            0.5,
            14.0,
        )
        order = np.argsort(-pts_cam[:, 2])
        for idx in order:
            px, py = pixels[idx]
            s = sigma[idx]
            radius = max(1, int(2.5 * s))
            for dy in range(-radius, radius + 1):
                yy = py + dy
                if yy < 0 or yy >= height:
                    continue
                for dx in range(-radius, radius + 1):
                    xx = px + dx
                    if xx < 0 or xx >= width:
                        continue
                    dist2 = dx * dx + dy * dy
                    weight = math.exp(-0.5 * dist2 / (s * s + 1e-3))
                    alpha = float(np.clip(opacity[idx], 0.2, 1.0)) * weight
                    if alpha < 0.02:
                        continue
                    color = colors[idx]
                    depth_val = pts_cam[idx, 2]
                    if depth_val < depth[yy, xx]:
                        depth[yy, xx] = depth_val
                        frame[yy, xx] = (
                            frame[yy, xx] * (1 - alpha) + color * alpha
                        )
        frame = np.clip(frame, 0.0, 1.0)
        return (frame * 255.0).astype(np.uint8)

    def _subsample(self, count: int) -> np.ndarray:
        if count <= self.max_points:
            return np.arange(count)
        return self.rng.choice(count, size=self.max_points, replace=False)

    def _view_matrix(self, pose: CameraPose) -> np.ndarray:
        forward = pose.forward / np.linalg.norm(pose.forward)
        right = np.cross(forward, pose.up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -view[:3, :3] @ pose.position
        return view


__all__ = ["GaussianRenderer"]
