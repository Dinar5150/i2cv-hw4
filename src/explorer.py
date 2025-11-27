"""
Scene exploration utilities for Gaussian Splatting point clouds.

This module is responsible for:
- Loading .ply Gaussian splat scenes and extracting useful metadata.
- Building lightweight spatial acceleration structures (KD-tree, density grid).
- Providing adaptive exploration strategies (indoor vs. outdoor) that return
  scene covering waypoints for downstream path planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

try:
    from plyfile import PlyData
except ImportError as exc:  # pragma: no cover - handled at runtime.
    PlyData = None


class SceneType(str, Enum):
    """Simple enum for handling indoor/outdoor heuristics."""

    INDOOR = "indoor"
    OUTDOOR = "outdoor"


@dataclass
class ObjectCandidate:
    """Represents an interesting object anchor within the scene."""

    name: str
    position: np.ndarray
    score: float


@dataclass
class GaussianScene:
    """Container for all raw and derived scene data."""

    name: str
    ply_path: Path
    positions: np.ndarray
    colors: np.ndarray
    scales: np.ndarray
    opacity: np.ndarray
    scene_type: SceneType
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    kd_tree: cKDTree = field(repr=False)
    density_grid: np.ndarray = field(repr=False)
    grid_origin: np.ndarray = field(repr=False)
    grid_spacing: np.ndarray = field(repr=False)
    object_candidates: List[ObjectCandidate] = field(default_factory=list)
    rotations: np.ndarray | None = None

    @property
    def diagonal_length(self) -> float:
        return float(np.linalg.norm(self.bounds_max - self.bounds_min))


def _safe_import_error(module_name: str) -> RuntimeError:  # pragma: no cover
    """Raise a friendly error when optional dependency is missing."""

    return RuntimeError(
        f"{module_name} is required but not installed. "
        f"Install the dependencies listed in requirements.txt."
    )


class ExplorationStrategy:
    """Abstract base for exploration heuristics."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def generate_waypoints(
        self, scene: GaussianScene, num_waypoints: int
    ) -> np.ndarray:
        raise NotImplementedError


class IndoorExplorationStrategy(ExplorationStrategy):
    """Tighter motion with more turns and short hops."""

    def generate_waypoints(
        self, scene: GaussianScene, num_waypoints: int
    ) -> np.ndarray:
        subset = self._sample_subset(scene.positions, fraction=0.35)
        seeds = self._farthest_point_sampling(subset, num_waypoints)
        heights = scene.bounds_min[1] + 0.1 * (scene.bounds_max[1] - scene.bounds_min[1])
        seeds[:, 1] = np.clip(
            seeds[:, 1],
            scene.bounds_min[1] + 0.1,
            heights,
        )
        return seeds

    def _sample_subset(self, points: np.ndarray, fraction: float) -> np.ndarray:
        count = max(2000, int(len(points) * fraction))
        idx = self.rng.choice(len(points), size=min(count, len(points)), replace=False)
        return points[idx]

    def _farthest_point_sampling(self, points: np.ndarray, k: int) -> np.ndarray:
        if len(points) == 0:
            raise ValueError("Cannot sample waypoints from empty scene.")
        chosen = []
        idx = int(self.rng.integers(len(points)))
        chosen.append(points[idx])
        distances = np.linalg.norm(points - points[idx], axis=1)
        for _ in range(1, k):
            idx = int(np.argmax(distances))
            chosen.append(points[idx])
            new_dist = np.linalg.norm(points - points[idx], axis=1)
            distances = np.minimum(distances, new_dist)
        return np.asarray(chosen)


class OutdoorExplorationStrategy(ExplorationStrategy):
    """Creates sweeping arcs and higher altitude samples."""

    def generate_waypoints(
        self, scene: GaussianScene, num_waypoints: int
    ) -> np.ndarray:
        bounds_min, bounds_max = scene.bounds_min, scene.bounds_max
        size = bounds_max - bounds_min
        base_center = (bounds_min + bounds_max) / 2.0
        arc_radius = 0.35 * np.linalg.norm(size[[0, 2]])
        angles = np.linspace(0, 2 * np.pi, num_waypoints, endpoint=False)
        waypoints = []
        for idx, angle in enumerate(angles):
            radius_scale = 0.8 + 0.2 * np.sin(angle * 2.0)
            pos = base_center.copy()
            pos[0] += arc_radius * radius_scale * np.cos(angle)
            pos[2] += arc_radius * radius_scale * np.sin(angle)
            height_bias = 0.3 + 0.6 * (idx / max(1, num_waypoints - 1))
            pos[1] = bounds_min[1] + height_bias * size[1]
            waypoints.append(pos)
        return np.asarray(waypoints)


class SceneExplorer:
    """High level API to ingest .ply scenes and suggest coverage waypoints."""

    def __init__(
        self,
        grid_resolution: int = 64,
        safety_margin: float = 0.5,
        rng_seed: int = 7,
    ) -> None:
        self.grid_resolution = grid_resolution
        self.safety_margin = safety_margin
        self.rng = np.random.default_rng(rng_seed)

    def load_scene(
        self, ply_path: Path | str, declared_type: Optional[SceneType] = None
    ) -> GaussianScene:
        path = Path(ply_path)
        if not path.exists():
            raise FileNotFoundError(path)
        positions, colors, scales, opacity, rotations = self._parse_ply(path)
        bounds_min = positions.min(axis=0)
        bounds_max = positions.max(axis=0)
        scene_type = declared_type or self._infer_scene_type(bounds_min, bounds_max)
        kd_tree = cKDTree(positions)
        density_grid, origin, spacing = self._build_density_field(positions)
        candidates = self._extract_object_candidates(positions, colors, opacity)
        return GaussianScene(
            name=path.stem,
            ply_path=path,
            positions=positions,
            colors=colors,
            scales=scales,
            opacity=opacity,
            rotations=rotations,
            scene_type=scene_type,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            kd_tree=kd_tree,
            density_grid=density_grid,
            grid_origin=origin,
            grid_spacing=spacing,
            object_candidates=candidates,
        )

    def generate_waypoints(
        self,
        scene: GaussianScene,
        coverage_ratio: float = 1.0,
        min_waypoints: int = 14,
        max_waypoints: int = 36,
    ) -> np.ndarray:
        coverage_ratio = float(np.clip(coverage_ratio, 0.2, 1.5))
        count = int(np.clip(coverage_ratio * 20, min_waypoints, max_waypoints))
        strategy: ExplorationStrategy
        if scene.scene_type == SceneType.INDOOR:
            strategy = IndoorExplorationStrategy(self.rng)
        else:
            strategy = OutdoorExplorationStrategy(self.rng)
        return strategy.generate_waypoints(scene, count)

    def estimate_clearance(self, scene: GaussianScene, point: np.ndarray) -> float:
        """Query KD-tree for minimal distance to nearest Gaussian center."""

        distance, _ = scene.kd_tree.query(point, k=1)
        return float(distance)

    def sample_density(self, scene: GaussianScene, point: np.ndarray) -> float:
        """Sample occupancy-like density from the voxel grid."""

        rel = (point - scene.grid_origin) / scene.grid_spacing
        idx = np.floor(rel).astype(int)
        if np.any(idx < 0) or np.any(idx + 1 >= self.grid_resolution):
            return 0.0
        frac = rel - idx
        corners = scene.density_grid[
            idx[0] : idx[0] + 2, idx[1] : idx[1] + 2, idx[2] : idx[2] + 2
        ]
        weights = self._trilinear_weights(frac)
        return float(np.sum(corners * weights))

    def _trilinear_weights(self, frac: np.ndarray) -> np.ndarray:
        a, b, c = frac
        weights = np.array(
            [
                (1 - a) * (1 - b) * (1 - c),
                (1 - a) * (1 - b) * c,
                (1 - a) * b * (1 - c),
                (1 - a) * b * c,
                a * (1 - b) * (1 - c),
                a * (1 - b) * c,
                a * b * (1 - c),
                a * b * c,
            ]
        )
        return weights.reshape(2, 2, 2)

    def _parse_ply(
        self, path: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        if PlyData is None:  # pragma: no cover - runtime dependency guard.
            raise _safe_import_error("plyfile")
        ply = PlyData.read(str(path))
        vertex = ply["vertex"].data
        if "packed_position" in vertex.dtype.names:
            return self._decode_supersplat(ply, vertex)
        positions = self._extract_positions(vertex)
        colors = self._extract_colors(vertex)
        scales = self._extract_scales(vertex)
        opacity = (
            vertex["opacity"].astype(np.float32)
            if "opacity" in vertex.dtype.names
            else np.ones(len(vertex), dtype=np.float32)
        )
        rotations = self._extract_rotations(vertex)
        return positions, colors, scales, opacity, rotations

    def _decode_supersplat(
        self, ply: "PlyData", vertex: np.ndarray, chunk_size: int = 256
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        chunks = ply["chunk"].data
        chunk_ids = np.arange(len(vertex), dtype=np.int64) // chunk_size
        chunk_ids = np.clip(chunk_ids, 0, len(chunks) - 1)

        mins = np.column_stack(
            (chunks["min_x"], chunks["min_y"], chunks["min_z"])
        ).astype(np.float32)
        maxs = np.column_stack(
            (chunks["max_x"], chunks["max_y"], chunks["max_z"])
        ).astype(np.float32)
        positions = self._decode_triplet(vertex["packed_position"])
        positions = mins[chunk_ids] + (maxs - mins)[chunk_ids] * positions

        log_min = np.column_stack(
            (chunks["min_scale_x"], chunks["min_scale_y"], chunks["min_scale_z"])
        ).astype(np.float32)
        log_max = np.column_stack(
            (chunks["max_scale_x"], chunks["max_scale_y"], chunks["max_scale_z"])
        ).astype(np.float32)
        scales_norm = self._decode_triplet(vertex["packed_scale"])
        log_scales = log_min[chunk_ids] + (log_max - log_min)[chunk_ids] * scales_norm
        scales = np.exp(log_scales)

        color_min = np.column_stack(
            (chunks["min_r"], chunks["min_g"], chunks["min_b"])
        ).astype(np.float32)
        color_max = np.column_stack(
            (chunks["max_r"], chunks["max_g"], chunks["max_b"])
        ).astype(np.float32)
        color_norm = self._decode_triplet(vertex["packed_color"])
        colors = color_min[chunk_ids] + (color_max - color_min)[chunk_ids] * color_norm
        colors = np.clip(colors, 0.0, 1.0)

        rotations = self._decode_rotations(vertex["packed_rotation"])
        opacity = np.ones(len(vertex), dtype=np.float32)
        return (
            positions.astype(np.float32),
            colors,
            scales.astype(np.float32),
            opacity,
            rotations,
        )

    def _decode_triplet(self, packed: np.ndarray) -> np.ndarray:
        vals = packed.astype(np.uint32)
        a = (vals >> 22) & 0x3FF
        b = (vals >> 12) & 0x3FF
        c = (vals >> 2) & 0x3FF
        arr = np.stack((a, b, c), axis=1).astype(np.float32) / 1023.0
        return arr

    def _decode_rotations(self, packed: np.ndarray) -> np.ndarray:
        vals = packed.astype(np.uint32)
        comp0 = (vals >> 21) & 0x7FF
        comp1 = (vals >> 11) & 0x3FF
        comp2 = (vals >> 1) & 0x3FF
        sign = np.where((vals & 0x1) > 0, -1.0, 1.0)
        x = comp0.astype(np.float32) / 2047.0 * 2.0 - 1.0
        y = comp1.astype(np.float32) / 1023.0 * 2.0 - 1.0
        z = comp2.astype(np.float32) / 1023.0 * 2.0 - 1.0
        mag = x * x + y * y + z * z
        w = np.sqrt(np.maximum(0.0, 1.0 - mag)) * sign.astype(np.float32)
        quats = np.stack((x, y, z, w), axis=1)
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        return (quats / norms).astype(np.float32)

    def _extract_positions(self, vertex: np.ndarray) -> np.ndarray:
        names = vertex.dtype.names
        candidate_sets = [
            ("x", "y", "z"),
            ("pos_x", "pos_y", "pos_z"),
            ("position_x", "position_y", "position_z"),
            ("center_x", "center_y", "center_z"),
            ("mean_x", "mean_y", "mean_z"),
        ]
        for triplet in candidate_sets:
            if all(name in names for name in triplet):
                return np.column_stack([vertex[name] for name in triplet]).astype(np.float32)
        axis_options = {
            "x": ["x", "pos_x", "posx", "position_x", "positionx", "center_x", "centerx"],
            "y": ["y", "pos_y", "posy", "position_y", "positiony", "center_y", "centery"],
            "z": ["z", "pos_z", "posz", "position_z", "positionz", "center_z", "centerz"],
        }
        axes = {}
        for axis, options in axis_options.items():
            for option in options:
                if option in names:
                    axes[axis] = vertex[option]
                    break
        if len(axes) == 3:
            return np.column_stack([axes["x"], axes["y"], axes["z"]]).astype(np.float32)
        vector_props = [
            "packed_position",
            "position",
            "positions",
            "xyz",
        ]
        for prop in vector_props:
            if prop in names:
                arr = self._unpack_vector_property(vertex[prop])
                if arr.shape[1] >= 3:
                    return arr[:, :3].astype(np.float32)
        raise ValueError(
            "Unable to locate XYZ coordinates in PLY vertex data. "
            f"Found properties: {names}"
        )

    def _extract_colors(self, vertex: np.ndarray) -> np.ndarray:
        names = vertex.dtype.names
        if all(col in names for col in ("f_dc_0", "f_dc_1", "f_dc_2")):
            colors = np.column_stack(
                (vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"])
            ).astype(np.float32)
            colors = 1 / (1 + np.exp(-colors))  # sigmoid decode
            colors = np.clip(colors, 0.0, 1.0)
        elif all(col in names for col in ("red", "green", "blue")):
            colors = (
                np.column_stack((vertex["red"], vertex["green"], vertex["blue"]))
                / 255.0
            ).astype(np.float32)
        elif "packed_color" in names:
            arr = self._unpack_vector_property(vertex["packed_color"])
            colors = np.clip(arr[:, :3], 0.0, 1.0).astype(np.float32)
        else:
            colors = np.ones((len(vertex), 3), dtype=np.float32) * 0.5
        return colors

    def _extract_scales(self, vertex: np.ndarray) -> np.ndarray:
        names = vertex.dtype.names
        scale_cols = [f"scale_{axis}" for axis in range(3)]
        if all(col in names for col in scale_cols):
            scales = np.column_stack([vertex[col] for col in scale_cols]).astype(
                np.float32
            )
        elif "packed_scale" in names:
            arr = self._unpack_vector_property(vertex["packed_scale"])
            scales = arr[:, :3].astype(np.float32)
        else:
            scales = np.full((len(vertex), 3), 0.02, dtype=np.float32)
        return scales

    def _extract_rotations(self, vertex: np.ndarray) -> np.ndarray | None:
        names = vertex.dtype.names
        candidate_sets = [
            tuple(f"rotation_{i}" for i in range(4)),
            tuple(f"rot_{i}" for i in range(4)),
            ("rot_x", "rot_y", "rot_z", "rot_w"),
            ("qx", "qy", "qz", "qw"),
        ]
        for cols in candidate_sets:
            if all(col in names for col in cols):
                quats = np.column_stack([vertex[col] for col in cols]).astype(np.float32)
                norms = np.linalg.norm(quats, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-6)
                return quats / norms
        return None

    def _unpack_vector_property(self, data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 1:
            # Common packed representation: void bytes containing f32 triplets.
            if arr.dtype.kind == "V":
                item_bytes = arr.dtype.itemsize
                if item_bytes % 4 == 0:
                    count = item_bytes // 4
                    return arr.view(np.float32).reshape(arr.shape[0], count)
                if item_bytes % 8 == 0:
                    count = item_bytes // 8
                    return arr.view(np.float64).reshape(arr.shape[0], count)
            if arr.dtype.kind == "O":
                stacked = np.vstack([np.asarray(item).reshape(-1) for item in arr])
                return stacked
            # Attempt to interpret as array-like objects.
            sample = arr[0]
            if np.isscalar(sample):
                return arr[:, None]
            try:
                stacked = np.vstack([np.asarray(item) for item in arr])
                return stacked
            except Exception as exc:  # pragma: no cover - fallback
                raise ValueError(f"Unable to unpack vector property with dtype {arr.dtype}") from exc
        raise ValueError(f"Unsupported vector property dimensions: {arr.shape}")

    def _infer_scene_type(
        self, bounds_min: np.ndarray, bounds_max: np.ndarray
    ) -> SceneType:
        size = bounds_max - bounds_min
        horizontal = np.linalg.norm(size[[0, 2]])
        vertical = size[1]
        if horizontal > 30.0 or vertical > 8.0:
            return SceneType.OUTDOOR
        return SceneType.INDOOR

    def _build_density_field(
        self, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bounds_min = positions.min(axis=0)
        bounds_max = positions.max(axis=0)
        grid = np.zeros(
            (self.grid_resolution,) * 3,
            dtype=np.float32,
        )
        grid_spacing = (bounds_max - bounds_min) / self.grid_resolution
        grid_spacing[grid_spacing == 0] = 1e-3
        normed = (positions - bounds_min) / grid_spacing
        idx = np.clip(normed.astype(int), 0, self.grid_resolution - 1)
        for i, j, k in idx:
            grid[i, j, k] += 1.0
        grid = grid / np.max(grid)
        return grid, bounds_min, grid_spacing

    def _extract_object_candidates(
        self, positions: np.ndarray, colors: np.ndarray, opacity: np.ndarray, top_k: int = 8
    ) -> List[ObjectCandidate]:
        luminance = 0.2126 * colors[:, 0] + 0.7152 * colors[:, 1] + 0.0722 * colors[:, 2]
        saturation = np.max(colors, axis=1) - np.min(colors, axis=1)
        score = 0.6 * saturation + 0.4 * luminance
        score *= opacity
        idx_sorted = np.argsort(score)[::-1]
        selected = []
        used_idx: List[int] = []
        for idx in idx_sorted:
            pos = positions[idx]
            if any(np.linalg.norm(pos - positions[used]) < 0.5 for used in used_idx):
                continue
            candidate = ObjectCandidate(
                name=f"feature_{len(selected)+1}",
                position=pos,
                score=float(score[idx]),
            )
            selected.append(candidate)
            used_idx.append(idx)
            if len(selected) >= top_k:
                break
        return selected


__all__ = [
    "GaussianScene",
    "SceneExplorer",
    "SceneType",
    "ObjectCandidate",
]
