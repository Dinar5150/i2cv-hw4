"""
CLI entry point for cinematic navigation across Gaussian Splatting scenes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from .explorer import SceneExplorer
from .path_planner import CameraPathPlanner
from .renderer import GaussianRenderer


def parse_resolution(value: str) -> Tuple[int, int]:
    width, height = value.lower().split("x")
    return int(width), int(height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaussian Splatting cinematic pipeline.")
    parser.add_argument(
        "--scenes",
        nargs="+",
        required=True,
        help="List of .ply files to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where frames and mp4 videos will be written.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "gsplat", "lite"],
        default="auto",
        help="Rendering backend preference.",
    )
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default="1280x720",
        help="Video resolution WIDTHxHEIGHT (e.g., 1920x1080).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explorer = SceneExplorer()
    planner = CameraPathPlanner()
    renderer = GaussianRenderer(backend=args.backend)

    for scene_path in args.scenes:
        run_scene(
            Path(scene_path),
            explorer=explorer,
            planner=planner,
            renderer=renderer,
            output_dir=args.output_dir,
            resolution=args.resolution,
        )


def run_scene(
    scene_path: Path,
    explorer: SceneExplorer,
    planner: CameraPathPlanner,
    renderer: GaussianRenderer,
    output_dir: Path,
    resolution: Tuple[int, int],
) -> None:
    print(f"\n=== Processing scene: {scene_path} ===")
    scene = explorer.load_scene(scene_path)
    path = planner.plan_cinematic_path(scene)
    print(
        f"Prepared forward-only cinematic path with {len(path.poses)} poses "
        f"inside the {scene.scene_type.value} scene bounds."
    )
    base_dir = output_dir / scene.name
    base_dir.mkdir(parents=True, exist_ok=True)

    frames = renderer.render_path(
        scene,
        path,
        resolution=resolution,
        progress_hook=lambda done, total: print(
            f"\rRendering exploratory tour {done}/{total}", end=""
        ),
    )
    print("\nRendering completed. Writing mp4...")
    video_path = renderer.save_video(
        frames,
        base_dir / f"{scene.name}_{path.description}.mp4",
        fps=planner.settings.fps,
    )
    print(f"Exploratory video saved to {video_path}")


if __name__ == "__main__":
    main()
