"""
CLI entry point for cinematic navigation across Gaussian Splatting scenes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .detector import ObjectDetector
from .explorer import GaussianScene, SceneExplorer, SceneType
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
        "--duration",
        type=float,
        default=90.0,
        help="Duration (seconds) of the exploratory video.",
    )
    parser.add_argument(
        "--object-duration",
        type=float,
        default=60.0,
        help="Duration (seconds) of the optional object-focused video.",
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
        default="1920x1080",
        help="Video resolution WIDTHxHEIGHT (e.g., 1920x1080).",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=1.0,
        help="Coverage multiplier for waypoint sampling.",
    )
    parser.add_argument(
        "--object-tour",
        action="store_true",
        help="Also produce an object-focused highlight video.",
    )
    parser.add_argument(
        "--object-count",
        type=int,
        default=4,
        help="Minimum number of objects to visit on highlight videos.",
    )
    parser.add_argument(
        "--detector-model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics model checkpoint for optional detection.",
    )
    parser.add_argument(
        "--scene-type-overrides",
        type=str,
        default=None,
        help="Optional JSON mapping from scene stem name to 'indoor'/'outdoor'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explorer = SceneExplorer()
    planner = CameraPathPlanner(explorer=explorer)
    renderer = GaussianRenderer(backend=args.backend)
    detector = ObjectDetector(model_name=args.detector_model) if args.object_tour else None
    scene_type_overrides = (
        json.loads(args.scene_type_overrides) if args.scene_type_overrides else {}
    )

    for scene_path in args.scenes:
        run_scene(
            Path(scene_path),
            explorer=explorer,
            planner=planner,
            renderer=renderer,
            detector=detector,
            output_dir=args.output_dir,
            coverage=args.coverage,
            duration=args.duration,
            object_duration=args.object_duration,
            object_count=args.object_count,
            resolution=args.resolution,
            declared_type=scene_type_overrides.get(Path(scene_path).stem),
        )


def run_scene(
    scene_path: Path,
    explorer: SceneExplorer,
    planner: CameraPathPlanner,
    renderer: GaussianRenderer,
    detector: ObjectDetector | None,
    output_dir: Path,
    coverage: float,
    duration: float,
    object_duration: float,
    object_count: int,
    resolution: Tuple[int, int],
    declared_type: str | None = None,
) -> None:
    print(f"\n=== Processing scene: {scene_path} ===")
    declared = SceneType(declared_type) if declared_type else None
    scene = explorer.load_scene(scene_path, declared_type=declared)
    waypoints = explorer.generate_waypoints(scene, coverage_ratio=coverage)
    print(f"Generated {len(waypoints)} coverage waypoints for {scene.scene_type.value} scene.")

    path = planner.plan_cinematic_path(scene, waypoints, duration=duration)
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

    if detector and object_count > 0:
        highlight_candidates = scene.object_candidates[:]
        if frames:
            preview_idx = np.linspace(0, len(frames) - 1, num=min(6, len(frames)), dtype=int)
            preview_frames = [frames[i] for i in preview_idx]
            preview_poses = [path.poses[i] for i in preview_idx]
            highlight_candidates = detector.select_objects_from_frames(
                scene,
                scene.object_candidates,
                preview_frames,
                preview_poses,
                resolution,
            )
        selected = highlight_candidates[: max(3, object_count)]
        highlight_path = planner.plan_object_focus_path(
            scene, selected, duration=object_duration
        )
        highlight_frames = renderer.render_path(
            scene,
            highlight_path,
            resolution=resolution,
            progress_hook=lambda done, total: print(
                f"\rRendering object tour {done}/{total}", end=""
            ),
        )
        print("\nRendering highlight video...")
        highlight_video = renderer.save_video(
            highlight_frames,
            base_dir / f"{scene.name}_{highlight_path.description}.mp4",
            fps=planner.settings.fps,
        )
        print(f"Object-focused video saved to {highlight_video}")


if __name__ == "__main__":
    main()
