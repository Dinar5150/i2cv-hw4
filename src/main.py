import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch

from .explorer import generate_exploration_waypoints
from .path_planner import smooth_camera_path
from .renderer import RenderConfig, render_video_frames, save_video
from .scene_loader import load_scene


def _parse_resolution(res: str) -> Tuple[int, int]:
    try:
        width, height = res.lower().split("x")
        return int(width), int(height)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Resolution must be WIDTHxHEIGHT, got {res}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cinematic fly-through generator for a single Gaussian Splatting scene.",
    )
    parser.add_argument("--scene", required=True, help="Path to the Gaussian splat .ply file.")
    parser.add_argument("--output", default=None, help="Path to the output MP4. Defaults to outputs/<scene>/tour.mp4")
    parser.add_argument(
        "--resolution",
        type=_parse_resolution,
        default=_parse_resolution("1920x1080"),
        help="Output resolution WIDTHxHEIGHT.",
    )
    parser.add_argument("--fps", type=int, default=24, help="Video frame rate.")
    parser.add_argument("--fov", type=float, default=60.0, help="Camera vertical field of view in degrees.")
    parser.add_argument("--speed", type=float, default=0.8, help="Nominal camera travel speed in scene units per second.")
    parser.add_argument("--device", default=None, help="Torch device to render on (e.g., cuda:0 or cpu).")
    parser.add_argument("--no-normalize", action="store_true", help="Disable scene centering/scaling.")
    parser.add_argument("--target-radius", type=float, default=1.5, help="Target radius after normalization.")
    parser.add_argument("--orbits", type=int, default=2, help="Number of wide orbits to seed the path.")
    parser.add_argument("--orbit-points", type=int, default=60, help="Samples per orbit.")
    parser.add_argument("--hold", type=float, default=0.6, help="Hold duration (seconds) at start/end.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    width, height = args.resolution
    device = torch.device(args.device) if args.device else None

    logging.info("Loading scene: %s", args.scene)
    scene, stats, _ = load_scene(
        args.scene,
        device=device,
        normalize=not args.no_normalize,
        target_radius=args.target_radius,
    )
    logging.info(
        "Scene stats | center: (%.3f, %.3f, %.3f) | radius: %.3f | scale: %.3f",
        stats.center[0],
        stats.center[1],
        stats.center[2],
        stats.radius,
        stats.scale,
    )

    logging.info("Generating exploration waypoints...")
    waypoints = generate_exploration_waypoints(
        stats,
        num_orbits=args.orbits,
        points_per_orbit=args.orbit_points,
    )

    logging.info("Smoothing camera path...")
    poses = smooth_camera_path(
        waypoints,
        fps=args.fps,
        nominal_speed=args.speed,
        fov=args.fov,
        hold_seconds=args.hold,
    )
    logging.info("Camera poses: %d frames", len(poses))

    config = RenderConfig(width=width, height=height, fps=args.fps, fov=args.fov)
    logging.info("Rendering frames at %dx%d...", width, height)
    frames = render_video_frames(scene, stats, poses, config)

    scene_name = Path(args.scene).stem
    output_path = Path(args.output) if args.output else Path("outputs") / scene_name / "panorama_tour.mp4"
    logging.info("Writing video to %s", output_path)
    save_video(frames, output_path, fps=args.fps)
    logging.info("Done.")


if __name__ == "__main__":
    main()
