# Gaussian Splatting Cinematic Pipeline

This project builds a complete pipeline for loading Gaussian Splatting `.ply` scenes, planning cinematic camera trajectories, rendering high-quality fly-throughs, and exporting MP4 videos. It supports all provided scenes:

- `ConferenceHall.ply`
- `Museum.ply`
- `outdoor-drone.ply`
- `outdoor-street.ply`
- `Theater.ply`

The pipeline produces two video styles per scene:

1. **Cinematic Exploratory Tour (60–120 s)** – maximizes coverage with smooth motion.
2. **Object-Focused Tour (optional, 45–90 s)** – revisits interesting objects using either detection cues or color heuristics.

## Project Structure

```
src/
  explorer.py      # Scene ingestion, density grids, adaptive waypoint strategies
  path_planner.py  # Spline planner with collision avoidance and ease-in/out speed
  renderer.py      # Gaussian renderer integration + MP4 export
  detector.py      # Optional YOLO-based object boosts
  main.py          # CLI orchestrating the full pipeline
requirements.txt
```

## Dependencies

Base requirements (install via `pip install -r requirements.txt`):

- `numpy`, `scipy`, `plyfile`, `imageio[ffmpeg]`

Optional accelerators:

- **GPU Gaussian rendering** – [`gsplat`](https://github.com/nerfstudio-project/gsplat) + `torch`. Install with:
  ```
  pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
  pip install git+https://github.com/nerfstudio-project/gsplat.git
  ```
- **Object detection** – [`ultralytics`](https://github.com/ultralytics/ultralytics) for YOLOv8 models: `pip install ultralytics`.

The renderer automatically falls back to a CPU-only preview rasterizer when `gsplat` is unavailable, so you can still preview trajectories on any machine.

## Preparing the Scenes

1. Place the `.ply` Gaussian point clouds in a directory of your choice (e.g., `data/ConferenceHall.ply`).
2. The explorer automatically infers whether a scene is indoor/outdoor using bounding-box statistics, but you can override this via the CLI.

## Running the Pipeline

From the repository root:

```
python -m src.main ^
  --scenes data/ConferenceHall.ply data/Museum.ply ^
  --output-dir outputs ^
  --duration 90 ^
  --object-tour ^
  --backend auto
```

Command-line flags:

- `--scenes`: Space-separated list of `.ply` files (required).
- `--output-dir`: Target directory for frames/videos (default `outputs/`).
- `--duration`: Exploratory video duration in seconds (default 90).
- `--object-tour`: Enable the object-focused video (default off).
- `--object-duration`: Duration of the object tour (default 60 s).
- `--object-count`: Minimum number of objects to highlight (default 4).
- `--backend {auto,gsplat,lite}`: Rendering backend preference.
- `--coverage`: Multiplier adjusting how many coverage waypoints to sample.
- `--resolution`: Output resolution as `WIDTHxHEIGHT` (default `1920x1080`).
- `--scene-type-overrides`: JSON mapping like `{"Museum":"indoor"}`.

The CLI prints progress for rendering and the output MP4 paths:
- Exploratory: `outputs/<scene>/<scene>_exploratory.mp4`
- Object tour: `outputs/<scene>/<scene>_object_focus.mp4`

## Module Overview

- **`explorer.py`**
  - Loads Gaussian splats via `plyfile`, extracts positions, colors, scales, and opacity.
  - Builds KD-trees and voxelized density grids for collision checks.
  - Provides indoor/outdoor waypoint strategies plus heuristic object candidates.

- **`path_planner.py`**
  - Catmull-Rom spline interpolation + smoothstep re-parameterization for ease-in/out motion.
  - Collision avoidance by sampling local densities and pushing poses outward.
  - Generates orientation vectors (forward/up) and optional object-focused tours.

- **`renderer.py`**
  - Chooses between CUDA (`gsplat`) or CPU fallback renderers.
  - Applies poses, renders RGB frames, and exports MP4 videos via `imageio`.

- **`detector.py`**
  - Optional YOLOv8 wrapper; boosts object candidates if detections overlap projected 3D points.

- **`main.py`**
  - CLI pipeline that ties everything together, handles progress reporting, and organizes outputs.

## Customization Tips

- **Coverage**: Increase `--coverage` for larger outdoor scenes (e.g., `1.2` for `outdoor-drone.ply`).
- **Scene types**: Force indoor/outdoor handling via `--scene-type-overrides` when heuristics misclassify.
- **Collision margins**: Adjust `PlannerSettings.min_clearance` in `path_planner.py` for tighter/safe paths.
- **Object tours**: Without YOLO, the explorer still selects high-contrast Gaussians as points of interest.

## Output

Each run yields:

- A set of rendered frames streamed directly into MP4 files.
- Optional highlight videos focusing on 3+ interesting objects.

The resulting trajectories respect cinematic rules (dolly sweeps, arcs, crane motion) and adapt behavior based on scene scale (indoor vs. outdoor). Use the generated files as templates to fine-tune motion styles, swap rendering backends, or integrate into larger production pipelines.
