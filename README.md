# Gaussian Splatting Cinematic Pipeline

This project builds a complete pipeline for loading Gaussian Splatting `.ply` scenes, planning cinematic camera trajectories, rendering high-quality fly-throughs, and exporting MP4 videos. It supports all provided scenes:

- `ConferenceHall.ply`
- `Museum.ply`
- `outdoor-drone.ply`
- `outdoor-street.ply`
- `Theater.ply`

The pipeline produces a single cinematic exploratory tour per scene with smooth, forward-only motion that stays inside the splat volume.

## Project Structure

```
src/
  explorer.py      # Scene ingestion, density grids, and indoor/outdoor heuristics
  path_planner.py  # Forward-only planner with interior-safe smoothing
  renderer.py      # Gaussian renderer integration + MP4 export
  detector.py      # Legacy YOLO-based object boosts (not used by the CLI)
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

From the repository root, render one or more `.ply` scenes:

```
python -m src.main \
  --scenes data/ConferenceHall.ply data/Museum.ply \
  --output-dir outputs \
  --backend gsplat \
  --resolution 1280x720
```

Command-line flags:

- `--scenes`: Space-separated list of `.ply` files (required).
- `--output-dir`: Target directory for frames/videos (default `outputs/`).
- `--backend {auto,gsplat,lite}`: Rendering backend preference (CUDA `gsplat` encouraged).
- `--resolution`: Output resolution as `WIDTHxHEIGHT` (default `1280x720`).

The CLI prints progress for rendering and the output MP4 path:
- Exploratory: `outputs/<scene>/<scene>_exploratory.mp4`

## Module Overview

- **`explorer.py`**
  - Loads Gaussian splats via `plyfile`, extracts positions, colors, scales, and opacity.
  - Builds KD-trees and voxelized density grids for collision checks.
  - Provides indoor/outdoor heuristics for setting sensible camera height and bounds.

- **`path_planner.py`**
  - Extracts a dominant horizontal travel axis from the splat cloud and carves out a padded interior corridor.
  - Generates smooth, forward-only poses with lateral sway for cinematic parallax while staying within bounds.
  - Keeps a fixed, comfortable eye height to avoid ceiling/floor collisions.

- **`renderer.py`**
  - Chooses between CUDA (`gsplat`) or CPU fallback renderers.
  - Applies poses, renders RGB frames, and exports MP4 videos via `imageio`.

- **`main.py`**
  - CLI pipeline that ties everything together, handles progress reporting, and organizes outputs.

## Customization Tips

- **Renderer choice**: Use `--backend gsplat` on CUDA for the highest quality; fall back to `lite` on CPU-only setups.
- **Resolution**: Higher resolutions improve clarity but increase render time; adjust `--resolution` as needed.

## Output

Each run yields:

- A rendered MP4 fly-through for each input scene.

The resulting trajectories respect cinematic rules (dolly sweeps, arcs, crane motion) and adapt behavior based on scene scale (indoor vs. outdoor). Use the generated files as templates to fine-tune motion styles, swap rendering backends, or integrate into larger production pipelines.
