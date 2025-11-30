# Gaussian Splatting Cinematic Pipeline

A single-scene cinematic renderer that loads a 3D Gaussian Splatting `.ply` with `gsplat`, plans a smooth exploration path, renders frames, and exports a polished fly-through MP4.

## Project Structure

```
src/
  scene_loader.py   # gsplat-based .ply loading, centering, scaling, stats
  explorer.py       # coarse exploration waypoints (orbits + dolly sweep)
  path_planner.py   # spline smoothing + look-at orientations
  renderer.py       # gsplat rasterization + MP4 assembly
  main.py           # CLI orchestration
requirements.txt
```

## Install

```
pip install -r requirements.txt
```

Key packages: `gsplat` + `torch` for rendering, `numpy`, `scipy`, `plyfile`, `imageio[ffmpeg]`, `tqdm`.

## Usage

Render one scene:

```
python -m src.main --scene data/scene.ply --resolution 1920x1080 --fps 24
```

Flags:

- `--scene`: Path to the `.ply` Gaussian model (required).
- `--output`: MP4 path (default `outputs/<scene>/panorama_tour.mp4`).
- `--resolution WIDTHxHEIGHT`: Output resolution (default `1920x1080`).
- `--fps`: Frame rate (default `24`).
- `--fov`: Vertical FOV in degrees (default `60`).
- `--speed`: Nominal camera speed in scene units/sec (default `0.8`).
- `--device`: Torch device (e.g., `cuda:0` or `cpu`).
- `--no-normalize`: Keep the original coordinates/scale (normalization on by default).

## Pipeline

1. **Load** – `scene_loader.load_scene` ingests the PLY with `gsplat`, recenters/scales, and reports stats.
2. **Explore** – `explorer.generate_exploration_waypoints` mixes wide orbits with a central dolly sweep.
3. **Smooth** – `path_planner.smooth_camera_path` splines waypoints, adds ease-in/out holds, and computes look-at rotations.
4. **Render** – `renderer.render_video_frames` rasterizes each pose with `gsplat` (placeholder fallback if unavailable).
5. **Export** – `renderer.save_video` writes the MP4 via `imageio`/ffmpeg.

## Notes

- Default normalization scales the scene to ~1.5 unit radius and centers it for stable camera speeds; disable with `--no-normalize` if you prefer original coordinates.
- If `gsplat` is missing, placeholder frames are produced so you can still validate the trajectory end-to-end.
