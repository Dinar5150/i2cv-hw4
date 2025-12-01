# Gaussian Splatting Cinematic Pipeline

A single-scene cinematic renderer that loads a 3D Gaussian Splatting `.ply` with `gsplat`, plans a smooth exploration path, renders frames, and exports a polished fly-through MP4. It also supports YOLO-based object detection overlay.

## Prerequisites: PLY File Format

**Important:** The `.ply` files provided in standard assignments or datasets often use compressed or custom formats. This pipeline requires **uncompressed** PLY files compatible with `gsplat`.

To prepare your data:
1.  Open your `.ply` file in [SuperSplat](https://superspl.at/editor).
2.  Import the file.
3.  Export it as an **uncompressed .ply**.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Dinar5150/i2cv-hw4.git
    cd i2cv-hw4
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    pip install torch   # CUDA version suitable for your system!
    pip install gsplat  # Required for rasterization
    ```

## Running on Kaggle

For convenience, you can run this pipeline on Kaggle using the following script. This assumes you have uploaded your uncompressed PLY files to a dataset.

```python
%cd /kaggle/working
!git clone https://github.com/Dinar5150/i2cv-hw4.git
%cd /kaggle/working/i2cv-hw4

%pip install -r requirements.txt
%pip install gsplat
# Copy your uncompressed PLY files to the workspace
!cp -r /kaggle/input/uncompressed-ply-files/ply/* /kaggle/working/i2cv-hw4/

# Run the renderer with detection enabled
!python -m src.main --scene Museume.ply --resolution 1280x720 --fps 24 --device cuda --detect --duration 90

!python -m src.main --scene outdoor-street.ply --resolution 1280x720 --fps 24 --device cuda --detect --duration 90
```

## Local Usage

To render a scene locally:

```bash
python -m src.main --scene path/to/your.ply --resolution 1280x720 --detect
```

### Command Line Arguments

-   `--scene`: Path to the `.ply` Gaussian model (required).
-   `--output`: Path to the output MP4 file. Defaults to `outputs/<scene>/panorama_tour.mp4`.
-   `--resolution`: Output resolution (e.g., `1920x1080`). Default: `1920x1080`.
-   `--fps`: Frame rate. Default: `24`.
-   `--duration`: **Force total video duration** in seconds. Overrides speed settings.
-   `--detect`: **Enable YOLO object detection**. Overlays bounding boxes on the rendered video.
-   `--device`: Torch device (e.g., `cuda:0` or `cpu`).
-   `--speed`: Nominal camera speed in scene units/sec (ignored if `--duration` is set).
-   `--no-normalize`: Disable scene centering and scaling.

P.s. I haven't succeeded in running it locally due to complex CUDA dependencies.

## Project Structure

```
src/
  scene_loader.py   # gsplat-based .ply loading, centering, scaling, stats
  explorer.py       # Generates safe, collision-aware flight paths
  path_planner.py   # Spline smoothing, collision avoidance, and orientation logic
  renderer.py       # gsplat rasterization + MP4 assembly
  detector.py       # YOLO object detection wrapper
  main.py           # CLI orchestration
requirements.txt
```

## Pipeline Details

1.  **Load**: `scene_loader.py` loads the PLY, converts attributes for `gsplat`, and computes robust scene statistics (ignoring outliers).
2.  **Plan**:
    *   `explorer.py` generates a Lissajous-style path within a "safe box" (85% of scene bounds).
    *   `path_planner.py` interpolates this path using cubic splines. It applies **collision avoidance** by repelling the camera from scene points and smooths the result.
3.  **Render**: `renderer.py` rasterizes the scene from the calculated poses.
4.  **Detect**: If enabled, `detector.py` runs YOLOv11 on the rendered frames to label objects.
5.  **Export**: Frames are compiled into an MP4 video.
