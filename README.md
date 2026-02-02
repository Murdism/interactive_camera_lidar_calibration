# interactive_camera_lidar_calibration

Python package scaffold for `sensorcal`.

## Install (editable)

```bash
python -m pip install -e .
```

## Install (pip)

```bash
python -m pip install sensorcal
```

## Quickstart

```bash
sensorcal recalibrate --use-sample
```

If you run `sensorcal recalibrate` with no arguments, it defaults to the sample data.
When installed via pip, sample data is bundled inside the package.

## Usage

Single image + pcd:

```bash
sensorcal recalibrate \
  --config path/to/calibrate.yaml \
  --image path/to/img_001.png \
  --pcd path/to/pc_001.pcd
```

Single image + pcd with parameters (no config):

```bash
sensorcal recalibrate \
  --image path/to/img_001.png \
  --pcd path/to/pc_001.pcd \
  --intrinsic-k 600 0 640 0 600 360 0 0 1 \
  --lidar-camera 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
```

Folder mode:

```bash
sensorcal recalibrate \
  --config path/to/calibrate.yaml \
  --image-folder path/to/images \
  --pcd-folder path/to/pcds
```

Sample data (3 images + 3 PCDs bundled in `samples/`): use the quickstart above.

## Input Structure

You can provide inputs in two ways:

1) Direct CLI arguments
- Single pair:
  - `--image path/to/img.png`
  - `--pcd path/to/cloud.pcd`
- Folder mode (paired by sorted filename order):
  - `--image-folder path/to/images`
  - `--pcd-folder path/to/pcds`

2) YAML config file (`--config path/to/calibrate.yaml`)

### Expected YAML structure

```yaml
transform:
  # 3x3 camera intrinsics in row-major order
  intrinsic_k: [fx, 0, cx, 0, fy, cy, 0, 0, 1]

  # 4x4 LiDAR-to-camera transform in row-major order
  lidar_camera: [r00, r01, r02, tx,
                 r10, r11, r12, ty,
                 r20, r21, r22, tz,
                 0,   0,   0,   1]

path:
  # Optional defaults for folder mode
  img_folder: /abs/or/relative/path/to/images
  pcd_folder: /abs/or/relative/path/to/pcds
```

### Notes

- If you pass `--intrinsic-k` or `--lidar-camera` on the CLI, those override the config file.
- If you pass both `--image/--pcd` and folder paths, the single pair wins.
- For folder mode, files are paired by **sorted filename order**, so keep names aligned.

## Python (pip installed)

```python
from sensorcal.app import SensorCalApp

app = SensorCalApp(
    config_path="path/to/calibrate.yaml",
    intrinsic_k=None,
    lidar_camera=None,
    image_path="path/to/img_001.png",
    pcd_path="path/to/pc_001.pcd",
    image_folder=None,
    pcd_folder=None,
    save_file="calibration_results.yaml",
)
app.process()
```

## Controls

- Single-window Tkinter app with live sliders, buttons, and dark mode toggle
- Sliders: tx/ty/tz (meters), roll/pitch/yaw (degrees), alpha, point size
- Density toggle: overlays a heatmap of point concentration (helps reveal clusters)
- Keyboard: A/D (X-/X+), W/S (Y-/Y+), Q/E (Z-/Z+)
- Rotate: J/L (roll-/+), I/K (pitch-/+), U/O (yaw-/+)
- Buttons: Prev, Next, Original, Save
- Save writes YAML plus a sibling `.txt` containing K and the transform.

## Python API

Use the `sensorcal.app.SensorCalApp` class shown above.
