"""SensorCal app: interactive LiDAR-to-camera calibration tool (Tkinter GUI)."""

from __future__ import annotations

import argparse
import importlib.resources as resources
import os
from datetime import datetime
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import yaml
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk


def _sample_path(*parts: str) -> str:
    base = resources.files("sensorcal") / "samples"
    return str(base.joinpath(*parts))


SAMPLE_IMAGE_FOLDER = _sample_path("images")
SAMPLE_PCD_FOLDER = _sample_path("pcds")
SAMPLE_INTRINSIC_K = [554.26, 0.0, 960.0, 0.0, 554.26, 540.0, 0.0, 0.0, 1.0]
SAMPLE_LIDAR_CAMERA = [
    0.5,
    -0.866,
    -0.0,
    -0.825,
    -0.0,
    -0.0,
    -1.0,
    -0.6,
    0.866,
    0.5,
    -0.0,
    -1.429,
    0.0,
    0.0,
    0.0,
    1.0,
]
MAX_TRANSLATION = 2.0
MAX_ROTATION_DEG = 30.0


class SensorCalApp:
    """Interactive calibration app with Tkinter GUI.

    Args:
        config_path: Optional YAML config path containing intrinsics/extrinsics.
        intrinsic_k: Optional 3x3 intrinsics (flat list of 9 values, row-major).
        lidar_camera: Optional 4x4 extrinsics (flat list of 16 values, row-major).
        image_path: Optional single image path.
        pcd_path: Optional single PCD path.
        image_folder: Optional folder containing images.
        pcd_folder: Optional folder containing PCDs.
        save_file: Output YAML path for saved transforms (TXT is created alongside).
        point_size: Rendered point size in pixels.
        point_color: Fallback RGB color when depth coloring is disabled (currently unused).
        depth_colormap: If True, color points by depth (always on in UI).

    Notes:
        - Provide either (image_path + pcd_path) or (image_folder + pcd_folder).
        - When using folders, files are paired by sorted filename order.
        - Calling `process()` starts the GUI event loop (blocks until window closes).
    """

    def __init__(
        self,
        *,
        config_path: Optional[str],
        intrinsic_k: Optional[Iterable[float]],
        lidar_camera: Optional[Iterable[float]],
        image_path: Optional[str],
        pcd_path: Optional[str],
        image_folder: Optional[str],
        pcd_folder: Optional[str],
        save_file: str,
        point_size: int = 2,
        point_color: Tuple[int, int, int] = (0, 255, 0),
        depth_colormap: bool = True,
    ) -> None:
        self.config = self._load_config(config_path) if config_path else {}

        self.img_folder = image_folder or self._get_config_path("img_folder")
        self.pcd_folder = pcd_folder or self._get_config_path("pcd_folder")

        self.Tr_lidar_to_cam = self._get_transform(lidar_camera)
        self.K = self._get_intrinsics(intrinsic_k)

        self.image_path = image_path
        self.pcd_path = pcd_path
        self.save_file = save_file

        # Visualization parameters
        self.point_size = point_size
        self.point_color = point_color
        self.depth_colormap = True if depth_colormap else True
        self.error_heatmap = False
        self.density_map = True
        self.depth_legend = True
        self.overlay_alpha = 0.7

        # State
        self.current_image = None
        self.current_points = None
        self.edge_distance = None
        self.original_transform = None
        self.base_transform = None

        # Pairs
        self.pairs = []
        self.current_index = 0

        # Tkinter UI
        self.root = None
        self.image_label = None
        self._photo = None
        self.pair_label_var = None
        self.tx_var = None
        self.ty_var = None
        self.tz_var = None
        self.roll_var = None
        self.pitch_var = None
        self.yaw_var = None
        self.alpha_var = None
        self.point_size_var = None
        self.density_var = None
        self.legend_var = None

        self.original_text = None
        self.current_text = None
        self._last_size = (0, 0)
        self._resize_job = None
        self.is_dark = True
        self.style = None
        self.dark_bg = "#1e1f24"
        self.dark_panel = "#2a2c33"
        self.dark_text = "#e6e6e6"
        self.dark_accent = "#3a7bd5"
        self.light_bg = "#f1f2f4"
        self.light_panel = "#ffffff"
        self.light_text = "#1f1f1f"
        self.font_base = ("Segoe UI", 11)
        self.font_small = ("Segoe UI", 10)
        self.font_mono = ("Consolas", 10)

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as config_file:
            return yaml.safe_load(config_file) or {}

    def _get_config_path(self, key: str) -> Optional[str]:
        return (self.config.get("path") or {}).get(key)

    def _get_transform(self, lidar_camera: Optional[Iterable[float]]) -> np.ndarray:
        raw = lidar_camera if lidar_camera is not None else (
            self.config.get("transform") or {}
        ).get("lidar_camera")
        if raw is None:
            return np.eye(4)
        return self._reshape_matrix(raw, (4, 4), "lidar_camera")

    def _get_intrinsics(self, intrinsic_k: Optional[Iterable[float]]) -> np.ndarray:
        raw = intrinsic_k if intrinsic_k is not None else (
            self.config.get("transform") or {}
        ).get("intrinsic_k")
        if raw is None:
            raise ValueError(
                "Missing camera intrinsics. Provide --intrinsic-k or transform.intrinsic_k in config."
            )
        return self._reshape_matrix(raw, (3, 3), "intrinsic_k")

    def _reshape_matrix(self, raw: Iterable[float], shape: Tuple[int, int], name: str) -> np.ndarray:
        arr = np.array(list(raw), dtype=float)
        expected = shape[0] * shape[1]
        if arr.size != expected:
            raise ValueError(f"{name} must have {expected} values, got {arr.size}.")
        return arr.reshape(shape)

    def load_pcd(self, file_path: str) -> np.ndarray:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        if points.size == 0:
            raise ValueError(f"Point cloud is empty: {file_path}")
        return points

    def project_points_to_image(
        self,
        points: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
        lidar_to_cam: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D LiDAR points into image space.

        Args:
            points: (N, 3) array in LiDAR frame.
            camera_matrix: Optional 3x3 intrinsics override.
            lidar_to_cam: Optional 4x4 extrinsics override.

        Returns:
            points_2d: (N, 2) image coordinates.
            points_3d: (N, 3) array of [u, v, depth] where depth is Z in camera frame.
        """
        K = np.array(camera_matrix) if camera_matrix is not None else self.K
        Tr = np.array(lidar_to_cam) if lidar_to_cam is not None else self.Tr_lidar_to_cam

        assert points.ndim == 2 and points.shape[1] == 3, f"Expected [N x 3] points, got {points.shape}"
        assert K.shape == (3, 3), f"Expected 3x3 intrinsic matrix, got {K.shape}"
        assert Tr.shape == (4, 4), f"Transformation matrix should be [4 x 4], got {Tr.shape}"

        points_homog = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))
        points_cam = (Tr @ points_homog.T).T[:, :3]

        points_proj = K @ points_cam.T
        points_2d = points_proj[:2, :] / points_proj[2, :]

        points_3d = np.column_stack([points_2d.T, points_cam[:, 2]])
        return points_2d.T, points_3d

    def get_depth_color(self, depth: float, min_depth: float, max_depth: float) -> Tuple[int, int, int]:
        norm_depth = np.clip((depth - min_depth) / (max_depth - min_depth + 1e-6), 0, 1)
        if norm_depth < 0.25:
            r = 0
            g = int(255 * (norm_depth / 0.25))
            b = 255
        elif norm_depth < 0.5:
            r = 0
            g = 255
            b = int(255 * (1 - (norm_depth - 0.25) / 0.25))
        elif norm_depth < 0.75:
            r = int(255 * ((norm_depth - 0.5) / 0.25))
            g = 255
            b = 0
        else:
            r = 255
            g = int(255 * (1 - (norm_depth - 0.75) / 0.25))
            b = 0
        return (b, g, r)

    def _compute_edge_distance(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edges_inv = cv2.bitwise_not(edges)
        return cv2.distanceTransform(edges_inv, cv2.DIST_L2, 3)

    def _draw_colorbar(
        self,
        image: np.ndarray,
        min_val: float,
        max_val: float,
        label: str,
        x: int = 10,
        y: int = 60,
        height: int = 120,
        width: int = 10,
    ) -> None:
        bar = np.linspace(1.0, 0.0, height, dtype=np.float32).reshape(height, 1)
        bar = (bar * 255).astype(np.uint8)
        bar = cv2.applyColorMap(bar, cv2.COLORMAP_TURBO)
        bar = cv2.resize(bar, (width, height), interpolation=cv2.INTER_NEAREST)
        image[y : y + height, x : x + width] = bar
        cv2.putText(image, f"{max_val:.2f}", (x + width + 6, y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f"{min_val:.2f}", (x + width + 6, y + height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, label, (x, y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def _render_density_map(self, image: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        density = np.zeros((h, w), dtype=np.float32)
        xs = np.clip(points_2d[:, 0].astype(int), 0, w - 1)
        ys = np.clip(points_2d[:, 1].astype(int), 0, h - 1)
        np.add.at(density, (ys, xs), 1.0)
        density = cv2.GaussianBlur(density, (0, 0), 3.0)
        if density.max() > 0:
            density = density / density.max()
        heat = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        return cv2.addWeighted(image, 0.7, heat, 0.3, 0)

    def overlay_points_on_image(
        self, image: np.ndarray, points_2d: np.ndarray, points_3d: np.ndarray
    ) -> np.ndarray:
        """Render projected points on the image.

        Args:
            image: BGR image.
            points_2d: (N, 2) image coordinates.
            points_3d: (N, 3) [u, v, depth] array (depth used for coloring).

        Returns:
            Image with points overlaid.
        """
        overlay = image.copy()
        h, w = image.shape[:2]
        valid_mask = (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h) &
            (points_3d[:, 2] > 0)
        )

        valid_points = points_2d[valid_mask]
        valid_depths = points_3d[valid_mask, 2]

        if len(valid_depths) == 0:
            return overlay

        min_depth = np.percentile(valid_depths, 5)
        max_depth = np.percentile(valid_depths, 95)

        if self.density_map:
            overlay = self._render_density_map(overlay, valid_points)

        colors = np.array(
            [self.get_depth_color(d, min_depth, max_depth) for d in valid_depths],
            dtype=np.uint8,
        )

        for (u, v), color in zip(valid_points, colors):
            if 0 <= u < w and 0 <= v < h:
                b, g, r = int(color[0]), int(color[1]), int(color[2])
                cv2.circle(overlay, (int(u), int(v)), self.point_size + 1, (0, 0, 0), -1)
                cv2.circle(overlay, (int(u), int(v)), self.point_size, (b, g, r), -1)

        result = cv2.addWeighted(overlay, self.overlay_alpha, image, 1 - self.overlay_alpha, 0)

        info_text = [
            f"Points visible: {len(valid_points)}/{len(points_2d)}",
            f"Depth range: {min_depth:.2f}m - {max_depth:.2f}m",
        ]

        y_offset = 30
        for text in info_text:
            cv2.putText(result, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 0, 0), 1, cv2.LINE_AA)
            y_offset += 25

        if self.depth_legend:
            x = w - 60
            self._draw_colorbar(result, min_depth, max_depth, "Depth (m)", x=x)

        return result

    def create_adjustment_matrix(self, adjustment: str, value: float) -> np.ndarray:
        mat = np.eye(4)
        if adjustment == "tx":
            mat[0, 3] = value
        elif adjustment == "ty":
            mat[1, 3] = value
        elif adjustment == "tz":
            mat[2, 3] = value
        elif adjustment == "rx":
            c, s = np.cos(value), np.sin(value)
            mat[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
        elif adjustment == "ry":
            c, s = np.cos(value), np.sin(value)
            mat[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        elif adjustment == "rz":
            c, s = np.cos(value), np.sin(value)
            mat[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        return mat

    def _apply_slider_transform(self) -> None:
        if self.base_transform is None:
            return
        tx = self.tx_var.get()
        ty = self.ty_var.get()
        tz = self.tz_var.get()
        roll = np.deg2rad(self.roll_var.get())
        pitch = np.deg2rad(self.pitch_var.get())
        yaw = np.deg2rad(self.yaw_var.get())

        delta = self.create_adjustment_matrix("tx", tx)
        delta = self.create_adjustment_matrix("ty", ty) @ delta
        delta = self.create_adjustment_matrix("tz", tz) @ delta
        delta = self.create_adjustment_matrix("rx", roll) @ delta
        delta = self.create_adjustment_matrix("ry", pitch) @ delta
        delta = self.create_adjustment_matrix("rz", yaw) @ delta

        self.Tr_lidar_to_cam = delta @ self.base_transform

    def _update_transform_text(self) -> None:
        if self.original_text is None or self.current_text is None:
            return
        self.original_text.config(state=tk.NORMAL)
        self.current_text.config(state=tk.NORMAL)
        self.original_text.delete("1.0", tk.END)
        self.current_text.delete("1.0", tk.END)
        if self.original_transform is not None:
            for row in self.original_transform:
                self.original_text.insert(tk.END, " ".join(f"{v:8.3f}" for v in row) + "\n")
        for row in self.Tr_lidar_to_cam:
            self.current_text.insert(tk.END, " ".join(f"{v:8.3f}" for v in row) + "\n")
        self.original_text.config(state=tk.DISABLED)
        self.current_text.config(state=tk.DISABLED)

    def update_display(self) -> None:
        if self.current_image is None or self.current_points is None:
            return
        self._apply_slider_transform()
        points_2d, points_3d = self.project_points_to_image(self.current_points)
        result = self.overlay_points_on_image(self.current_image, points_2d, points_3d)
        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        target_w = self.image_label.winfo_width()
        target_h = self.image_label.winfo_height()
        if target_w > 10 and target_h > 10:
            scale = min(target_w / img.width, target_h / img.height)
            new_w = max(1, int(img.width * scale))
            new_h = max(1, int(img.height * scale))
            if (new_w, new_h) != self._last_size:
                self._last_size = (new_w, new_h)
            img = img.resize((new_w, new_h), Image.BILINEAR)

        self._photo = ImageTk.PhotoImage(image=img)
        self.image_label.configure(image=self._photo)
        self._update_transform_text()

    def save_transformation(self, pair_number: int) -> None:
        if os.path.exists(self.save_file):
            with open(self.save_file, "r") as file:
                data = yaml.safe_load(file) or {}
        else:
            data = {}

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        entry_name = f"pair_{pair_number:03d}_{timestamp}"

        new_entry = {
            entry_name: {
                "pair_number": pair_number,
                "timestamp": timestamp,
                "Tr_lidar_to_cam": self.Tr_lidar_to_cam.tolist(),
                "translation": {
                    "x": float(self.Tr_lidar_to_cam[0, 3]),
                    "y": float(self.Tr_lidar_to_cam[1, 3]),
                    "z": float(self.Tr_lidar_to_cam[2, 3]),
                },
                "rotation_matrix": self.Tr_lidar_to_cam[:3, :3].tolist(),
            }
        }

        data.update(new_entry)

        with open(self.save_file, "w") as file:
            yaml.dump(data, file, default_flow_style=False)

        print(f"Transformation saved: {entry_name} -> {self.save_file}")

    def save_transform_txt(self, pair_number: int) -> None:
        base, _ext = os.path.splitext(self.save_file)
        txt_path = f"{base}.txt"
        with open(txt_path, "a", encoding="ascii") as f:
            f.write(f"pair {pair_number}\n")
            f.write("K:\n")
            for row in self.K:
                f.write("  " + " ".join(f"{v:.6f}" for v in row) + "\n")
            f.write("Tr_lidar_to_cam:\n")
            for row in self.Tr_lidar_to_cam:
                f.write("  " + " ".join(f"{v:.6f}" for v in row) + "\n")
            f.write("\n")
        print(f"Text saved: {txt_path}")

    def _iter_pairs(self) -> Iterable[Tuple[str, str]]:
        if self.image_path and self.pcd_path:
            yield self.image_path, self.pcd_path
            return

        if not self.img_folder or not self.pcd_folder:
            raise ValueError("Provide --image and --pcd, or --image-folder and --pcd-folder.")

        if not os.path.exists(self.img_folder) or not os.path.exists(self.pcd_folder):
            raise ValueError(
                f"Folders do not exist: image={self.img_folder}, pcd={self.pcd_folder}"
            )

        image_files = sorted(
            f for f in os.listdir(self.img_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        pcd_files = sorted(
            f for f in os.listdir(self.pcd_folder) if f.lower().endswith(".pcd")
        )

        if not image_files or not pcd_files:
            raise ValueError(
                f"No files found. Images={len(image_files)}, PCDs={len(pcd_files)}"
            )

        num_pairs = min(len(image_files), len(pcd_files))
        print(f"Found {num_pairs} pair(s): {len(image_files)} images, {len(pcd_files)} point clouds")

        for i in range(num_pairs):
            image_path = os.path.join(self.img_folder, image_files[i])
            pcd_path = os.path.join(self.pcd_folder, pcd_files[i])
            yield image_path, pcd_path

    def _load_pair(self, index: int) -> None:
        image_path, pcd_path = self.pairs[index]
        self.current_image = cv2.imread(image_path)
        if self.current_image is None:
            raise ValueError(f"Could not read image: {image_path}")
        self.current_points = self.load_pcd(pcd_path)
        self.edge_distance = self._compute_edge_distance(self.current_image)

        self.original_transform = self.Tr_lidar_to_cam.copy()
        self.base_transform = self.Tr_lidar_to_cam.copy()

        self.tx_var.set(0.0)
        self.ty_var.set(0.0)
        self.tz_var.set(0.0)
        self.roll_var.set(0.0)
        self.pitch_var.set(0.0)
        self.yaw_var.set(0.0)

        name = f"{os.path.basename(image_path)} | {os.path.basename(pcd_path)}"
        self.pair_label_var.set(f"Pair {index + 1}/{len(self.pairs)}: {name}")
        self.update_display()

    def _on_save(self) -> None:
        self.save_transformation(self.current_index + 1)
        self.save_transform_txt(self.current_index + 1)

    def _on_reset(self) -> None:
        if self.original_transform is None:
            return
        self.Tr_lidar_to_cam = self.original_transform.copy()
        self.base_transform = self.original_transform.copy()
        self.tx_var.set(0.0)
        self.ty_var.set(0.0)
        self.tz_var.set(0.0)
        self.roll_var.set(0.0)
        self.pitch_var.set(0.0)
        self.yaw_var.set(0.0)
        self.update_display()

    def _on_prev(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self._load_pair(self.current_index)

    def _on_next(self) -> None:
        if self.current_index < len(self.pairs) - 1:
            self.current_index += 1
            self._load_pair(self.current_index)

    def _on_alpha(self, _val: str) -> None:
        self.overlay_alpha = self.alpha_var.get()
        self.update_display()

    def _on_point_size(self, _val: str) -> None:
        self.point_size = self.point_size_var.get()
        self.update_display()

    def _on_toggle(self) -> None:
        self.density_map = bool(self.density_var.get())
        self.depth_legend = bool(self.legend_var.get())
        self.update_display()

    def _on_slider(self, _val: str) -> None:
        self.update_display()

    def _on_key(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key == "a":
            self.tx_var.set(max(-MAX_TRANSLATION, self.tx_var.get() - 0.001))
        elif key == "d":
            self.tx_var.set(min(MAX_TRANSLATION, self.tx_var.get() + 0.001))
        elif key == "w":
            self.ty_var.set(max(-MAX_TRANSLATION, self.ty_var.get() - 0.001))
        elif key == "s":
            self.ty_var.set(min(MAX_TRANSLATION, self.ty_var.get() + 0.001))
        elif key == "q":
            self.tz_var.set(max(-MAX_TRANSLATION, self.tz_var.get() - 0.001))
        elif key == "e":
            self.tz_var.set(min(MAX_TRANSLATION, self.tz_var.get() + 0.001))
        elif key == "j":
            self.roll_var.set(max(-MAX_ROTATION_DEG, self.roll_var.get() - 0.1))
        elif key == "l":
            self.roll_var.set(min(MAX_ROTATION_DEG, self.roll_var.get() + 0.1))
        elif key == "i":
            self.pitch_var.set(max(-MAX_ROTATION_DEG, self.pitch_var.get() - 0.1))
        elif key == "k":
            self.pitch_var.set(min(MAX_ROTATION_DEG, self.pitch_var.get() + 0.1))
        elif key == "u":
            self.yaw_var.set(max(-MAX_ROTATION_DEG, self.yaw_var.get() - 0.1))
        elif key == "o":
            self.yaw_var.set(min(MAX_ROTATION_DEG, self.yaw_var.get() + 0.1))
        elif key == "r":
            self._on_reset()
        elif key == "p":
            self._on_save()
        elif key == "n":
            self._on_next()
        elif key == "bracketleft":
            self.point_size_var.set(max(1, self.point_size_var.get() - 1))
        elif key == "bracketright":
            self.point_size_var.set(min(10, self.point_size_var.get() + 1))
        elif key == "minus":
            self.alpha_var.set(max(0.0, self.alpha_var.get() - 0.1))
            self._on_alpha("")
        elif key == "equal":
            self.alpha_var.set(min(1.0, self.alpha_var.get() + 0.1))
            self._on_alpha("")

    def _on_resize(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            try:
                self.root.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.root.after(60, self.update_display)

    def _apply_theme(self) -> None:
        if self.root is None:
            return
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        if self.is_dark:
            bg = self.dark_bg
            panel = self.dark_panel
            text = self.dark_text
            button_bg = self.dark_accent
        else:
            bg = self.light_bg
            panel = self.light_panel
            text = self.light_text
            button_bg = "#4a7bd8"

        self.root.configure(bg=bg)
        self.style.configure("TFrame", background=bg)
        self.style.configure("Panel.TFrame", background=panel)
        self.style.configure("TLabel", background=bg, foreground=text, font=self.font_base)
        self.style.configure("Panel.TLabel", background=panel, foreground=text, font=self.font_base)
        self.style.configure("TButton", background=button_bg, foreground="white", font=self.font_base)
        self.style.map("TButton", background=[("active", button_bg)])
        self.style.configure("TCheckbutton", background=bg, foreground=text, font=self.font_base)

        for text_widget in (self.original_text, self.current_text):
            if text_widget is not None:
                text_widget.configure(
                    bg=panel,
                    fg=text,
                    insertbackground=text,
                    highlightbackground=panel,
                    font=self.font_mono,
                )

    def _toggle_theme(self) -> None:
        self.is_dark = not self.is_dark
        self._apply_theme()

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("SensorCal — Recalibrate (LiDAR/Camera Calibration)")
        self.root.geometry("1400x900")
        self.root.bind("<Key>", self._on_key)

        self.pair_label_var = tk.StringVar(master=self.root)
        self.tx_var = tk.DoubleVar(master=self.root, value=0.0)
        self.ty_var = tk.DoubleVar(master=self.root, value=0.0)
        self.tz_var = tk.DoubleVar(master=self.root, value=0.0)
        self.roll_var = tk.DoubleVar(master=self.root, value=0.0)
        self.pitch_var = tk.DoubleVar(master=self.root, value=0.0)
        self.yaw_var = tk.DoubleVar(master=self.root, value=0.0)
        self.alpha_var = tk.DoubleVar(master=self.root, value=self.overlay_alpha)
        self.point_size_var = tk.IntVar(master=self.root, value=self.point_size)
        self.density_var = tk.BooleanVar(master=self.root, value=self.density_map)
        self.legend_var = tk.BooleanVar(master=self.root, value=self.depth_legend)

        self._apply_theme()

        main = ttk.Frame(self.root, padding=8, style="TFrame")
        main.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        left = ttk.Frame(main, style="TFrame")
        left.grid(row=0, column=0, sticky="nsew")
        right = ttk.Frame(main, style="Panel.TFrame")
        right.grid(row=0, column=1, sticky="ns")

        self.image_label = ttk.Label(left, style="TLabel")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        left.bind("<Configure>", self._on_resize)

        ttk.Label(right, textvariable=self.pair_label_var, wraplength=360, style="Panel.TLabel").pack(
            anchor="w", pady=(0, 8)
        )

        def slider(parent, label, var, from_, to_, resolution, command):
            frame = ttk.Frame(parent, style="Panel.TFrame")
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label, width=12, style="Panel.TLabel").pack(side=tk.LEFT)
            scale = tk.Scale(
                frame,
                variable=var,
                from_=from_,
                to=to_,
                resolution=resolution,
                orient=tk.HORIZONTAL,
                length=240,
                command=command,
            )
            scale.configure(font=self.font_small)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            val = ttk.Label(frame, textvariable=var, width=8, style="Panel.TLabel")
            val.pack(side=tk.RIGHT)

        slider(right, "tx (m)", self.tx_var, -MAX_TRANSLATION, MAX_TRANSLATION, 0.001, self._on_slider)
        slider(right, "ty (m)", self.ty_var, -MAX_TRANSLATION, MAX_TRANSLATION, 0.001, self._on_slider)
        slider(right, "tz (m)", self.tz_var, -MAX_TRANSLATION, MAX_TRANSLATION, 0.001, self._on_slider)
        slider(right, "roll (deg)", self.roll_var, -MAX_ROTATION_DEG, MAX_ROTATION_DEG, 0.1, self._on_slider)
        slider(right, "pitch (deg)", self.pitch_var, -MAX_ROTATION_DEG, MAX_ROTATION_DEG, 0.1, self._on_slider)
        slider(right, "yaw (deg)", self.yaw_var, -MAX_ROTATION_DEG, MAX_ROTATION_DEG, 0.1, self._on_slider)
        slider(right, "alpha", self.alpha_var, 0.0, 1.0, 0.05, self._on_alpha)
        slider(right, "pt size", self.point_size_var, 1, 10, 1, self._on_point_size)

        toggles = ttk.Frame(right, style="Panel.TFrame")
        toggles.pack(fill=tk.X, pady=(6, 6))
        ttk.Checkbutton(toggles, text="Density", variable=self.density_var, command=self._on_toggle).pack(side=tk.LEFT)
        ttk.Checkbutton(toggles, text="Legend", variable=self.legend_var, command=self._on_toggle).pack(side=tk.LEFT)
        ttk.Button(toggles, text="Dark/Light", command=self._toggle_theme).pack(side=tk.RIGHT)

        btns = ttk.Frame(right, style="Panel.TFrame")
        btns.pack(fill=tk.X, pady=(4, 10))
        ttk.Button(btns, text="Prev", command=self._on_prev).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Next", command=self._on_next).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Original", command=self._on_reset).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Save", command=self._on_save).pack(side=tk.LEFT, padx=2)

        ttk.Label(right, text="Original Transform", style="Panel.TLabel").pack(anchor="w")
        self.original_text = tk.Text(right, width=44, height=6, wrap=tk.NONE, font=self.font_mono)
        self.original_text.pack(fill=tk.X, pady=(0, 6))
        self.original_text.config(state=tk.DISABLED)

        ttk.Label(right, text="Current Transform", style="Panel.TLabel").pack(anchor="w")
        self.current_text = tk.Text(right, width=44, height=6, wrap=tk.NONE, font=self.font_mono)
        self.current_text.pack(fill=tk.X, pady=(0, 6))
        self.current_text.config(state=tk.DISABLED)

        self._apply_theme()

    def process(self) -> None:
        """Launch the GUI and start interactive calibration."""
        print("\nINTERACTIVE LIDAR-CAMERA CALIBRATION")
        self.pairs = list(self._iter_pairs())
        if not self.pairs:
            raise ValueError("No pairs found to process.")

        self._build_ui()
        self._load_pair(0)
        self.root.mainloop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive LiDAR-to-camera calibration with GUI controls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", help="Path to YAML config with intrinsics and transform")
    parser.add_argument(
        "--intrinsic-k",
        nargs=9,
        type=float,
        metavar=("K00", "K01", "K02", "K10", "K11", "K12", "K20", "K21", "K22"),
        help="Camera intrinsics as 9 values (row-major 3x3)",
    )
    parser.add_argument(
        "--lidar-camera",
        nargs=16,
        type=float,
        metavar=("T00", "T01", "T02", "T03", "T10", "T11", "T12", "T13",
                 "T20", "T21", "T22", "T23", "T30", "T31", "T32", "T33"),
        help="LiDAR-to-camera transform as 16 values (row-major 4x4)",
    )

    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--pcd", help="Path to a single PCD")
    parser.add_argument("--image-folder", help="Folder containing images")
    parser.add_argument("--pcd-folder", help="Folder containing point clouds")
    parser.add_argument("--use-sample", action="store_true", help="Use bundled sample data")

    parser.add_argument(
        "--save-file",
        default="calibration_results.yaml",
        help="Path to output YAML file (default: calibration_results.yaml)",
    )

    parser.add_argument("--point-size", type=int, default=2, help="Size of projected points")
    parser.add_argument(
        "--point-color",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        default=[0, 255, 0],
        help="RGB color for points when depth coloring is off (unused)",
    )

    args = parser.parse_args()

    if (args.image and not args.pcd) or (args.pcd and not args.image):
        parser.error("--image and --pcd must be provided together")
    if (args.image_folder and not args.pcd_folder) or (args.pcd_folder and not args.image_folder):
        parser.error("--image-folder and --pcd-folder must be provided together")
    if not ((args.image and args.pcd) or (args.image_folder and args.pcd_folder) or args.use_sample):
        print("No input specified, using sample data")
        args.use_sample = True

    return args


def main() -> None:
    args = _parse_args()

    if args.use_sample:
        if not args.image_folder and not args.image:
            args.image_folder = os.path.abspath(SAMPLE_IMAGE_FOLDER)
            args.pcd_folder = os.path.abspath(SAMPLE_PCD_FOLDER)
        if args.intrinsic_k is None and args.config is None:
            args.intrinsic_k = SAMPLE_INTRINSIC_K
        if args.lidar_camera is None and args.config is None:
            args.lidar_camera = SAMPLE_LIDAR_CAMERA
        print(
            f"Using sample data from:\n  Images: {args.image_folder}\n  PCDs: {args.pcd_folder}"
        )

    app = SensorCalApp(
        config_path=args.config,
        intrinsic_k=args.intrinsic_k,
        lidar_camera=args.lidar_camera,
        image_path=args.image,
        pcd_path=args.pcd,
        image_folder=args.image_folder,
        pcd_folder=args.pcd_folder,
        save_file=args.save_file,
        point_size=args.point_size,
        point_color=tuple(args.point_color[::-1]),
        depth_colormap=True,
    )

    app.process()


if __name__ == "__main__":
    main()
