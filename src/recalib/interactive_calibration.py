"""Interactive LiDAR-to-camera calibration tool."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import yaml


SAMPLE_IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "samples", "images")
SAMPLE_PCD_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "samples", "pcds")
SAMPLE_INTRINSIC_K = [554.26, 0.0, 960.0, 0.0, 554.26, 540.0, 0.0, 0.0, 1.0]
SAMPLE_LIDAR_CAMERA = [
    0.86603,
    -0.5,
    -2.4936e-13,
    -1.4289,
    -7.7825e-09,
    -1.3479e-08,
    -1.0,
    -0.6,
    0.5,
    0.86603,
    -1.5565e-08,
    -0.825,
    0.0,
    0.0,
    0.0,
    1.0,
]


class InteractiveCalibration:
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
    ) -> None:
        self.config = self._load_config(config_path) if config_path else {}

        self.img_folder = image_folder or self._get_config_path("img_folder")
        self.pcd_folder = pcd_folder or self._get_config_path("pcd_folder")

        self.Tr_lidar_to_cam = self._get_transform(lidar_camera)
        self.K = self._get_intrinsics(intrinsic_k)

        self.image_path = image_path
        self.pcd_path = pcd_path
        self.save_file = save_file

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as config_file:
            return yaml.safe_load(config_file) or {}

    def _get_config_path(self, key: str) -> Optional[str]:
        return (self.config.get("path") or {}).get(key)

    def _get_transform(self, lidar_camera: Optional[Iterable[float]]) -> np.ndarray:
        raw = lidar_camera if lidar_camera is not None else (self.config.get("transform") or {}).get(
            "lidar_camera"
        )
        if raw is None:
            return np.eye(4)
        return self._reshape_matrix(raw, (4, 4), "lidar_camera")

    def _get_intrinsics(self, intrinsic_k: Optional[Iterable[float]]) -> np.ndarray:
        raw = intrinsic_k if intrinsic_k is not None else (self.config.get("transform") or {}).get(
            "intrinsic_k"
        )
        if raw is None:
            raise ValueError(
                "Missing camera intrinsics. Provide --intrinsic-k or transform.intrinsic_k in config."
            )
        return self._reshape_matrix(raw, (3, 3), "intrinsic_k")

    def _reshape_matrix(
        self, raw: Iterable[float], shape: Tuple[int, int], name: str
    ) -> np.ndarray:
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
        # Add homogeneous coordinate
        return np.hstack((points, np.ones((points.shape[0], 1))))

    def project_points_to_image(self, points: np.ndarray) -> np.ndarray:
        # Transform points from lidar to camera coordinate
        points_cam = self.Tr_lidar_to_cam @ points.T

        # Project to image plane
        points_2d = self.K @ points_cam[:3, :]
        points_2d = points_2d[:2, :] / points_2d[2, :]
        return points_2d.T

    def overlay_points_on_image(self, image: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
        overlay = image.copy()
        for (u, v) in points_2d:
            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                cv2.circle(overlay, (int(u), int(v)), 4, (0, 255, 0), -1)
        return cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

    def adjust_transformation(self, adjustment: str, value: float) -> None:
        adjustment_matrix = np.eye(4)
        if adjustment in ["r", "l", "u", "d"]:
            axis = {"r": 0, "l": 0, "u": 1, "d": 1}[adjustment]
            sign = 1 if adjustment in ["r", "u"] else -1
            adjustment_matrix[axis, 3] = sign * value
        elif adjustment in ["rl", "rr", "ru", "rd"]:
            axis = {"rl": 2, "rr": 2, "ru": 1, "rd": 0}[adjustment]
            angle = value if adjustment in ["rr", "rd"] else -value
            c, s = np.cos(angle), np.sin(angle)
            if axis == 0:
                adjustment_matrix[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
            elif axis == 1:
                adjustment_matrix[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
            else:
                adjustment_matrix[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]

        self.Tr_lidar_to_cam = adjustment_matrix @ self.Tr_lidar_to_cam

    def update_display(self, image: np.ndarray, points: np.ndarray) -> None:
        points_2d = self.project_points_to_image(points)
        result = self.overlay_points_on_image(image, points_2d)
        cv2.imshow("LiDAR Overlay", result)
        cv2.waitKey(1)

    def save_transformation(self, pair_number: int) -> None:
        if os.path.exists(self.save_file):
            with open(self.save_file, "r") as file:
                data = yaml.safe_load(file) or {}
        else:
            data = {}

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = {
            f"transformation_{timestamp}": {
                "pair_number": pair_number,
                "Tr_lidar_to_cam": self.Tr_lidar_to_cam.tolist(),
            }
        }

        data.update(new_entry)

        with open(self.save_file, "w") as file:
            yaml.dump(data, file)

        print(f"Transformation for pair {pair_number} saved to {self.save_file}")

    def process_pair(self, image_path: str, pcd_path: str, pair_number: int) -> None:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        points = self.load_pcd(pcd_path)

        cv2.namedWindow("LiDAR Overlay", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("LiDAR Overlay", image.shape[1], image.shape[0])

        self.update_display(image, points)

        while True:
            choice = input("Adjust [r,l,u,d,rl,rr,ru,rd], save (s), or next (n)? ").lower()

            if choice == "s":
                print("New Tr_lidar_to_cam:")
                print(self.Tr_lidar_to_cam)
                self.save_transformation(pair_number)
            elif choice == "n":
                print("Tr_lidar_to_cam:")
                print(self.Tr_lidar_to_cam)
                break
            elif choice in ["r", "l", "u", "d", "rl", "rr", "ru", "rd"]:
                value = float(input(f"Enter value for {choice}: "))
                self.adjust_transformation(choice, value)
                print("New Tr_lidar_to_cam:")
                print(self.Tr_lidar_to_cam)
                self.update_display(image, points)
            else:
                print("Invalid choice. Please try again.")

    def _iter_pairs(self) -> Iterable[Tuple[str, str]]:
        if self.image_path and self.pcd_path:
            yield self.image_path, self.pcd_path
            return

        if not self.img_folder or not self.pcd_folder:
            raise ValueError("Provide --image and --pcd, or --image-folder and --pcd-folder.")

        if not os.path.exists(self.img_folder) or not os.path.exists(self.pcd_folder):
            raise ValueError(
                "One or both folders do not exist: "
                f"image folder={self.img_folder}, pcd folder={self.pcd_folder}"
            )

        image_files = sorted(
            f for f in os.listdir(self.img_folder) if f.startswith("img_") and f.endswith(".png")
        )
        pcd_files = sorted(
            f for f in os.listdir(self.pcd_folder) if f.startswith("pc_") and f.endswith(".pcd")
        )

        num_pairs = min(len(image_files), len(pcd_files))
        if num_pairs == 0:
            raise ValueError(
                f"No matching pairs found. Images={len(image_files)}, PCDs={len(pcd_files)}"
            )

        for i in range(num_pairs):
            image_path = os.path.join(self.img_folder, image_files[i])
            pcd_path = os.path.join(self.pcd_folder, pcd_files[i])
            yield image_path, pcd_path

    def process(self) -> None:
        print("Starting interactive calibration...")
        pairs = list(self._iter_pairs())
        print(f"Found {len(pairs)} pair(s) to process")

        blank_image = None
        first_image = True

        for i, (image_path, pcd_path) in enumerate(pairs, start=1):
            try:
                print(f"Processing pair {i}/{len(pairs)}: {os.path.basename(image_path)} - {os.path.basename(pcd_path)}")
                self.process_pair(image_path, pcd_path, i)

                if first_image:
                    first_image = False
                    img = cv2.imread(image_path)
                    if img is not None:
                        blank_image = np.zeros(img.shape, np.uint8)
            except Exception as exc:
                print(f"Error processing pair {i}/{len(pairs)}: {exc}")
                if blank_image is not None:
                    cv2.imshow("LiDAR Overlay", blank_image)
                    cv2.waitKey(1)
                continue

        print("\nProcessing complete!")
        print("\nFinal transformation matrix:")
        print(self.Tr_lidar_to_cam)
        print("\nPress any key in the image window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive LiDAR-to-camera calibration")
    parser.add_argument("--config", help="Path to YAML config with intrinsics and transform")
    parser.add_argument(
        "--intrinsic-k",
        nargs=9,
        type=float,
        metavar=("K00", "K01", "K02", "K10", "K11", "K12", "K20", "K21", "K22"),
        help="Camera intrinsics as 9 values (row-major 3x3). Overrides config.",
    )
    parser.add_argument(
        "--lidar-camera",
        nargs=16,
        type=float,
        metavar=(
            "T00",
            "T01",
            "T02",
            "T03",
            "T10",
            "T11",
            "T12",
            "T13",
            "T20",
            "T21",
            "T22",
            "T23",
            "T30",
            "T31",
            "T32",
            "T33",
        ),
        help="LiDAR-to-camera transform as 16 values (row-major 4x4). Overrides config.",
    )

    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--pcd", help="Path to a single PCD")

    parser.add_argument("--image-folder", help="Folder containing images (img_*.png)")
    parser.add_argument("--pcd-folder", help="Folder containing PCDs (pc_*.pcd)")

    parser.add_argument(
        "--save-file",
        default="transformations.yaml",
        help="Path to output YAML for saved transformations",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use bundled sample images/PCDs and default intrinsics.",
    )

    args = parser.parse_args()

    if (args.image and not args.pcd) or (args.pcd and not args.image):
        parser.error("--image and --pcd must be provided together")

    if (args.image_folder and not args.pcd_folder) or (args.pcd_folder and not args.image_folder):
        parser.error("--image-folder and --pcd-folder must be provided together")

    if not ((args.image and args.pcd) or (args.image_folder and args.pcd_folder) or args.use_sample):
        args.use_sample = True

    return args


def main() -> None:
    args = _parse_args()
    if args.use_sample:
        if not args.image_folder and not args.image and not args.pcd and not args.pcd_folder:
            args.image_folder = os.path.abspath(SAMPLE_IMAGE_FOLDER)
            args.pcd_folder = os.path.abspath(SAMPLE_PCD_FOLDER)
        if args.intrinsic_k is None and args.config is None:
            args.intrinsic_k = SAMPLE_INTRINSIC_K
        if args.lidar_camera is None and args.config is None:
            args.lidar_camera = SAMPLE_LIDAR_CAMERA
    app = InteractiveCalibration(
        config_path=args.config,
        intrinsic_k=args.intrinsic_k,
        lidar_camera=args.lidar_camera,
        image_path=args.image,
        pcd_path=args.pcd,
        image_folder=args.image_folder,
        pcd_folder=args.pcd_folder,
        save_file=args.save_file,
    )
    app.process()


if __name__ == "__main__":
    main()
