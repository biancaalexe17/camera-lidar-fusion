
"""Load validation frames for the existing camera-LiDAR fusion pipeline."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.config import PATHS, VALIDATION_CONTEXTS


def component_path(component, context):
    """Validation component files use <context>.parquet filenames."""
    path = PATHS.validation_data / component / f"{context}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation file: {path}")
    return path


def load_validation_calibration(context, camera_name=1):
    """Convert a flattened Waymo calibration row to the fusion format."""
    df = pd.read_parquet(component_path("camera_calibration", context))
    rows = df[df["key.camera_name"] == camera_name]
    if rows.empty:
        raise ValueError(f"Camera {camera_name} missing for {context}")

    row = rows.iloc[0]
    prefix = "[CameraCalibrationComponent]."

    intrinsic = {
        key: float(row[f"{prefix}intrinsic.{key}"])
        for key in ("f_u", "f_v", "c_u", "c_v", "k1", "k2", "p1", "p2", "k3")
    }

    transform = np.asarray(
        row[f"{prefix}extrinsic.transform"], dtype=np.float64
    ).reshape(4, 4)

    return {
        "intrinsic": intrinsic,
        "extrinsic": transform,
        "width": int(row[f"{prefix}width"]),
        "height": int(row[f"{prefix}height"]),
    }


def load_validation_camera_frames(context, camera_name=1):
    """Load camera images and their timestamps, ordered chronologically."""
    df = pd.read_parquet(component_path("camera_image", context))
    df = df[df["key.camera_name"] == camera_name].copy()
    df = df.sort_values("key.frame_timestamp_micros")

    frames = []
    for _, row in df.iterrows():
        image = Image.open(
            io.BytesIO(row["[CameraImageComponent].image"])
        ).convert("RGB")
        frames.append({
            "timestamp": int(row["key.frame_timestamp_micros"]),
            "image": image,
        })

    if not frames:
        raise ValueError(f"No camera frames found for {context}")

    return frames


def load_validation_lidar_frames(context):
    """Load TOP LiDAR point clouds through the existing Waymo toolkit."""
    from src.data.waymo_loader import load_top_lidar_points

    lidar_df = pd.read_parquet(component_path("lidar", context))
    timestamps = sorted(
        int(t) for t in lidar_df["key.frame_timestamp_micros"].unique()
    )

    point_clouds = {}
    for timestamp in timestamps:
        point_clouds[timestamp] = load_top_lidar_points(
            context,
            timestamp,
            split="validation",
            data_root=PATHS.drive_root,
        )

    return point_clouds
