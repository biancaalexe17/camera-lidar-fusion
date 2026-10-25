
"""Convert one Waymo v2 context into the pickle format used by fusion."""

# CONTEXT_NAME = "10017090168044687777_6380_000_6400_000"
# SPLIT = "training"
# DATA_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/3DPerception")
# OUTPUT_ROOT = Path("/content/waymo_fusion_converted")

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from waymo_open_dataset import v2

SPLIT = "validation"
DATA_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/3DPerception/validation")
OUTPUT_ROOT = Path("/content/waymo_fusion_validation_converted")
VALIDATION_CONTEXTS = [
    ("10203656353524179475_7625_000_7645_000", 198),
    ("1024360143612057520_3580_000_3600_000", 199),
    ("10247954040621004675_2180_000_2200_000", 198),
    ("10289507859301986274_4200_000_4220_000", 198),
    ("10335539493577748957_1372_870_1392_870", 198),
]

def _component_path(root, split, component, context):
    """Resolve Waymo v2 component paths for the selected split."""
    root = Path(root)

    if split == "validation":
        return root / component / f"{context}.parquet"

    prefixed = {"lidar", "lidar_calibration"}
    filename = (
        f"{split}_{component}_{context}.parquet"
        if component in prefixed
        else f"{context}.parquet"
    )
    return root / split / component / filename

def _save_pickle(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def _camera_calibration(row, context):
    """Keep the calibration keys expected by the fusion projection code."""
    return {
        "camera_name": int(row["key.camera_name"]),
        "context": context,
        "intrinsic": {
            key: float(row[f"[CameraCalibrationComponent].intrinsic.{key}"])
            for key in ("f_u", "f_v", "c_u", "c_v", "k1", "k2", "k3", "p1", "p2")
        },
        "extrinsic": np.asarray(
            row["[CameraCalibrationComponent].extrinsic.transform"]
        ).reshape(4, 4).tolist(),
        "width": int(row["[CameraCalibrationComponent].width"]),
        "height": int(row["[CameraCalibrationComponent].height"]),
        "rolling_shutter_direction": int(
            row["[CameraCalibrationComponent].rolling_shutter_direction"]
        ),
    }


def _point_clouds_for_sensor(lidar_row, calibration_row, pose_row, vehicle_row):
    """Convert both TOP-LiDAR returns to [x, y, z, intensity, elongation]."""
    lidar = v2.LiDARComponent.from_dict(lidar_row.to_dict())
    calibration = v2.LiDARCalibrationComponent.from_dict(
        calibration_row.to_dict()
    )
    vehicle_pose = v2.VehiclePoseComponent.from_dict(vehicle_row.to_dict())

    pixel_pose = None
    if pose_row is not None:
        pixel_pose = v2.LiDARPoseComponent.from_dict(
            pose_row.to_dict()
        ).range_image_return1

    point_clouds = {}
    for return_number in (1, 2):
        attribute = f"range_image_return{return_number}"
        range_image = getattr(lidar, attribute, None)
        if range_image is None:
            continue

        points = v2.convert_range_image_to_point_cloud(
            range_image=range_image,
            calibration=calibration,
            pixel_pose=pixel_pose,
            frame_pose=vehicle_pose,
            keep_polar_features=True,
        )
        points = np.asarray(points, dtype=np.float32)
        point_clouds[f"1_return{return_number}"] = points[:, [3, 4, 5, 1, 2]]

    return point_clouds


def convert_context(
    root=DATA_ROOT,
    output_root=OUTPUT_ROOT,
    split=SPLIT,
    context=None,
):
    """Convert synchronized camera 1 and TOP-LiDAR frames for one context."""
    root = Path(root)
    output_root = Path(output_root)

    paths = {
        name: _component_path(root, split, name, context)
        for name in (
            "camera_image",
            "camera_calibration",
            "lidar",
            "lidar_calibration",
            "vehicle_pose",
            "lidar_pose",
        )
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required Waymo Parquet files:\n" + "\n".join(missing)
        )

    timestamp_col = "key.frame_timestamp_micros"
    camera_id_col = "key.camera_name"
    lidar_id_col = "key.laser_name"
    image_col = "[CameraImageComponent].image"

    # These tables are small compared with the serialized LiDAR range images.
    camera_df = pd.read_parquet(paths["camera_image"])
    camera_cal_df = pd.read_parquet(paths["camera_calibration"])
    lidar_cal_df = pd.read_parquet(paths["lidar_calibration"])
    vehicle_df = pd.read_parquet(paths["vehicle_pose"])

    timestamps = sorted(
        int(ts) for ts in vehicle_df[timestamp_col].dropna().unique()
    )
    if not timestamps:
        raise ValueError(f"No vehicle-pose timestamps found for {context}.")

    camera_rows = camera_df[camera_df[camera_id_col] == 1].set_index(
        timestamp_col, drop=False
    )
    camera_cal_rows = camera_cal_df[camera_cal_df[camera_id_col] == 1]
    lidar_cal_rows = lidar_cal_df[lidar_cal_df[lidar_id_col] == 1]

    if camera_cal_rows.empty:
        raise ValueError(f"Camera 1 calibration not found for {context}.")
    if lidar_cal_rows.empty:
        raise ValueError(f"TOP-LiDAR calibration not found for {context}.")

    context_dir = output_root / context
    camera_dir = context_dir / "camera_image" / "1"
    lidar_return1_dir = context_dir / "lidar" / "1" / "return1"
    lidar_return2_dir = context_dir / "lidar" / "1" / "return2"

    calibration = _camera_calibration(camera_cal_rows.iloc[0], context)
    _save_pickle(
        context_dir / "camera_calibration" / "calibration.pkl",
        {1: calibration},
    )

    lidar_calibration_row = lidar_cal_rows.iloc[0]


    written = 0
    skipped = 0
    total = len(timestamps)

    camera_dir.mkdir(parents=True, exist_ok=True)
    lidar_return1_dir.mkdir(parents=True, exist_ok=True)
    lidar_return2_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, timestamp in enumerate(timestamps):
        filename = f"{frame_idx:03d}.pkl"
        camera_path = camera_dir / filename
        return1_path = lidar_return1_dir / filename
        return2_path = lidar_return2_dir / filename

        # Resume: skip only if all three files already exist.
        if camera_path.is_file() and return1_path.is_file() and return2_path.is_file():
            skipped += 1
            continue

        # Preserve the original converter's shared, timestamp-based numbering.
        if timestamp not in camera_rows.index:
            continue

        camera_row = camera_rows.loc[timestamp]
        if isinstance(camera_row, pd.DataFrame):
            camera_row = camera_row.iloc[0]

        vehicle_rows = vehicle_df[vehicle_df[timestamp_col] == timestamp]
        if vehicle_rows.empty:
            continue
        vehicle_row = vehicle_rows.iloc[0]

        lidar_frame = pd.read_parquet(
            paths["lidar"],
            filters=[
                (timestamp_col, "==", timestamp),
                (lidar_id_col, "==", 1),
            ],
        )
        if lidar_frame.empty:
            continue

        pose_frame = pd.read_parquet(
            paths["lidar_pose"],
            filters=[
                (timestamp_col, "==", timestamp),
                (lidar_id_col, "==", 1),
            ],
        )
        pose_row = pose_frame.iloc[0] if not pose_frame.empty else None

        point_clouds = _point_clouds_for_sensor(
            lidar_frame.iloc[0],
            lidar_calibration_row,
            pose_row,
            vehicle_row,
        )

        if "1_return1" not in point_clouds or "1_return2" not in point_clouds:
            continue

        image_bytes = camera_row[image_col]
        if isinstance(image_bytes, np.ndarray):
            image_bytes = image_bytes.tobytes()
        else:
            image_bytes = bytes(image_bytes)

        _save_pickle(camera_path, {"image": image_bytes, "timestamp": timestamp})
        _save_pickle(
            return1_path,
            {"points": point_clouds["1_return1"], "timestamp": timestamp},
        )
        _save_pickle(
            return2_path,
            {"points": point_clouds["1_return2"], "timestamp": timestamp},
        )

        written += 1

        if written % 5 == 0:
            print(
                f"[{context}] timestamp {frame_idx + 1}/{total} | "
                f"new frames: {written} | skipped: {skipped}",
                flush=True,
            )

    camera_count = len(list(camera_dir.glob("*.pkl")))
    return1_count = len(list(lidar_return1_dir.glob("*.pkl")))
    return2_count = len(list(lidar_return2_dir.glob("*.pkl")))

    print(
        f"Finished {context}: new={written}, skipped={skipped}, "
        f"camera={camera_count}, return1={return1_count}, "
        f"return2={return2_count}",
        flush=True,
    )

    if not (camera_count == return1_count == return2_count):
        raise RuntimeError(f"Frame count mismatch for {context}")

    if camera_count == 0:
        raise RuntimeError(f"No synchronized frames converted for {context}.")

    return context_dir


if __name__ == "__main__":
    results = []

    for context, expected_frames in VALIDATION_CONTEXTS:
        print(f"\n{'=' * 70}")
        print(f"Converting context: {context}")
        print(f"Expected frames: {expected_frames}")
        print(f"{'=' * 70}")

        context_dir = convert_context(
            root=DATA_ROOT,
            output_root=OUTPUT_ROOT,
            split=SPLIT,
            context=context,
        )

        camera_dir = context_dir / "camera_image" / "1"
        lidar_return1_dir = context_dir / "lidar" / "1" / "return1"
        lidar_return2_dir = context_dir / "lidar" / "1" / "return2"

        camera_count = len(list(camera_dir.glob("*.pkl")))
        return1_count = len(list(lidar_return1_dir.glob("*.pkl")))
        return2_count = len(list(lidar_return2_dir.glob("*.pkl")))

        results.append({
            "context": context,
            "expected_frames": expected_frames,
            "camera_frames": camera_count,
            "lidar_return1_frames": return1_count,
            "lidar_return2_frames": return2_count,
            "output_dir": str(context_dir),
        })

        print(f"Camera frames:      {camera_count}")
        print(f"LiDAR return 1:     {return1_count}")
        print(f"LiDAR return 2:     {return2_count}")

        if not (
            camera_count == return1_count == return2_count
        ):
            raise RuntimeError(
                f"Frame count mismatch for {context}"
            )

    print(f"\n{'=' * 70}")
    print("VALIDATION CONVERSION SUMMARY")
    print(f"{'=' * 70}")

    total_frames = 0

    for result in results:
        print(
            f"{result['context']}: "
            f"{result['camera_frames']}/{result['expected_frames']} frames"
        )
        total_frames += result["camera_frames"]

    print(f"\nTotal converted frames: {total_frames}")
    print("Expected total frames: 991")

    if total_frames != 991:
        print("WARNING: Converted total differs from expected total.")