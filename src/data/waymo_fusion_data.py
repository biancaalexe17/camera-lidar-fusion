
from pathlib import Path
import pickle
import shutil
import io

import numpy as np
from PIL import Image

CONTEXT_NAME = "10017090168044687777_6380_000_6400_000"

CONVERTED_ROOT = Path("/content/waymo_fusion_converted")
BACKUP_ROOT = (
    Path("/content/drive/MyDrive/Colab Notebooks/3DPerception")
    / "fusion_workspace_backup/converted_context"
)

def ensure_converted_context():
    """Restore the converted context from Drive if it is missing locally."""
    local_context = CONVERTED_ROOT / CONTEXT_NAME
    backup_context = BACKUP_ROOT / CONTEXT_NAME

    if not local_context.exists():
        if not backup_context.exists():
            raise FileNotFoundError(
                f"Neither the local context nor its Drive backup exists:\n"
                f"Local: {local_context}\nBackup: {backup_context}"
            )
        CONVERTED_ROOT.mkdir(parents=True, exist_ok=True)
        shutil.copytree(backup_context, local_context)

    required = [
        local_context / "camera_image/1",
        local_context / "lidar/1/return1",
        local_context / "lidar/1/return2",
        local_context / "camera_calibration/calibration.pkl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The converted context is incomplete:\n" + "\n".join(missing)
        )

    print(f"Converted context ready: {local_context}")
    return local_context


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def load_camera_calibration(context_dir):
    calibration_data = load_pickle(
        context_dir / "camera_calibration/calibration.pkl"
    )

    # This dataset's calibration was previously accessed using key 1.
    if isinstance(calibration_data, dict) and 1 in calibration_data:
        calibration = calibration_data[1]
    elif isinstance(calibration_data, (list, tuple)) and len(calibration_data) > 1:
        calibration = calibration_data[1]
    else:
        calibration = calibration_data

    if not isinstance(calibration, dict):
        raise TypeError(
            f"Unexpected calibration structure: {type(calibration_data)}"
        )

    return calibration


def available_frame_indices(context_dir):
    """Return camera frame indices that also have both LiDAR returns."""
    camera_dir = context_dir / "camera_image/1"
    lidar1_dir = context_dir / "lidar/1/return1"
    lidar2_dir = context_dir / "lidar/1/return2"

    def indices(folder):
        return {
            int(path.stem)
            for path in folder.glob("*.pkl")
            if path.stem.isdigit()
        }

    common = indices(camera_dir) & indices(lidar1_dir) & indices(lidar2_dir)
    result = sorted(common)

    if not result:
        raise RuntimeError("No matching camera and LiDAR frame indices found.")

    print(f"Frames with camera + both LiDAR returns: {len(result)}")
    print(f"First available frame indices: {result[:10]}")
    return result


def load_fusion_frame(context_dir, frame_idx):
    """Load one camera image and combine both raw LiDAR returns."""
    filename = f"{frame_idx:03d}.pkl"

    camera_data = load_pickle(
        context_dir / "camera_image/1" / filename
    )
    lidar1 = load_pickle(
        context_dir / "lidar/1/return1" / filename
    )
    lidar2 = load_pickle(
        context_dir / "lidar/1/return2" / filename
    )

    image_bytes = camera_data["image"]
    if isinstance(image_bytes, np.ndarray):
        image_bytes = image_bytes.tobytes()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    points1 = np.asarray(lidar1["points"], dtype=np.float32)
    points2 = np.asarray(lidar2["points"], dtype=np.float32)
    points = np.concatenate([points1, points2], axis=0)

    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(
            f"Unexpected LiDAR point shape for frame {frame_idx}: {points.shape}"
        )

    # Keep the full point array for projection; the model loader will
    # pass only the first four columns to PointPillars.
    return {
        "frame_idx": frame_idx,
        "image": image,
        "points": points,
        "camera_timestamp": camera_data.get("timestamp"),
        "lidar1_timestamp": lidar1.get("timestamp"),
        "lidar2_timestamp": lidar2.get("timestamp"),
    }
