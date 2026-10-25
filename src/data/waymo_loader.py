"""Loading helpers for the Waymo v2 Parquet files used by this project.

This module deliberately does not mount Google Drive, install dependencies, clone
repositories, or run model inference. Mount Drive and prepare the Waymo toolkit in
the notebook/runtime before calling the relevant functions.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.config import PATHS


# The reference notebook uses these names in the training folder.
_TOOLKIT_COMPONENTS = {"lidar", "lidar_box", "lidar_calibration"}
_CAMERA_COMPONENTS = {"camera_image", "camera_calibration"}


def component_parquet_path(
    component: str,
    segment: str,
    *,
    split: str = "training",
    data_root: Path | str | None = None,
) -> Path:
    """Return the Parquet path following the naming convention in the notebook.

    LiDAR/toolkit components use names such as
    ``training_lidar_<segment>.parquet``. Camera components use
    ``<segment>.parquet``.
    """
    if component not in _TOOLKIT_COMPONENTS | _CAMERA_COMPONENTS:
        raise ValueError(f"Unsupported Waymo component: {component!r}")
    if split not in {"training", "validation", "testing"}:
        raise ValueError(f"Unsupported split: {split!r}")

    root = Path(data_root) if data_root is not None else PATHS.drive_root
    split_dir = root / split
    filename = (
        f"{split}_{component}_{segment}.parquet"
        if component in _TOOLKIT_COMPONENTS
        else f"{segment}.parquet"
    )
    return split_dir / component / filename


def create_cache_links(
    components: Iterable[str],
    *,
    split: str = "training",
    data_root: Path | str | None = None,
    cache_root: Path | str | None = None,
) -> dict[str, int]:
    """Create runtime cache symlinks to existing Drive Parquet files.

    This mirrors the cache-link approach in original notebook cells 3 and 11.
    It writes only to the runtime cache, never into the source data directories.
    Existing matching symlinks are reused. A conflicting cache entry is replaced.
    """
    root = Path(data_root) if data_root is not None else PATHS.drive_root
    cache_base = Path(cache_root) if cache_root is not None else PATHS.runtime_cache
    results: dict[str, int] = {}

    for component in components:
        if component not in _TOOLKIT_COMPONENTS | _CAMERA_COMPONENTS:
            raise ValueError(f"Unsupported Waymo component: {component!r}")
        source_dir = root / split / component
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset directory: {source_dir}")

        target_dir = cache_base / split / component
        target_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for source in sorted(source_dir.glob("*.parquet")):
            if component in _TOOLKIT_COMPONENTS:
                prefix = f"{split}_{component}_"
                if not source.name.startswith(prefix):
                    continue
                target_name = source.name.removeprefix(prefix)
            else:
                target_name = source.name

            target = target_dir / target_name
            if target.is_symlink() and target.resolve() == source.resolve():
                count += 1
                continue
            if target.exists() or target.is_symlink():
                if target.is_dir() and not target.is_symlink():
                    raise IsADirectoryError(
                        f"Refusing to replace a directory with a cache link: {target}"
                    )
                target.unlink()
            target.symlink_to(source.resolve())
            count += 1
        results[component] = count

    return results


def load_camera_image(
    segment: str,
    camera_name: int = 1,
    *,
    split: str = "training",
    data_root: Path | str | None = None,
):
    """Load one RGB camera image from the selected Waymo camera Parquet file.

    Returns ``(image, dataframe_row)``. ``image`` is a PIL RGB image. The row is
    returned so callers can access timestamps or other synchronized metadata.
    """
    import pandas as pd
    from PIL import Image

    path = component_parquet_path(
        "camera_image", segment, split=split, data_root=data_root
    )
    if not path.is_file():
        raise FileNotFoundError(f"Camera Parquet file not found: {path}")

    camera_df = pd.read_parquet(path)
    camera_col = "key.camera_name"
    image_col = "[CameraImageComponent].image"
    missing = [name for name in (camera_col, image_col) if name not in camera_df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {path.name}: {missing}")

    rows = camera_df[camera_df[camera_col] == camera_name]
    if rows.empty:
        raise ValueError(f"Camera {camera_name} not found in {path.name}")

    row = rows.iloc[0]
    image = Image.open(io.BytesIO(row[image_col])).convert("RGB")
    return image, row


def load_lidar_point_sets(
    segment: str,
    timestamp: int,
    *,
    toolkit: Any | None = None,
    split: str = "training",
    save_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> list[np.ndarray]:
    """Load synchronized LiDAR point sets through the Waymo toolkit.

    The toolkit is injected by the caller so this module does not hard-code the
    external repository import or force its dependency environment on other code.
    """
    if toolkit is None:
        from modules.waymo_open_dataset import ToolKit

        toolkit = ToolKit(
            split=split,
            save_dir=str(save_dir or PATHS.runtime_root / "waymo_output"),
            cache_dir=str(cache_dir or PATHS.runtime_cache),
        )

    toolkit.assign_segment(segment)
    point_sets = toolkit.load_lidar_points_xyzi(int(timestamp))
    if point_sets is None or len(point_sets) == 0:
        raise ValueError(
            f"No LiDAR point clouds loaded for segment={segment}, timestamp={timestamp}"
        )
    return [np.asarray(points) for points in point_sets]


def load_top_lidar_points(
    segment: str,
    timestamp: int,
    *,
    toolkit: Any | None = None,
    split: str = "training",
    save_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> np.ndarray:
    """Load the first LiDAR point set, matching the reference notebook's TOP LiDAR."""
    point_sets = load_lidar_point_sets(
        segment,
        timestamp,
        toolkit=toolkit,
        split=split,
        save_dir=save_dir,
        cache_dir=cache_dir,
    )
    points_top = point_sets[0]
    if points_top.ndim != 2 or points_top.shape[1] < 4:
        raise ValueError(
            f"Unexpected TOP LiDAR shape: {points_top.shape}; expected (N, >=4)."
        )
    return points_top
