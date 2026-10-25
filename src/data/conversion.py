"""Waymo v2 conversion pipeline shared by Experiments 2 and 3."""

from __future__ import annotations

import configparser
import subprocess
from pathlib import Path

DEFAULT_DATASET_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/3DPerception"
)
DEFAULT_OUTPUT_ROOT = Path("/content/waymo_fusion_converted")
DEFAULT_WORK_DIR = Path("/content/waymo_fusion_converter_work")
CONVERTER_REPO = Path("/content/waymo-open-dataset-v2-converter")
CONVERTER_PYTHON = Path("/content/waymo_converter_env/bin/python")
CONVERTER_SCRIPT = CONVERTER_REPO / "waymo_v2_converter_fusion.py"

DEFAULT_DATATYPES = {
    "vehicle_pose": True,
    "lidar": True,
    "lidar_calibration": True,
    "lidar_box": False,
    "camera_image": True,
    "camera_calibration": True,
    "camera_box": False,
    "camera_segmentation": False,
}


def prepare_conversion(
    contexts: list[str],
    *,
    split: str = "training",
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    work_dir: str | Path = DEFAULT_WORK_DIR,
    datatypes: dict[str, bool] | None = None,
) -> tuple[Path, Path]:
    """Write the converter configuration and list of requested contexts."""
    if split not in {"training", "validation", "testing"}:
        raise ValueError(f"Unsupported split: {split}")

    contexts = [context.strip() for context in contexts]
    if not contexts or any(not context for context in contexts):
        raise ValueError("At least one non-empty context is required.")
    if len(contexts) != len(set(contexts)):
        raise ValueError("Context list contains duplicates.")

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    work_dir = Path(work_dir)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    work_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    selected = DEFAULT_DATATYPES.copy()
    if datatypes is not None:
        unknown = set(datatypes) - set(selected)
        if unknown:
            raise ValueError(f"Unknown converter datatypes: {sorted(unknown)}")
        selected.update(datatypes)

    config = configparser.ConfigParser()
    config["Directories"] = {
        "dataset_dir": str(dataset_root),
        "output_dir": str(output_root),
        "split": split,
    }
    config["Datatypes"] = {
        name: "1" if enabled else "0"
        for name, enabled in selected.items()
    }

    config_path = work_dir / f"{split}_converter.ini"
    contexts_path = work_dir / f"{split}_contexts.txt"

    with config_path.open("w", encoding="utf-8") as file:
        config.write(file)

    contexts_path.write_text(
        "\n".join(contexts) + "\n",
        encoding="utf-8",
    )

    return config_path, contexts_path


def run_conversion(
    config_path: str | Path,
    contexts_path: str | Path,
) -> None:
    """Run conversion in its dedicated environment, isolated from the notebook."""
    config_path = Path(config_path)
    contexts_path = Path(contexts_path)

    for path in (
        CONVERTER_PYTHON,
        CONVERTER_SCRIPT,
        config_path,
        contexts_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required conversion file missing: {path}")

    command = [
        str(CONVERTER_PYTHON),
        "-u",
        str(CONVERTER_SCRIPT),
        "--cfg",
        str(config_path),
        "--contexts",
        str(contexts_path),
    ]

    log_path = config_path.with_suffix(".log")
    print(f"Conversion log: {log_path}", flush=True)

    # Stream output to disk instead of accumulating a potentially large log.
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=CONVERTER_REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(
            f"Converter exited with code {result.returncode}.\n"
            f"Last log lines:\n{tail}"
        )

    print("Converter process completed.", flush=True)


def validate_converted_context(
    context: str,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Check the expected camera, LiDAR, and calibration outputs."""
    context_dir = Path(output_root) / context

    required = (
        context_dir / "camera_image" / "1",
        context_dir / "lidar" / "1" / "return1",
        context_dir / "lidar" / "1" / "return2",
        context_dir / "camera_calibration" / "calibration.pkl",
    )
    missing = [path for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Converted context is incomplete:\n"
            + "\n".join(map(str, missing))
        )

    camera_frames = list(required[0].glob("*.pkl"))
    lidar_return1_frames = list(required[1].glob("*.pkl"))
    lidar_return2_frames = list(required[2].glob("*.pkl"))

    if not camera_frames or not lidar_return1_frames or not lidar_return2_frames:
        raise RuntimeError("One or more converted frame directories are empty.")

    print(f"Converted context: {context_dir}")
    print(f"Camera frames: {len(camera_frames)}")
    print(f"LiDAR return 1 frames: {len(lidar_return1_frames)}")
    print(f"LiDAR return 2 frames: {len(lidar_return2_frames)}")

    return context_dir


def convert_contexts(
    contexts: list[str],
    *,
    split: str = "training",
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    work_dir: str | Path = DEFAULT_WORK_DIR,
    datatypes: dict[str, bool] | None = None,
) -> list[Path]:
    """Prepare, convert, and validate one or more Waymo contexts."""
    config_path, contexts_path = prepare_conversion(
        contexts,
        split=split,
        dataset_root=dataset_root,
        output_root=output_root,
        work_dir=work_dir,
        datatypes=datatypes,
    )

    run_conversion(config_path, contexts_path)

    return [
        validate_converted_context(context, output_root=output_root)
        for context in contexts
    ]