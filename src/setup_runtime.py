
"""Restore the runtime used by the original camera-LiDAR fusion experiment."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path("/content")
DRIVE = ROOT / "drive/MyDrive/Colab Notebooks/3DPerception"

TOOLKIT = ROOT / "WaymoOpenDatasetToolKit"
TOOLKIT_URL = "https://github.com/KushalBKusram/WaymoOpenDatasetToolKit.git"

PP_ENV = ROOT / "pp_env"
PP_REPO = ROOT / "pointpillars_pretrained"
PP_URL = "https://github.com/m-mcninch/pointpillars.git"

CONVERTER = ROOT / "waymo-open-dataset-v2-converter"
CONVERTER_ENV = ROOT / "waymo_converter_env"
CONVERTER_BACKUP = DRIVE / "fusion_workspace_backup/converter"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clone_if_missing(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"Reuse repository: {destination}")
        return
    run(["git", "clone", url, str(destination)])


def setup_drive() -> None:
    from google.colab import drive
    drive.mount("/content/drive")
    if not DRIVE.is_dir():
        raise FileNotFoundError(f"Project Drive directory not found: {DRIVE}")


def setup_pointpillars_environment():
    python_bin = PP_ENV / "bin" / "python"

    # Reuse the existing environment if it is valid.
    if python_bin.exists():
        result = subprocess.run(
            [str(python_bin), "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"Reusing PointPillars environment: {result.stdout.strip()}")
        else:
            raise RuntimeError(
                f"PointPillars Python exists but failed: {result.stderr}"
            )
    else:
        # Install Python only if the runtime does not already provide it.
        if shutil.which("python3.11") is None:
            raise RuntimeError(
                "Python 3.11 is unavailable in this Colab runtime. "
                "The original setup used Python 3.11; do not continue "
                "with an incompatible interpreter."
            )

        run(["python3.11", "-m", "venv", str(PP_ENV)])
        python_bin = PP_ENV / "bin" / "python"

    requirements = TOOLKIT / "requirements.txt"
    if not requirements.exists():
        raise FileNotFoundError(f"Missing requirements file: {requirements}")

    # Install only if this environment is missing required toolkit packages.
    check = subprocess.run(
        [
            str(python_bin), "-c",
            "import torch, yaml, cv2"
        ],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(python_bin), "-m", "pip", "install", "-r", str(requirements)])

    print("PointPillars environment ready:", python_bin)
    
def setup_converter_environment() -> None:
    # Restore the patched converter files saved by the original project.
    if not CONVERTER.exists():
        if not (CONVERTER_BACKUP / "waymo_v2_converter_fusion.py").is_file():
            raise FileNotFoundError(
                "The original patched converter was not found in the runtime "
                "or Drive backup. Restore it from the original project backup."
            )
        CONVERTER.mkdir(parents=True)
        for name in (
            "waymo_v2_converter_fusion.py",
            "fusion_test.ini",
            "fusion_contexts.txt",
            "requirements.txt",
        ):
            source = CONVERTER_BACKUP / name
            if source.is_file():
                shutil.copy2(source, CONVERTER / name)

    python = CONVERTER_ENV / "bin/python"
    if not python.is_file():
        # Reproduce the original Python 3.10 environment.
        run(["uv", "python", "install", "3.10.21"])
        run([
            "uv", "venv", "--python", "3.10.21",
            str(CONVERTER_ENV),
        ])

    requirements = CONVERTER / "requirements.txt"
    if not requirements.is_file():
        raise FileNotFoundError(requirements)

    # Check the API actually used by the original converter.
    check = subprocess.run(
        [
            str(python), "-c",
            "import tensorflow as tf; "
            "from waymo_open_dataset import v2"
        ],
        capture_output=True,
    )
    if check.returncode != 0:
        run(["uv", "pip", "install", "--python", str(python),
             "-r", str(requirements)])

    print("Converter Python:", subprocess.check_output(
        [str(python), "--version"], text=True
    ).strip())


def setup_pretrained_detector() -> None:
    clone_if_missing(PP_URL, PP_REPO)

    required = [
        PP_REPO / "model/pointpillars.py",
        PP_REPO / "model/anchors.py",
        PP_REPO / "model/prediction.py",
        PP_REPO / "config/pillars.yaml",
        PP_REPO / "checkpoints/pointpillars_waymo_40x30.pth",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Pretrained PointPillars files missing:\n" + "\n".join(missing)
        )


def setup_cache_links() -> None:
    cache_root = TOOLKIT / ".waymo_cache/training"
    components = ("lidar", "lidar_box", "lidar_calibration")

    for component in components:
        source_dir = DRIVE / "training" / component
        target_dir = cache_root / component

        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)

        target_dir.mkdir(parents=True, exist_ok=True)
        for source in source_dir.glob("*.parquet"):
            prefix = f"training_{component}_"
            if not source.name.startswith(prefix):
                continue

            target = target_dir / source.name.removeprefix(prefix)
            if target.is_symlink() and target.resolve() == source.resolve():
                continue
            if target.exists() or target.is_symlink():
                if target.is_dir() and not target.is_symlink():
                    raise IsADirectoryError(target)
                target.unlink()
            target.symlink_to(source.resolve())

        print(f"{component}: {len(list(target_dir.glob('*.parquet')))} cache links")

def restore_four_class_detector(toolkit_root):
    import sys
    import importlib
    from pathlib import Path

    toolkit_root = Path(toolkit_root).resolve()
    models_dir = toolkit_root / "models"
    original = models_dir / "pointpillars_detector.py"
    custom = models_dir / "pointpillars_detector_4class.py"
    init_file = models_dir / "__init__.py"

    if not original.is_file():
        raise FileNotFoundError(f"Missing original detector: {original}")

    if not custom.exists():
        source = original.read_text()

        replacements = {
            "@register_detector('PointPillarsDetector')":
                "@register_detector('PointPillarsDetector4Class')",
            "class PointPillarsDetector(BaseDetector):":
                "class PointPillarsDetector4Class(BaseDetector):",
        }

        for old, new in replacements.items():
            if old not in source:
                raise RuntimeError(f"Expected detector code not found: {old}")
            source = source.replace(old, new, 1)

        marker = "    _ANCHOR_SIZES: dict[int, tuple[float, float]] = {"
        if marker not in source:
            raise RuntimeError("Anchor dictionary not found.")

        source = source.replace(
            marker,
            marker + "\n        3: (0.08, 0.62),  # sign",
            1,
        )
        custom.write_text(source)

    import_line = "from . import pointpillars_detector_4class  # noqa: F401, E402"
    init_text = init_file.read_text()

    if "from . import pointpillars_detector_4class" not in init_text:
        init_file.write_text(init_text.rstrip() + "\n" + import_line + "\n")

    toolkit_path = str(toolkit_root)
    if toolkit_path not in sys.path:
        sys.path.insert(0, toolkit_path)

    importlib.invalidate_caches()
    importlib.import_module("models.pointpillars_detector_4class")

    print("Four-class PointPillars detector ready.")

def setup_runtime() -> None:
    setup_drive()
    clone_if_missing(TOOLKIT_URL, TOOLKIT)
    setup_pointpillars_environment()
    setup_converter_environment()
    setup_pretrained_detector()
    setup_cache_links()
    restore_four_class_detector(TOOLKIT)

    print("\nRuntime setup completed.")
    print("Project source should be uploaded to /content/src from VS Code.")
    print("No training, conversion, or inference was started.")


if __name__ == "__main__":
    setup_runtime()
