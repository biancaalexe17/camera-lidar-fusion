"""Central paths and fixed validation metadata for the 3D perception project.

This module does not mount Google Drive or create directories automatically.
Mount Drive in a notebook first, then use PATHS and the helper functions.
"""

from dataclasses import dataclass
import os
from pathlib import Path


# This is the existing Google Drive project root used in the reference notebook.
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/Colab Notebooks/3DPerception"


@dataclass(frozen=True)
class ProjectPaths:
    """Paths used by the existing dataset/models and the new modular project."""

    drive_root: Path
    runtime_root: Path

    @property
    def training_data(self) -> Path:
        return self.drive_root / "training"

    @property
    def validation_data(self) -> Path:
        return self.drive_root / "validation"

    @property
    def custom_model_project(self) -> Path:
        return self.drive_root / "project"

    @property
    def custom_model_config(self) -> Path:
        return self.custom_model_project / "configs" / "pointpillars_30seg.yaml"

    @property
    def custom_model_checkpoint(self) -> Path:
        return self.custom_model_project / "30seg" / "checkpoints" / "latest.pt"

    @property
    def existing_backup(self) -> Path:
        return self.drive_root / "fusion_workspace_backup"

    @property
    def modular_project_root(self) -> Path:
        """Separate Drive folder for new code/configs/results; original files stay untouched."""
        return self.drive_root / "fusion_modular"

    @property
    def persistent_outputs(self) -> Path:
        return self.modular_project_root / "outputs"

    @property
    def experiment_1_outputs(self) -> Path:
        return self.persistent_outputs / "experiment_1"

    @property
    def experiment_2_outputs(self) -> Path:
        return self.persistent_outputs / "experiment_2"

    @property
    def validation_outputs(self) -> Path:
        return self.persistent_outputs / "validation"

    @property
    def runtime_cache(self) -> Path:
        return self.runtime_root / "waymo_local_cache"

    @property
    def runtime_validation_outputs(self) -> Path:
        """The existing notebook writes validation work here during a Colab session."""
        return self.runtime_root / "waymo_fusion_validation"

    @property
    def pretrained_checkpoint(self) -> Path | None:
        """Optional: set PRETRAINED_POINTPILLARS_CHECKPOINT after confirming its real path."""
        value = os.getenv("PRETRAINED_POINTPILLARS_CHECKPOINT")
        return Path(value).expanduser() if value else None


PATHS = ProjectPaths(
    drive_root=Path(os.getenv("PERCEPTION_DRIVE_ROOT", DEFAULT_DRIVE_ROOT)).expanduser(),
    runtime_root=Path(os.getenv("PERCEPTION_RUNTIME_ROOT", "/content")).expanduser(),
)


# Context IDs and frame counts copied from the reference notebook's consolidated
# evaluation cell (cell 233). We will verify the exact validation protocol later.
VALIDATION_CONTEXTS: tuple[tuple[str, int], ...] = (
    ("10203656353524179475_7625_000_7645_000", 198),
    ("1024360143612057520_3580_000_3600_000", 199),
    ("10247954040621004675_2180_000_2200_000", 198),
    ("10289507859301986274_4200_000_4220_000", 198),
    ("10335539493577748957_1372_870_1392_870", 198),
)


def ensure_output_dirs() -> None:
    """Create only the new modular project's output folders on mounted Drive."""
    for path in (
        PATHS.experiment_1_outputs,
        PATHS.experiment_2_outputs,
        PATHS.validation_outputs,
    ):
        path.mkdir(parents=True, exist_ok=True)


def check_reference_paths() -> dict[str, bool]:
    """Return existence checks for important existing files/directories.

    Call after mounting Google Drive. This function only checks; it never edits
    or creates anything in the original project.
    """
    expected = {
        "drive_root": PATHS.drive_root,
        "training_data": PATHS.training_data,
        "custom_model_config": PATHS.custom_model_config,
        "custom_model_checkpoint": PATHS.custom_model_checkpoint,
        "validation_data": PATHS.validation_data,
        "existing_backup": PATHS.existing_backup,
    }
    return {name: path.exists() for name, path in expected.items()}


def print_path_check() -> None:
    """Print a readable report and fail only if the main Drive root is absent."""
    checks = check_reference_paths()
    expected = {
        "drive_root": PATHS.drive_root,
        "training_data": PATHS.training_data,
        "custom_model_config": PATHS.custom_model_config,
        "custom_model_checkpoint": PATHS.custom_model_checkpoint,
        "validation_data": PATHS.validation_data,
        "existing_backup": PATHS.existing_backup,
    }
    for name, exists in checks.items():
        print(f"{'OK' if exists else 'MISSING':7} {name}: {expected[name]}")

    if not checks["drive_root"]:
        raise FileNotFoundError(
            f"Google Drive project root not found: {PATHS.drive_root}. "
            "Mount Drive first and verify PERCEPTION_DRIVE_ROOT."
        )
