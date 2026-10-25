
from pathlib import Path
import sys
import torch
import yaml

TOOLKIT_ROOT = Path("/content/WaymoOpenDatasetToolKit")
PROJECT_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/3DPerception/project")
CONFIG_PATH = PROJECT_ROOT / "configs/pointpillars_30seg_4class.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "30seg_4class/checkpoints/latest.pt"

def load_pointpillars(device=None):
    """Load the trained four-class detector without retraining."""
    if not TOOLKIT_ROOT.exists():
        raise FileNotFoundError(f"Missing toolkit: {TOOLKIT_ROOT}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}")

    if str(TOOLKIT_ROOT) not in sys.path:
        sys.path.insert(0, str(TOOLKIT_ROOT))

    # Import the custom detector so its class is registered.
    import models
    from models import pointpillars_detector_4class  # noqa: F401

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    saved_model = checkpoint.get("model")
    if not isinstance(saved_model, torch.nn.Module):
        raise TypeError(
            "Expected checkpoint['model'] to contain the trained model object."
        )

    model = saved_model.to(device).eval()
    detector = models.build_detector(cfg)

    print(f"Loaded checkpoint from segment {checkpoint.get('seg', '?')}")
    print(f"Device: {device}")
    print(f"Classes: {cfg['model']['class_names']}")

    return detector, model, cfg, device

def predict_lidar(detector, model, points, cfg):
    """Predict boxes from a single (N, 4) XYZI point cloud."""
    import numpy as np

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"Expected points shaped (N, >=4), got {points.shape}")

    return detector.predict(model, points[:, :4], cfg)
