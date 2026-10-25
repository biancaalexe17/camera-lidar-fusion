
"""External pretrained PointPillars model for Experiments 2 and 3.

This module is intentionally separate from pointpillars.py,
which remains unchanged for Experiment 1.
"""

from pathlib import Path
import sys

import numpy as np
import torch
import yaml


PRETRAINED_ROOT = Path("/content/pointpillars_pretrained")
CONFIG_PATH = PRETRAINED_ROOT / "config/pillars.yaml"
CHECKPOINT_PATH = (
    PRETRAINED_ROOT / "checkpoints/pointpillars_waymo_40x30.pth"
)


def load_pretrained_pointpillars(device=None):
    """Load the external pretrained model without training."""
    required = [
        PRETRAINED_ROOT / "model/pointpillars.py",
        PRETRAINED_ROOT / "model/anchors.py",
        PRETRAINED_ROOT / "model/prediction.py",
        CONFIG_PATH,
        CHECKPOINT_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing pretrained PointPillars files:\n"
            + "\n".join(missing)
        )

    if str(PRETRAINED_ROOT) not in sys.path:
        sys.path.insert(0, str(PRETRAINED_ROOT))

    from model.pointpillars import PointPillars
    from model.anchors import create_axis_aligned_anchors

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model_cfg = cfg["model"]
    grid = model_cfg["grid"]
    features = model_cfg["features"]
    point_cfg = model_cfg["points"]
    classes = model_cfg["detection"]["classes"]

    pillar_size = grid["pillar_size"]
    stride = model_cfg["strides"][0]

    for anchor in cfg["anchors"]:
        anchor["rotation"] = torch.as_tensor(
            anchor["rotation"], dtype=torch.float32, device=device
        )

    grid_dim = [
        int(
            (grid["y_max"] - grid["y_min"])
            / (pillar_size * stride)
        ),
        int(
            (grid["x_max"] - grid["x_min"])
            / (pillar_size * stride)
        ),
    ]

    anchors, anchor_class_ids, _, _ = create_axis_aligned_anchors(
        pillar_size=pillar_size * stride,
        x_min=grid["x_min"],
        y_min=grid["y_min"],
        grid_dim=grid_dim,
        anchor_configs=cfg["anchors"],
        anchor_thresholds=cfg["anchor_thresholds"],
        device=device,
    )

    model = PointPillars(
        n_input_features=features["n_input_features"],
        n_classes=len(classes),
        n_anchors=len(anchor_class_ids),
        x_min=grid["x_min"],
        x_max=grid["x_max"],
        y_min=grid["y_min"],
        y_max=grid["y_max"],
        pillar_size=pillar_size,
        max_points_pillar=point_cfg["max_points_pillar"],
        max_pillars=point_cfg["max_pillars"],
        strides=model_cfg["strides"],
        pillar_subsample_block_size=point_cfg["subsample_block_size"],
        include_intensity=features["intensity"],
        include_elongation=features["elongation"],
        intensity_norm=model_cfg["normalization"]["intensity"],
        elongation_norm=model_cfg["normalization"]["elongation"],
        device=device,
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "The pretrained checkpoint has no 'model_state_dict'."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    inference = cfg["inference"]
    runtime = {
        "anchors": anchors,
        "conf_dict": dict(enumerate(inference["conf"])),
        "nms_dict": dict(enumerate(inference["nms"])),
        "classes": classes,
        "device": device,
    }

    print(f"Loaded pretrained checkpoint: {CHECKPOINT_PATH}")
    print(f"Device: {device}")
    print(f"Classes: {classes}")

    return model, runtime, cfg


def _prepare_points(points, cfg):
    """Filter finite points and apply the pretrained model's spatial limits."""
    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] < 5:
        raise ValueError(
            "Expected an (N, >=5) array with "
            "[x, y, z, intensity, elongation]. "
            f"Received shape {points.shape}."
        )

    points = points[:, :5]
    points = points[np.isfinite(points).all(axis=1)]

    grid = cfg["model"]["grid"]
    inside = (
        (points[:, 0] >= grid["x_min"])
        & (points[:, 0] <= grid["x_max"])
        & (points[:, 1] >= grid["y_min"])
        & (points[:, 1] <= grid["y_max"])
    )

    z_min = grid.get("z_min")
    z_max = grid.get("z_max")

    if z_min is not None:
        inside &= points[:, 2] >= float(z_min)
    if z_max is not None:
        inside &= points[:, 2] <= float(z_max)
    
    return points[inside]


def _estimate_vertical_extent(bev_box, points):
    """Estimate z-centre and height from points inside a BEV box.

    These estimates are not produced by the pretrained model.
    """
    cx, cy, dx, dy, heading = map(float, bev_box)
    if dx <= 0 or dy <= 0:
        return None

    xyz = points[:, :3]
    delta_x = xyz[:, 0] - cx
    delta_y = xyz[:, 1] - cy

    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    local_x = cos_h * delta_x + sin_h * delta_y
    local_y = -sin_h * delta_x + cos_h * delta_y

    inside = (
        (np.abs(local_x) <= dx / 2)
        & (np.abs(local_y) <= dy / 2)
    )
    z_values = xyz[inside, 2]

    if len(z_values) < 3:
        return None

    z_min, z_max = np.percentile(z_values, [5, 95])
    height = float(z_max - z_min)

    if not np.isfinite(height) or height <= 0:
        return None

    return float((z_min + z_max) / 2), height


def predict_pretrained_lidar(model, runtime, points, cfg):
    """Return detections in the fusion runner's 7-value box format."""
    from model.prediction import decode_predictions, nms

    points = _prepare_points(points, cfg)
    if len(points) == 0:
        return []

    device = runtime["device"]

    # The external model expects:
    # [batch_index, x, y, z, intensity, elongation].
    model_points = np.column_stack(
        [np.zeros(len(points), dtype=np.float32), points]
    )
    model_points = torch.as_tensor(
        model_points, dtype=torch.float32, device=device
    )

    with torch.no_grad():
        raw_predictions = model(model_points, 1)

        boxes, scores, labels = decode_predictions(
            device,
            raw_predictions,
            runtime["anchors"],
            runtime["conf_dict"],
            1,
            len(runtime["classes"]),
        )
        boxes, scores, labels = nms(
            device,
            boxes,
            scores,
            labels,
            runtime["nms_dict"],
        )

    boxes = boxes[0].detach().cpu().numpy()
    scores = scores[0].detach().cpu().numpy()
    labels = labels[0].detach().cpu().numpy().astype(int)

    detections = []
    for bev_box, score, label in zip(boxes, scores, labels):
        if len(bev_box) != 5:
            raise ValueError(
                "Expected BEV box [x, y, dx, dy, heading], "
                f"received {bev_box}."
            )

        vertical = _estimate_vertical_extent(bev_box, points)
        if vertical is None:
            continue

        cx, cy, dx, dy, heading = map(float, bev_box)
        cz, dz = vertical

        detections.append({
            "box": [cx, cy, cz, dx, dy, dz, heading],
            "score": float(score),
            "label": int(label),
        })

    return detections
