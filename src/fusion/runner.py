
from pathlib import Path
import csv
import time

import numpy as np
import cv2

from src.data.waymo_fusion_data import (
    ensure_converted_context,
    load_camera_calibration,
    load_fusion_frame,
)
from src.models.pointpillars import predict_lidar
from src.models.camera_detector import predict_camera
from src.fusion.projection import (
    project_lidar_box,
    load_camera_geometry,
)
from src.fusion.matching import merge_camera_boxes
from src.fusion.association import associate_detections


PROJECT_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/3DPerception/project"
)
RESULTS_DIR = PROJECT_ROOT / "experiments" / "experiment_1_results"


def _finite_points(points, cfg):
    """Remove non-finite points and apply the configured XY ROI if available."""
    points = np.asarray(points, dtype=np.float32)
    valid = np.isfinite(points[:, :4]).all(axis=1)
    points = points[valid]

    # Apply the configured ROI only if its structure is recognized.
    # The detector may also apply its own ROI filtering.
    model_cfg = cfg.get("model", {})
    roi = model_cfg.get("point_cloud_range", cfg.get("point_cloud_range"))

    if roi is not None and len(roi) >= 5:
        x_min, y_min, _, x_max, y_max = (
            float(roi[0]), float(roi[1]), float(roi[2]),
            float(roi[3]), float(roi[4])
        )
        inside = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        )
        points = points[inside]

    return points


def run_fusion_frame(
    frame_idx,
    detector,
    lidar_model,
    cfg,
    camera_model,
    context_dir=None,
    output_dir=RESULTS_DIR,
    camera_confidence=0.25,
    camera_merge_iou=0.80,
    match_iou_threshold=0.30,
    save_visualization=True,
):
    """Run the two detectors, projection and class-aware association on one frame."""
    start_total = time.perf_counter()

    if context_dir is None:
        context_dir = ensure_converted_context()

    calibration = load_camera_calibration(context_dir)
    geometry = load_camera_geometry(calibration)
    frame = load_fusion_frame(context_dir, frame_idx)

    image = frame["image"]
    points_raw = frame["points"]
    points_model = _finite_points(points_raw, cfg)

    # 1. LiDAR inference: the wrapper passes XYZI to PointPillars.
    t0 = time.perf_counter()
    lidar_detections = predict_lidar(
        detector, lidar_model, points_model, cfg
    )
    lidar_seconds = time.perf_counter() - t0

    lidar_boxes = []
    lidar_labels = []
    lidar_scores = []

    for detection in lidar_detections:
        box = np.asarray(detection["box"], dtype=float)
        if box.size != 7:
            raise ValueError(
                f"Expected 7 values in a LiDAR box, got {box.size}: {box}"
            )
        lidar_boxes.append(box)
        lidar_labels.append(int(detection["label"]))
        lidar_scores.append(float(detection["score"]))

    # 2. Camera inference.
    t0 = time.perf_counter()
    camera_output = predict_camera(
        camera_model, image, confidence=camera_confidence
    )
    camera_seconds = time.perf_counter() - t0

    camera_names = camera_output["names"]
    camera_boxes = camera_output["boxes"]
    camera_scores = camera_output["scores"]
    camera_labels = camera_output["labels"]

    # 3. Merge overlapping camera detections into candidates.
    camera_candidates = merge_camera_boxes(
        camera_boxes,
        camera_scores,
        camera_labels,
        camera_names,
        threshold=camera_merge_iou,
    )

    # 4. Project each predicted LiDAR box into the camera image.
    projected_boxes = []
    for box in lidar_boxes:
        projected_boxes.append(
            project_lidar_box(
                box,
                points_raw,  # Preserve all original point columns here.
                geometry["T_camera_vehicle"],
                geometry["K"],
                geometry["dist"],
                geometry["image_width"],
                geometry["image_height"],
            )
        )

    # 5. Class-aware, one-to-one camera–LiDAR association.
    class_names = cfg["model"]["class_names"]
    iou_matrix, matches = associate_detections(
        projected_boxes,
        lidar_labels,
        class_names,
        camera_candidates,
        threshold=match_iou_threshold,
    )

    # Evaluation counts (no changes to predictions or matching).
    num_projected = sum(box is not None for box in projected_boxes)
    num_eligible_pairs = int(np.count_nonzero(iou_matrix >= match_iou_threshold))
    matched_ious = [float(iou) for _, _, iou in matches]
    num_matched_lidar = len({i for i, _, _ in matches})
    num_matched_camera = len({j for _, j, _ in matches})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save one row per LiDAR detection.
    csv_path = output_dir / "experiment_1_detections.csv"
    rows = []
    matched_by_lidar = {i: (j, overlap) for i, j, overlap in matches}

    for i, box in enumerate(lidar_boxes):
        match = matched_by_lidar.get(i)
        candidate = camera_candidates[match[0]] if match else None
        overlap = match[1] if match else None
        projected = projected_boxes[i]

        rows.append({
            "frame_idx": frame_idx,
            "lidar_index": i,
            "lidar_class": class_names[lidar_labels[i]],
            "lidar_score": lidar_scores[i],
            "lidar_box_3d": box.tolist(),
            "projected_box_2d": projected,
            "matched_camera_candidate": match[0] if match else None,
            "camera_labels": candidate["labels"] if candidate else None,
            "camera_box_2d": candidate["box"] if candidate else None,
            "matched_iou": overlap,
            "camera_timestamp": frame["camera_timestamp"],
            "lidar1_timestamp": frame["lidar1_timestamp"],
            "lidar2_timestamp": frame["lidar2_timestamp"],
        })

    # Append so repeated one-frame tests don't overwrite earlier frames.
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys() if rows else [
            "frame_idx", "lidar_index", "lidar_class", "lidar_score",
            "lidar_box_3d", "projected_box_2d", "matched_camera_candidate",
            "camera_labels", "camera_box_2d", "matched_iou",
            "camera_timestamp", "lidar1_timestamp", "lidar2_timestamp"
        ])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    # 6. Optional visualization: projected LiDAR boxes in red,
    # camera candidate boxes in blue, matched pairs in green.
    image_path = None
    if save_visualization:
        vis = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        for box in camera_candidates:
            x1, y1, x2, y2 = map(int, box["box"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 2)

        matched_lidar = {i for i, _, _ in matches}
        for i, box in enumerate(projected_boxes):
            if box is None:
                continue
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if i in matched_lidar else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis, class_names[lidar_labels[i]],
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

        image_path = output_dir / f"frame_{frame_idx:03d}_fusion.jpg"
        cv2.imwrite(str(image_path), vis)

    total_seconds = time.perf_counter() - start_total

    return {
        "frame_idx": frame_idx,
        "num_points_raw": len(points_raw),
        "num_points_model": len(points_model),
        "num_lidar_detections": len(lidar_boxes),
        "num_camera_detections": len(camera_boxes),
        "num_camera_candidates": len(camera_candidates),
        "num_projected_boxes": num_projected,
        "mean_matched_iou": (
            float(np.mean([iou for _, _, iou in matches])
                  ) if matches else None
        ),
        "lidar_inference_seconds": lidar_seconds,
        "camera_inference_seconds": camera_seconds,
        "total_seconds": total_seconds,
        "csv_path": str(csv_path),
        "visualization_path": str(image_path) if image_path else None,
        "iou_matrix": iou_matrix,
        "matches": matches,
        "lidar_detections": lidar_detections,
        "camera_output": camera_output,
        "num_unprojected_boxes": len(lidar_boxes) - num_projected,
        "num_eligible_pairs": num_eligible_pairs,
        "num_unmatched_projected_boxes": num_projected - num_matched_lidar,
        "num_unmatched_camera_candidates": len(camera_candidates) - num_matched_camera,
        "matched_ious": matched_ious,
    }
