
from pathlib import Path
import csv
import time

import cv2
import numpy as np

from src.data.waymo_fusion_data import (
    ensure_converted_context,
    load_camera_calibration,
    load_fusion_frame,
    available_frame_indices,
)
from src.models.camera_detector import (
    load_camera_detector,
    predict_camera,
)
from src.models.pointpillars_pretrained import (
    load_pretrained_pointpillars,
    predict_pretrained_lidar,
)
from src.fusion.projection import (
    load_camera_geometry,
    project_lidar_box,
)
from src.fusion.matching import merge_camera_boxes
from src.fusion.association import associate_detections


PROJECT_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/3DPerception/project"
)
RESULTS_DIR = PROJECT_ROOT / "experiments" / "experiment_2_results"

# Start with one frame before expanding the experiment.
MAX_FRAMES = 198
CAMERA_CONFIDENCE = 0.25
CAMERA_MERGE_IOU = 0.80
MATCH_IOU_THRESHOLD = 0.30
LIDAR_CONFIDENCE = 0.30
SAVE_VISUALIZATIONS = True


def finite_points(points):
    """Keep finite points while preserving all original point columns."""
    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(
            f"Expected a 2D point array with at least 4 columns; "
            f"received shape {points.shape}"
        )

    valid = np.isfinite(points[:, :4]).all(axis=1)
    return points[valid]


def draw_results(image, projected_boxes, lidar_labels, lidar_scores,
                 class_names, camera_candidates, matches, frame_idx):
    """Draw projected LiDAR detections and associated camera candidates."""
    canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    matched_by_lidar = {
        lidar_idx: (camera_idx, iou)
        for lidar_idx, camera_idx, iou in matches
    }

    for i, box in enumerate(projected_boxes):
        if box is None:
            continue

        x1, y1, x2, y2 = np.round(box).astype(int)
        name = class_names[int(lidar_labels[i])]
        score = float(lidar_scores[i])

        if i in matched_by_lidar:
            camera_idx, iou = matched_by_lidar[i]
            color = (0, 200, 0)
            text = f"{name} {score:.2f} | match IoU {iou:.2f}"
        else:
            color = (0, 165, 255)
            text = f"{name} {score:.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas, text, (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            cv2.LINE_AA,
        )

    # Mark camera candidates that did not match a LiDAR detection.
    matched_camera = {j for _, j, _ in matches}
    for j, candidate in enumerate(camera_candidates):
        if j in matched_camera:
            continue

        x1, y1, x2, y2 = np.round(candidate["box"]).astype(int)
        labels = ", ".join(candidate["labels"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(
            canvas, f"Camera only: {labels}",
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas, f"Experiment 2 | frame {frame_idx}",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return canvas


def run_frame(frame_idx, context_dir, geometry, lidar_model,
              lidar_runtime, lidar_cfg, camera_model):
    """Run pretrained LiDAR inference, camera inference and association."""
    start = time.perf_counter()

    frame = load_fusion_frame(context_dir, frame_idx)
    image = frame["image"]
    points_raw = finite_points(frame["points"])

    print(f"\nFrame {frame_idx}: {len(points_raw):,} LiDAR points")

    # 1. External pretrained PointPillars inference.
    lidar_detections = predict_pretrained_lidar(
        lidar_model, lidar_runtime, points_raw, lidar_cfg
    )

    lidar_boxes = []
    lidar_labels = []
    lidar_scores = []

    for detection in lidar_detections:
        box = np.asarray(detection["box"], dtype=float).reshape(-1)

        if box.size != 7:
            raise ValueError(
                "predict_pretrained_lidar must return 7-value boxes "
                "[x, y, z, dx, dy, dz, heading]. "
                f"Received {box.size} values: {box}"
            )

        lidar_boxes.append(box)
        lidar_labels.append(int(detection["label"]))
        lidar_scores.append(float(detection["score"]))

    lidar_labels = np.asarray(lidar_labels, dtype=int)
    lidar_scores = np.asarray(lidar_scores, dtype=float)

    # 2. Camera inference using the existing camera detector.
    camera_output = predict_camera(
        camera_model, image, confidence=CAMERA_CONFIDENCE
    )

    camera_candidates = merge_camera_boxes(
        camera_output["boxes"],
        camera_output["scores"],
        camera_output["labels"],
        camera_output["names"],
        threshold=CAMERA_MERGE_IOU,
    )

    # 3. Project LiDAR boxes into the camera image.
    projected_boxes = []
    for box in lidar_boxes:
        projected = project_lidar_box(
            box,
            points_raw,
            geometry["T_camera_vehicle"],
            geometry["K"],
            geometry["dist"],
            geometry["image_width"],
            geometry["image_height"],
        )
        projected_boxes.append(projected)

    # 4. Match projected LiDAR detections to camera candidates.
    # Class names must follow the external checkpoint's label ordering.
    class_names = lidar_cfg["model"]["detection"]["classes"]
    if len(class_names) <= max(lidar_labels, default=-1):
        raise ValueError(
            "A predicted LiDAR label is outside lidar_cfg['model']"
            "['class_names']. Verify the external model's class mapping."
        )

    iou_matrix, matches = associate_detections(
        projected_boxes,
        lidar_labels,
        class_names,
        camera_candidates,
        threshold=MATCH_IOU_THRESHOLD,
    )

    # 5. Save detection-level results.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "detections.csv"

    matched_by_lidar = {
        i: (j, iou) for i, j, iou in matches
    }

    rows = []
    for i, box in enumerate(lidar_boxes):
        match = matched_by_lidar.get(i)
        candidate = camera_candidates[match[0]] if match else None

        rows.append({
            "frame_idx": frame_idx,
            "lidar_class": class_names[int(lidar_labels[i])],
            "lidar_score": float(lidar_scores[i]),
            "x": float(box[0]),
            "y": float(box[1]),
            "z": float(box[2]),
            "dx": float(box[3]),
            "dy": float(box[4]),
            "dz": float(box[5]),
            "heading": float(box[6]),
            "projected_camera_box": (
                projected_boxes[i]
                if projected_boxes[i] is not None else None
            ),
            "matched_camera_labels": (
                ", ".join(candidate["labels"]) if candidate else ""
            ),
            "matched_camera_score": (
                candidate["score"] if candidate else None
            ),
            "match_iou": float(match[1]) if match else None,
            "matched": match is not None,
        })

    fieldnames = [
        "frame_idx", "lidar_class", "lidar_score",
        "x", "y", "z", "dx", "dy", "dz", "heading",
        "projected_camera_box", "matched_camera_labels",
        "matched_camera_score", "match_iou", "matched",
    ]

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    # 6. Save an image with projected boxes and associations.
    if SAVE_VISUALIZATIONS:
        visual = draw_results(
            image,
            projected_boxes,
            lidar_labels,
            lidar_scores,
            class_names,
            camera_candidates,
            matches,
            frame_idx,
        )
        image_path = RESULTS_DIR / f"frame_{frame_idx:03d}.jpg"
        if not cv2.imwrite(str(image_path), visual):
            raise OSError(f"Could not save visualization: {image_path}")
        print(f"Visualization: {image_path}")

    elapsed = time.perf_counter() - start
    print(f"LiDAR detections: {len(lidar_detections)}")
    print(f"Camera candidates: {len(camera_candidates)}")
    print(f"Projected LiDAR boxes: {sum(b is not None for b in projected_boxes)}")
    print(f"Matched pairs: {len(matches)}")
    print(f"Saved results: {csv_path}")
    print(f"Frame processing time: {elapsed:.2f} s")

    return {
        "frame_idx": frame_idx,
        "num_lidar": len(lidar_detections),
        "num_camera": len(camera_candidates),
        "num_projected": sum(b is not None for b in projected_boxes),
        "num_matches": len(matches),
        "iou_matrix": iou_matrix,
    }


def main(camera_model):
    """Run Experiment 2 on the first available converted frame(s)."""
    context_dir = ensure_converted_context()
    calibration = load_camera_calibration(context_dir)
    geometry = load_camera_geometry(calibration)

    # This loader must use the external pretrained checkpoint, not
    # Experiment 1's custom checkpoint.
    lidar_model, lidar_runtime, lidar_cfg = load_pretrained_pointpillars()

    print("\nExperiment 2 initialized")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"LiDAR class names: {lidar_cfg['model']['detection']['classes']}")
    frame_indices = available_frame_indices(context_dir)[:MAX_FRAMES]
    summaries = []

    for frame_idx in frame_indices:
        summary = run_frame(
            frame_idx=frame_idx,
            context_dir=context_dir,
            geometry=geometry,
            lidar_model=lidar_model,
            lidar_runtime=lidar_runtime,
            lidar_cfg=lidar_cfg,
            camera_model=camera_model,
        )
        summaries.append(summary)

    print("\nExperiment 2 summary")
    for summary in summaries:
        print(
            f"Frame {summary['frame_idx']}: "
            f"LiDAR={summary['num_lidar']}, "
            f"camera={summary['num_camera']}, "
            f"projected={summary['num_projected']}, "
            f"matches={summary['num_matches']}"
        )

    print("\nExperiment 2 finished.")
    return summaries


if __name__ == "__main__":
    camera_model = load_camera_detector()
    main(camera_model)
