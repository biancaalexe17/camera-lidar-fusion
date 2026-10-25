
"""Reproducible multi-frame runner for Camera–LiDAR Fusion Experiment 1."""

import csv
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.config import PATHS
from src.data.waymo_fusion_data import (
    available_frame_indices,
    ensure_converted_context,
)
from src.fusion.runner import run_fusion_frame


# A new timestamped directory prevents mixing results from different runs.
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUTPUT_DIR = Path("/content/drive/MyDrive/Colab Notebooks/3DPerception/project/experiments/experiment_1_runs") / RUN_ID

# Set to 1 for the first test; use None to process all available frames.
MAX_FRAMES = None

CAMERA_CONFIDENCE = 0.25
CAMERA_MERGE_IOU = 0.80
MATCH_IOU_THRESHOLD = 0.30
SAVE_VISUALIZATIONS = True


def main(detector, lidar_model, cfg, camera_model):
    global RUN_ID, OUTPUT_DIR
    RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    OUTPUT_DIR = Path(
        "/content/drive/MyDrive/Colab Notebooks/3DPerception/project/experiments/experiment_1_runs"
    ) / RUN_ID
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    context_dir = ensure_converted_context()
    frame_indices = available_frame_indices(context_dir)

    if MAX_FRAMES is not None:
        frame_indices = frame_indices[:MAX_FRAMES]

    print(f"Run ID: {RUN_ID}")
    print(f"Context: {context_dir}")
    print(f"Frames selected: {len(frame_indices)}")
    print(f"Output: {OUTPUT_DIR}")

    config = {
        "run_id": RUN_ID,
        "context": str(context_dir),
        "frame_indices": frame_indices,
        "camera_confidence": CAMERA_CONFIDENCE,
        "camera_merge_iou": CAMERA_MERGE_IOU,
        "match_iou_threshold": MATCH_IOU_THRESHOLD,
        "save_visualizations": SAVE_VISUALIZATIONS,
        "max_frames": MAX_FRAMES,
    }
    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps(config, indent=2)
    )

    summary_rows = []
    match_rows = []
    failed_rows = []
    started = time.perf_counter()

    for position, frame_idx in enumerate(frame_indices, start=1):
        print(
            f"[{position}/{len(frame_indices)}] Frame {frame_idx}",
            flush=True,
        )
        try:
            result = run_fusion_frame(
                frame_idx=frame_idx,
                detector=detector,
                lidar_model=lidar_model,
                cfg=cfg,
                camera_model=camera_model,
                context_dir=context_dir,
                output_dir=OUTPUT_DIR,
                camera_confidence=CAMERA_CONFIDENCE,
                camera_merge_iou=CAMERA_MERGE_IOU,
                match_iou_threshold=MATCH_IOU_THRESHOLD,
                save_visualization=SAVE_VISUALIZATIONS,
            )

            matches = result["matches"]
            summary_rows.append({
                "frame_idx": frame_idx,
                "num_points_raw": result["num_points_raw"],
                "num_points_model": result["num_points_model"],
                "num_lidar_detections": result["num_lidar_detections"],
                "num_projected_boxes": result["num_projected_boxes"],
                "num_unprojected_boxes": result["num_unprojected_boxes"],
                "num_camera_detections": result["num_camera_detections"],
                "num_camera_candidates": result["num_camera_candidates"],
                "num_eligible_pairs": result["num_eligible_pairs"],
                "num_matches": len(matches),
                "num_unmatched_projected_boxes":
                    result["num_unmatched_projected_boxes"],
                "num_unmatched_camera_candidates":
                    result["num_unmatched_camera_candidates"],
                "mean_matched_iou": result["mean_matched_iou"],
                "lidar_inference_seconds": result["lidar_inference_seconds"],
                "camera_inference_seconds": result["camera_inference_seconds"],
                "total_seconds": result["total_seconds"],
            })

            for lidar_idx, camera_idx, iou in matches:
                match_rows.append({
                    "frame_idx": frame_idx,
                    "lidar_index": lidar_idx,
                    "camera_candidate_index": camera_idx,
                    "iou": float(iou),
                })

            print(
                f"  LiDAR={result['num_lidar_detections']}, "
                f"projected={result['num_projected_boxes']}, "
                f"camera candidates={result['num_camera_candidates']}, "
                f"matches={len(matches)}, "
                f"mean IoU={result['mean_matched_iou']}",
                flush=True,
            )

        except Exception as exc:
            failed_rows.append({
                "frame_idx": frame_idx,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"  FAILED: {exc}", flush=True)

    def write_csv(filename, rows):
        if not rows:
            print(f"No rows to write: {filename}")
            return
        path = OUTPUT_DIR / filename
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("fusion_summary.csv", summary_rows)
    write_csv("fusion_matched_pairs.csv", match_rows)
    write_csv("fusion_failed_frames.csv", failed_rows)

    elapsed = time.perf_counter() - started
    completed = len(summary_rows)
    total_matches = sum(row["num_matches"] for row in summary_rows)

    run_summary = {
        **config,
        "frames_requested": len(frame_indices),
        "frames_completed": completed,
        "frames_failed": len(failed_rows),
        "total_matches": total_matches,
        "mean_frame_matched_iou": (
            sum(
                row["mean_matched_iou"]
                for row in summary_rows
                if row["mean_matched_iou"] is not None
            )
            / sum(
                row["mean_matched_iou"] is not None
                for row in summary_rows
            )
            if any(
                row["mean_matched_iou"] is not None
                for row in summary_rows
            )
            else None
        ),
        "elapsed_seconds": elapsed,
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2)
    )

    print("\nExperiment run finished.")
    print(json.dumps(run_summary, indent=2))
    print(f"Results saved to: {OUTPUT_DIR}")


# Call main(detector, lidar_model, cfg, camera_model) after model initialization.
