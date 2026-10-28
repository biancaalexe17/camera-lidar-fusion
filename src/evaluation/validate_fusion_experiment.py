#!/usr/bin/env python3
"""
Validate the existing camera-supported filtering experiment against the
PointPillars baseline. This script does NOT run either model.

Outputs:
  bev_evaluation/fusion_validation_checks.txt
  bev_evaluation/fusion_validation_metrics_by_context.csv
  bev_evaluation/fusion_validation_metrics_overall.csv
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/3DPerception"
)
OUTPUT_ROOT = DRIVE_ROOT / "fusion_modular/outputs/validation"
PREDICTIONS_ROOT = OUTPUT_ROOT / "cleaned_predictions"
VALIDATION_ROOT = DRIVE_ROOT / "validation"
EVALUATION_ROOT = OUTPUT_ROOT / "bev_evaluation"

CONTEXTS = [
    "10203656353524179475_7625_000_7645_000",
    "1024360143612057520_3580_000_3600_000",
    "10247954040621004675_2180_000_2200_000",
    "10289507859301986274_4200_000_4220_000",
    "10335539493577748957_1372_870_1392_870",
]

IOU_THRESHOLD = 0.50
X_MIN, X_MAX = 0.0, 40.96
Y_MIN, Y_MAX = -15.36, 15.36

CLASS_MAPPING = {
    "pedestrian": 2,
    "cyclist": 4,
    "vehicle": 1,
}

# Support either location used in the existing Colab setup.
for candidate in [
    DRIVE_ROOT / "project",
    Path("/content"),
]:
    if (candidate / "src" / "evaluation" / "bev_metrics.py").exists():
        sys.path.insert(0, str(candidate))
        break

from src.evaluation.bev_metrics import evaluate_frame_boxes


def parse_matched(series):
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


def in_roi(df, x_col="x", y_col="y"):
    return (
        df[x_col].ge(X_MIN) & df[x_col].lt(X_MAX)
        & df[y_col].ge(Y_MIN) & df[y_col].lt(Y_MAX)
    )



def metrics_for_context(pred_df, gt_df, context_id):
    # Keep predictions inside the evaluation ROI.
    pred_df = pred_df[in_roi(pred_df)].copy()
    pred_df["lidar_class"] = pred_df["lidar_class"].astype(str).str.lower()
    pred_df["matched_bool"] = parse_matched(pred_df["matched"])

    gt_type = "[LiDARBoxComponent].type"
    gt_x = "[LiDARBoxComponent].box.center.x"
    gt_y = "[LiDARBoxComponent].box.center.y"
    gt_dx = "[LiDARBoxComponent].box.size.x"
    gt_dy = "[LiDARBoxComponent].box.size.y"
    gt_heading = "[LiDARBoxComponent].box.heading"
    gt_ts = "key.frame_timestamp_micros"

    required = [gt_type, gt_x, gt_y, gt_dx, gt_dy, gt_heading, gt_ts]
    missing = [col for col in required if col not in gt_df.columns]
    if missing:
        raise KeyError(f"{context_id}: missing GT columns: {missing}")

    timestamps = sorted(
        int(ts) for ts in gt_df[gt_ts].dropna().unique()
    )
    timestamp_to_frame = {ts: idx for idx, ts in enumerate(timestamps)}

    results = []
    checks = []

    for class_name, gt_class_id in CLASS_MAPPING.items():
        class_pred = pred_df[
            pred_df["lidar_class"].eq(class_name)
        ].copy()

        class_gt = gt_df[
            gt_df[gt_type].eq(gt_class_id)
            & in_roi(gt_df, gt_x, gt_y)
        ].copy()

        gt_by_frame = {}
        for ts, group in class_gt.groupby(gt_ts):
            frame_idx = timestamp_to_frame[int(ts)]
            gt_by_frame[frame_idx] = [
                [float(x), float(y), float(dx), float(dy), float(heading)]
                for x, y, dx, dy, heading in group[
                    [gt_x, gt_y, gt_dx, gt_dy, gt_heading]
                ].itertuples(index=False, name=None)
            ]

        baseline_by_frame = {}
        supported_by_frame = {}

        for frame_idx, group in class_pred.groupby("frame_idx"):
            frame_idx = int(frame_idx)
            baseline_by_frame[frame_idx] = [
                [float(r.x), float(r.y), float(r.dx), float(r.dy), float(r.heading)]
                for r in group.itertuples()
            ]

            matched_group = group[group["matched_bool"]]
            supported_by_frame[frame_idx] = [
                [float(r.x), float(r.y), float(r.dx), float(r.dy), float(r.heading)]
                for r in matched_group.itertuples()
            ]

        # Explicitly verify that each supported box corresponds to a baseline box.
        # Both sets are built from the same CSV, so use the original row indices
        # to verify the subset without relying on box-count comparisons.
        matched_indices = set(class_pred.index[class_pred["matched_bool"]])
        baseline_indices = set(class_pred.index)

        subset_ok = matched_indices.issubset(baseline_indices)

        def evaluate(predictions):
            tp = total_pred = total_gt = 0
            matched_ious, center_errors = [], []
            frame_ids = set(range(len(timestamps)))
            frame_ids |= set(predictions) | set(gt_by_frame)

            for frame_idx in frame_ids:
                pred_boxes = predictions.get(frame_idx, [])
                gt_boxes = gt_by_frame.get(frame_idx, [])
                total_pred += len(pred_boxes)
                total_gt += len(gt_boxes)

                matches = evaluate_frame_boxes(
                    pred_boxes, gt_boxes, IOU_THRESHOLD
                )
                tp += len(matches)

                for pred_idx, gt_idx, iou in matches:
                    matched_ious.append(float(iou))
                    center_errors.append(float(np.linalg.norm(
                        np.asarray(pred_boxes[pred_idx][:2], dtype=float)
                        - np.asarray(gt_boxes[gt_idx][:2], dtype=float)
                    )))

            precision = tp / total_pred if total_pred else 0.0
            recall = tp / total_gt if total_gt else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )

            return {
                "TP": tp,
                "Predictions": total_pred,
                "GT": total_gt,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Mean_BEV_IoU": (
                    float(np.mean(matched_ious)) if matched_ious else np.nan
                ),
                "Mean_center_error_m": (
                    float(np.mean(center_errors)) if center_errors else np.nan
                ),
            }

        baseline = evaluate(baseline_by_frame)
        supported = evaluate(supported_by_frame)

        results.extend([
            {"context_id": context_id, "class": class_name,
             "method": "PointPillars baseline", **baseline},
            {"context_id": context_id, "class": class_name,
             "method": "Camera-supported filtering", **supported},
        ])

        checks.append({
            "context_id": context_id,
            "class": class_name,
            "frames_in_gt": len(timestamps),
            "baseline_predictions": baseline["Predictions"],
            "camera_supported_predictions": supported["Predictions"],
            "matched_flag_rows": int(class_pred["matched_bool"].sum()),
            "ground_truth_boxes_in_roi": baseline["GT"],
            "camera_supported_subset_count_check": subset_ok,
            "matched_flag_count_agrees_with_supported_count": (
                int(class_pred["matched_bool"].sum())
                == supported["Predictions"]
            ),
        })

    return results, checks


def main():
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    per_context_rows = []
    checks = []

    overall_counts = {
        (class_name, method): {
            "TP": 0, "Predictions": 0, "GT": 0
        }
        for class_name in CLASS_MAPPING
        for method in [
            "PointPillars baseline",
            "Camera-supported filtering",
        ]
    }

    for context_id in CONTEXTS:
        pred_path = PREDICTIONS_ROOT / f"{context_id}_detections.csv"
        gt_path = VALIDATION_ROOT / "lidar_box" / f"{context_id}.parquet"
        summary_path = OUTPUT_ROOT / context_id / "frame_summary.csv"

        for path in (pred_path, gt_path, summary_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required existing output not found: {path}"
                )

        predictions = pd.read_csv(pred_path)
        ground_truth = pd.read_parquet(gt_path)
        summary = pd.read_csv(summary_path)

        context_results, context_checks = metrics_for_context(
            predictions, ground_truth, context_id
        )
        per_context_rows.extend(context_results)
        checks.extend(context_checks)

        # Check the saved association count against the original summary.
        csv_match_count = int(
            parse_matched(predictions["matched"]).sum()
        )
        summary_match_count = int(summary["num_matches"].sum())

        checks.append({
            "context_id": context_id,
            "class": "ALL",
            "csv_matched_pairs": csv_match_count,
            "summary_matched_pairs": summary_match_count,
            "matched_pair_count_agrees_with_summary":
                csv_match_count == summary_match_count,
        })

        for result in context_results:
            key = (result["class"], result["method"])
            for field in ("TP", "Predictions", "GT"):
                overall_counts[key][field] += result[field]

    # Aggregate counts across contexts separately for each class and method.
    overall_rows = []
    for class_name in CLASS_MAPPING:
        for method in [
            "PointPillars baseline",
            "Camera-supported filtering",
        ]:
            counts = overall_counts[(class_name, method)]
            tp = counts["TP"]
            n_pred = counts["Predictions"]
            n_gt = counts["GT"]

            precision = tp / n_pred if n_pred else 0.0
            recall = tp / n_gt if n_gt else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )

            overall_rows.append({
                "class": class_name,
                "method": method,
                "IoU_threshold": IOU_THRESHOLD,
                "TP": tp,
                "Predictions": n_pred,
                "GT": n_gt,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            })

    per_context_df = pd.DataFrame(per_context_rows)
    overall_df = pd.DataFrame(overall_rows)
    checks_df = pd.DataFrame(checks)

    per_context_path = (
        EVALUATION_ROOT / "fusion_validation_metrics_by_context.csv"
    )
    overall_path = (
        EVALUATION_ROOT / "fusion_validation_metrics_overall.csv"
    )
    report_path = EVALUATION_ROOT / "fusion_validation_checks.txt"

    per_context_df.to_csv(per_context_path, index=False)
    overall_df.to_csv(overall_path, index=False)

    lines = [
        "FUSION EXPERIMENT VALIDATION CHECKS",
        f"BEV IoU threshold: {IOU_THRESHOLD}",
        f"ROI: x=[{X_MIN}, {X_MAX}), y=[{Y_MIN}, {Y_MAX})",
        f"GT class mapping: {CLASS_MAPPING}",
        "Prediction classes: pedestrian, cyclist, vehicle",
        "Frame indexing: reset and evaluated independently per context",
        "",
        "Per-class checks:",
        checks_df.to_string(index=False),
        "",
        "Overall metrics (aggregated across contexts, per class):",
        overall_df.to_string(index=False),
        "",
        "This script evaluates existing files and does not run inference.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("OVERALL METRICS")
    print(overall_df.round(4).to_string(index=False))
    print("\nVALIDATION CHECKS")
    print(checks_df.to_string(index=False))
    print("\nSaved:")
    print(per_context_path)
    print(overall_path)
    print(report_path)


if __name__ == "__main__":
    main()
