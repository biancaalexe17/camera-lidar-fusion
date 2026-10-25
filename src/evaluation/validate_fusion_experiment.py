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
    # Keep only vehicle predictions in the same ROI used by the baseline.
    pred_df = pred_df[
        pred_df["lidar_class"].astype(str).str.lower().eq("vehicle")
        & in_roi(pred_df)
    ].copy()
    pred_df["matched_bool"] = parse_matched(pred_df["matched"])

    # The frame index is local to each context. Never combine frame 0 from
    # different contexts; each context is evaluated independently.
    timestamps = sorted(
        int(ts)
        for ts in gt_df["key.frame_timestamp_micros"].dropna().unique()
    )
    timestamp_to_frame = {ts: idx for idx, ts in enumerate(timestamps)}

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

    gt_vehicle = gt_df[
        gt_df[gt_type].eq(1) & in_roi(gt_df, gt_x, gt_y)
    ].copy()

    gt_by_frame = {}
    for ts, group in gt_vehicle.groupby(gt_ts):
        frame_idx = timestamp_to_frame[int(ts)]
        gt_by_frame[frame_idx] = [
            [float(x), float(y), float(dx), float(dy), float(heading)]
            for x, y, dx, dy, heading in group[
                [gt_x, gt_y, gt_dx, gt_dy, gt_heading]
            ].itertuples(index=False, name=None)
        ]

    baseline_by_frame = {}
    supported_by_frame = {}
    for frame_idx, group in pred_df.groupby("frame_idx"):
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

    # Verify the supported set is a subset of baseline by construction and
    # check the per-frame count invariant.
    subset_ok = True
    for frame_idx, boxes in supported_by_frame.items():
        if len(boxes) > len(baseline_by_frame.get(frame_idx, [])):
            subset_ok = False

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

    return evaluate(baseline_by_frame), evaluate(supported_by_frame), {
        "context_id": context_id,
        "frames_in_gt": len(timestamps),
        "baseline_vehicle_predictions": sum(map(len, baseline_by_frame.values())),
        "camera_supported_vehicle_predictions": sum(map(len, supported_by_frame.values())),
        "camera_supported_subset_count_check": subset_ok,
        "ground_truth_vehicle_boxes_in_roi": sum(map(len, gt_by_frame.values())),
        "matched_flag_rows_in_cleaned_csv": int(pred_df["matched_bool"].sum()),
    }


def main():
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    per_context_rows = []
    checks = []
    overall_counts = {
        "baseline": {"TP": 0, "Predictions": 0, "GT": 0, "ious": [], "errors": []},
        "camera_supported": {"TP": 0, "Predictions": 0, "GT": 0, "ious": [], "errors": []},
    }

    for context_id in CONTEXTS:
        pred_path = PREDICTIONS_ROOT / f"{context_id}_detections.csv"
        gt_path = VALIDATION_ROOT / "lidar_box" / f"{context_id}.parquet"
        summary_path = OUTPUT_ROOT / context_id / "frame_summary.csv"

        for path in (pred_path, gt_path, summary_path):
            if not path.exists():
                raise FileNotFoundError(f"Required existing output not found: {path}")

        predictions = pd.read_csv(pred_path)
        ground_truth = pd.read_parquet(gt_path)
        summary = pd.read_csv(summary_path)

        baseline, supported, check = metrics_for_context(
            predictions, ground_truth, context_id
        )

        # The saved frame summary counts matches across all classes, whereas
        # check["matched_flag_rows_in_cleaned_csv"] also includes all vehicle
        # predictions only after the vehicle filter. Do a separate all-class
        # count check directly from the CSV.
        all_predictions = pd.read_csv(pred_path)
        csv_match_count = int(parse_matched(all_predictions["matched"]).sum())
        summary_match_count = int(summary["num_matches"].sum())
        check["csv_matched_pairs"] = csv_match_count
        check["summary_matched_pairs"] = summary_match_count
        check["matched_pair_count_agrees_with_summary"] = (
            csv_match_count == summary_match_count
        )

        checks.append(check)
        for method, result in [
            ("PointPillars baseline", baseline),
            ("Camera-supported filtering", supported),
        ]:
            per_context_rows.append({
                "context_id": context_id,
                "method": method,
                **result,
            })
            key = "baseline" if method == "PointPillars baseline" else "camera_supported"
            for field in ("TP", "Predictions", "GT"):
                overall_counts[key][field] += result[field]

    # Overall metrics are aggregated from per-context metrics. Frames are not
    # mixed across contexts, preventing frame_idx collisions between scenes.
    overall_rows = []
    for method, key in [
        ("PointPillars baseline", "baseline"),
        ("Camera-supported filtering", "camera_supported"),
    ]:
        c = overall_counts[key]
        tp, n_pred, n_gt = c["TP"], c["Predictions"], c["GT"]
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gt if n_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        overall_rows.append({
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

    per_context_path = EVALUATION_ROOT / "fusion_validation_metrics_by_context.csv"
    overall_path = EVALUATION_ROOT / "fusion_validation_metrics_overall.csv"
    report_path = EVALUATION_ROOT / "fusion_validation_checks.txt"

    per_context_df.to_csv(per_context_path, index=False)
    overall_df.to_csv(overall_path, index=False)

    lines = [
        "FUSION EXPERIMENT VALIDATION CHECKS",
        f"BEV IoU threshold: {IOU_THRESHOLD}",
        f"ROI: x=[{X_MIN}, {X_MAX}), y=[{Y_MIN}, {Y_MAX})",
        "GT filter: Waymo type == 1 (vehicle)",
        "Prediction filter: lidar_class == vehicle",
        "Frame indexing: reset and evaluated independently per context",
        "",
    ]
    for row in checks:
        lines.extend([
            f"Context: {row['context_id']}",
            f"  GT frames: {row['frames_in_gt']}",
            f"  Baseline vehicle predictions: {row['baseline_vehicle_predictions']}",
            f"  Camera-supported vehicle predictions: {row['camera_supported_vehicle_predictions']}",
            f"  Supported count <= baseline count: {row['camera_supported_subset_count_check']}",
            f"  GT vehicle boxes in ROI: {row['ground_truth_vehicle_boxes_in_roi']}",
            f"  CSV matched pairs: {row['csv_matched_pairs']}",
            f"  Summary matched pairs: {row['summary_matched_pairs']}",
            f"  CSV/summary match counts agree: {row['matched_pair_count_agrees_with_summary']}",
            "",
        ])
    lines.extend([
        "Overall metrics (summed TP / prediction / GT counts across contexts):",
        overall_df.to_string(index=False),
        "",
        "Note: this script recomputes the metrics from existing files and does not run inference.",
    ])
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
