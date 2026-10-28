#!/usr/bin/env python3
"""
Detection confusion-matrix evaluation for the saved PointPillars predictions.

Runs AFTER inference; it does not run either detector.
Outputs:
  bev_evaluation/detection_confusion_counts.csv
  bev_evaluation/detection_confusion_matrix_<method>.csv
  bev_evaluation/detection_confusion_matrices.png

Important:
- Object detection has no natural finite set of true-negative (TN) locations.
  Therefore TN is reported as NaN / not defined, rather than inventing a count.
- The background-inclusive matrix counts unmatched predictions as
  predicted-class -> background (false positives), and unmatched GT boxes as
  background -> GT-class (false negatives).
- Matching uses the project's evaluate_frame_boxes() and BEV IoU threshold.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

CLASSES = ["pedestrian", "cyclist", "vehicle"]
GT_CLASS_IDS = {"pedestrian": 2, "cyclist": 4, "vehicle": 1}
METHODS = ["PointPillars baseline", "Camera-supported filtering"]

IOU_THRESHOLD = 0.50
X_MIN, X_MAX = 0.0, 40.96
Y_MIN, Y_MAX = -15.36, 15.36

# Find the project package containing the existing BEV matching function.
for candidate in [DRIVE_ROOT / "project", Path("/content")]:
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


def box_from_row(row):
    return [
        float(row.x), float(row.y), float(row.dx),
        float(row.dy), float(row.heading)
    ]


def load_ground_truth(gt_df, context_id):
    cols = {
        "type": "[LiDARBoxComponent].type",
        "x": "[LiDARBoxComponent].box.center.x",
        "y": "[LiDARBoxComponent].box.center.y",
        "dx": "[LiDARBoxComponent].box.size.x",
        "dy": "[LiDARBoxComponent].box.size.y",
        "heading": "[LiDARBoxComponent].box.heading",
        "ts": "key.frame_timestamp_micros",
    }
    missing = [col for col in cols.values() if col not in gt_df.columns]
    if missing:
        raise KeyError(f"{context_id}: missing GT columns: {missing}")

    timestamps = sorted(int(ts) for ts in gt_df[cols["ts"]].dropna().unique())
    timestamp_to_frame = {ts: i for i, ts in enumerate(timestamps)}
    gt_by_class_frame = {}

    for class_name, type_id in GT_CLASS_IDS.items():
        subset = gt_df[
            gt_df[cols["type"]].eq(type_id)
            & in_roi(gt_df, cols["x"], cols["y"])
        ]
        for ts, group in subset.groupby(cols["ts"]):
            frame_idx = timestamp_to_frame[int(ts)]
            boxes = [
                [
                    float(r[cols["x"]]), float(r[cols["y"]]),
                    float(r[cols["dx"]]), float(r[cols["dy"]]),
                    float(r[cols["heading"]])
                ]
                for _, r in group.iterrows()
            ]
            gt_by_class_frame[(frame_idx, class_name)] = boxes

    return timestamps, gt_by_class_frame


def evaluate_context(pred_df, gt_df, context_id):
    pred_df = pred_df.copy()
    pred_df["lidar_class"] = pred_df["lidar_class"].astype(str).str.lower()
    pred_df = pred_df[
        pred_df["lidar_class"].isin(CLASSES) & in_roi(pred_df)
    ].copy()
    pred_df["matched_bool"] = parse_matched(pred_df["matched"])

    timestamps, gt_by_class_frame = load_ground_truth(gt_df, context_id)

    # Confusion matrix labels are true classes on rows, predicted classes on columns.
    # "background" represents an unmatched GT or prediction.
    labels = CLASSES + ["background"]
    matrices = {
        method: pd.DataFrame(0, index=labels, columns=labels, dtype=np.int64)
        for method in METHODS
    }

    for class_name in CLASSES:
        # Evaluate each class independently, so only same-class boxes can be TP.
        for method in METHODS:
            method_pred = pred_df
            if method == "Camera-supported filtering":
                method_pred = pred_df[pred_df["matched_bool"]].copy()

            pred_by_frame = {}
            for frame_idx, group in method_pred.groupby("frame_idx"):
                pred_by_frame[int(frame_idx)] = group.reset_index(drop=True)

            for frame_idx in range(len(timestamps)):
                group = pred_by_frame.get(frame_idx)
                if group is None:
                    group = pd.DataFrame(columns=method_pred.columns)
                group = group[group["lidar_class"].eq(class_name)].reset_index(drop=True)
                pred_boxes = [box_from_row(r) for r in group.itertuples()]

                gt_boxes = gt_by_class_frame.get((frame_idx, class_name), [])
                matches = evaluate_frame_boxes(
                    pred_boxes, gt_boxes, IOU_THRESHOLD
                )
                matched_pred = {int(p) for p, _, _ in matches}
                matched_gt = {int(g) for _, g, _ in matches}

                # True positives: same-class matched pairs.
                for pred_idx, gt_idx, _ in matches:
                    matrices[method].loc[class_name, class_name] += 1

                # False positives: predictions left unmatched.
                for pred_idx in range(len(pred_boxes)):
                    if pred_idx not in matched_pred:
                        matrices[method].loc["background", class_name] += 1

                # False negatives: GT objects left unmatched.
                for gt_idx in range(len(gt_boxes)):
                    if gt_idx not in matched_gt:
                        matrices[method].loc[class_name, "background"] += 1

    return matrices


def main():
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    labels = CLASSES + ["background"]
    overall = {
        method: pd.DataFrame(0, index=labels, columns=labels, dtype=np.int64)
        for method in METHODS
    }

    for context_id in CONTEXTS:
        pred_path = PREDICTIONS_ROOT / f"{context_id}_detections.csv"
        gt_path = VALIDATION_ROOT / "lidar_box" / f"{context_id}.parquet"
        for path in (pred_path, gt_path):
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")

        pred_df = pd.read_csv(pred_path)
        gt_df = pd.read_parquet(gt_path)
        matrices = evaluate_context(pred_df, gt_df, context_id)

        for method in METHODS:
            overall[method] += matrices[method]

    # Derive one-vs-rest TP, FP, FN for each class from the background-inclusive
    # matching counts. TN is not defined for object detection without a defined
    # finite set of negative/background locations.
    count_rows = []
    for method, matrix in overall.items():
        for cls in CLASSES:
            tp = int(matrix.loc[cls, cls])
            fp = int(matrix.loc["background", cls])
            fn = int(matrix.loc[cls, "background"])
            count_rows.append({
                "method": method,
                "class": cls,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": np.nan,
                "TN_note": "Not defined for object detection without a finite negative-instance set",
                "Precision": tp / (tp + fp) if tp + fp else 0.0,
                "Recall": tp / (tp + fn) if tp + fn else 0.0,
                "F1": (
                    2 * tp / (2 * tp + fp + fn)
                    if 2 * tp + fp + fn else 0.0
                ),
            })

    counts_df = pd.DataFrame(count_rows)
    counts_path = EVALUATION_ROOT / "detection_confusion_counts.csv"
    counts_df.to_csv(counts_path, index=False)

    for method, matrix in overall.items():
        safe_method = "baseline" if method == "PointPillars baseline" else "camera_supported"
        matrix_path = EVALUATION_ROOT / f"detection_confusion_matrix_{safe_method}.csv"
        matrix.to_csv(matrix_path)

    # Plot the background-inclusive matrices. Cells are counts, not percentages.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (method, matrix) in zip(axes, overall.items()):
        im = ax.imshow(matrix.to_numpy(), cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Ground-truth class")
        ax.set_title(method)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{matrix.iloc[i, j]:,}",
                        ha="center", va="center",
                        color="white" if matrix.iloc[i, j] > matrix.to_numpy().max() * 0.5 else "black")
    fig.suptitle(f"Detection matching counts (BEV IoU ≥ {IOU_THRESHOLD:.2f})")
    fig.tight_layout()
    fig.savefig(EVALUATION_ROOT / "detection_confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("\nTP / FP / FN by class (TN is not naturally defined for object detection):")
    print(counts_df[["method", "class", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1"]].to_string(index=False))
    print("\nBackground-inclusive matrices (rows=true class, columns=predicted class):")
    for method, matrix in overall.items():
        print(f"\n{method}\n{matrix}")
    print("\nSaved:")
    print(counts_path)
    print(EVALUATION_ROOT / "detection_confusion_matrix_baseline.csv")
    print(EVALUATION_ROOT / "detection_confusion_matrix_camera_supported.csv")
    print(EVALUATION_ROOT / "detection_confusion_matrices.png")


if __name__ == "__main__":
    main()
