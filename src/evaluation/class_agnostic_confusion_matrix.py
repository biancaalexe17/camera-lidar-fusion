#!/usr/bin/env python3
"""
Class-agnostic spatial matching confusion matrices for the saved detections.

This is complementary to the existing class-aware TP/FP/FN evaluation.
It matches boxes using BEV IoU WITHOUT requiring predicted and GT classes
to agree, then compares labels to expose cross-class confusions.

Rows = ground-truth class; columns = predicted class.
"background" in the last row/column represents unmatched predictions/GT boxes.
This does not define a natural object-detection TN count.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/3DPerception")
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
LABELS = CLASSES + ["background"]

IOU_THRESHOLD = 0.50
X_MIN, X_MAX = 0.0, 40.96
Y_MIN, Y_MAX = -15.36, 15.36

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


def main():
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    matrices = {
        method: pd.DataFrame(0, index=LABELS, columns=LABELS, dtype=np.int64)
        for method in METHODS
    }

    gt_type = "[LiDARBoxComponent].type"
    gt_x = "[LiDARBoxComponent].box.center.x"
    gt_y = "[LiDARBoxComponent].box.center.y"
    gt_dx = "[LiDARBoxComponent].box.size.x"
    gt_dy = "[LiDARBoxComponent].box.size.y"
    gt_heading = "[LiDARBoxComponent].box.heading"
    gt_ts = "key.frame_timestamp_micros"

    for context_id in CONTEXTS:
        pred_path = PREDICTIONS_ROOT / f"{context_id}_detections.csv"
        gt_path = VALIDATION_ROOT / "lidar_box" / f"{context_id}.parquet"
        for path in (pred_path, gt_path):
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")

        pred = pd.read_csv(pred_path)
        gt = pd.read_parquet(gt_path)

        required_gt = [gt_type, gt_x, gt_y, gt_dx, gt_dy, gt_heading, gt_ts]
        missing = [c for c in required_gt if c not in gt.columns]
        if missing:
            raise KeyError(f"{context_id}: missing GT columns: {missing}")

        pred["lidar_class"] = pred["lidar_class"].astype(str).str.lower()
        pred = pred[pred["lidar_class"].isin(CLASSES) & in_roi(pred)].copy()
        pred["matched_bool"] = parse_matched(pred["matched"])

        # Map timestamps to frame indices consistently with the existing evaluator.
        timestamps = sorted(int(ts) for ts in gt[gt_ts].dropna().unique())
        timestamp_to_frame = {ts: i for i, ts in enumerate(timestamps)}

        gt_valid = gt[
            gt[gt_type].isin(list(GT_CLASS_IDS.values()))
            & in_roi(gt, gt_x, gt_y)
        ].copy()
        gt_valid["class_name"] = gt_valid[gt_type].map(
            {v: k for k, v in GT_CLASS_IDS.items()}
        )
        gt_valid["frame_idx"] = gt_valid[gt_ts].map(
            lambda ts: timestamp_to_frame[int(ts)]
        )

        for method in METHODS:
            method_pred = pred
            if method == "Camera-supported filtering":
                method_pred = pred[pred["matched_bool"]]

            pred_by_frame = {
                int(frame_idx): group.reset_index(drop=True)
                for frame_idx, group in method_pred.groupby("frame_idx")
            }
            gt_by_frame = {
                int(frame_idx): group.reset_index(drop=True)
                for frame_idx, group in gt_valid.groupby("frame_idx")
            }

            # Include every GT frame, even if it has no predictions or GT boxes.
            frame_ids = set(range(len(timestamps)))
            frame_ids |= set(pred_by_frame) | set(gt_by_frame)

            for frame_idx in frame_ids:
                pg = pred_by_frame.get(frame_idx, pd.DataFrame(columns=method_pred.columns))
                gg = gt_by_frame.get(frame_idx, pd.DataFrame(columns=gt_valid.columns))

                pred_boxes = [
                    [float(r.x), float(r.y), float(r.dx), float(r.dy), float(r.heading)]
                    for r in pg.itertuples()
                ]
                gt_boxes = [
                    [
                        float(r[gt_x]), float(r[gt_y]), float(r[gt_dx]),
                        float(r[gt_dy]), float(r[gt_heading])
                    ]
                    for _, r in gg.iterrows()
                ]

                # Class-agnostic one-to-one matching by BEV IoU.
                matches = evaluate_frame_boxes(
                    pred_boxes, gt_boxes, IOU_THRESHOLD
                )
                matched_pred = set()
                matched_gt = set()

                for pred_idx, gt_idx, _ in matches:
                    pred_class = str(pg.iloc[pred_idx]["lidar_class"])
                    gt_class = str(gg.iloc[gt_idx]["class_name"])
                    matrices[method].loc[gt_class, pred_class] += 1
                    matched_pred.add(int(pred_idx))
                    matched_gt.add(int(gt_idx))

                # Unmatched prediction = background GT row, predicted class column.
                for pred_idx in range(len(pred_boxes)):
                    if pred_idx not in matched_pred:
                        pred_class = str(pg.iloc[pred_idx]["lidar_class"])
                        matrices[method].loc["background", pred_class] += 1

                # Unmatched GT = true class row, background prediction column.
                for gt_idx in range(len(gt_boxes)):
                    if gt_idx not in matched_gt:
                        gt_class = str(gg.iloc[gt_idx]["class_name"])
                        matrices[method].loc[gt_class, "background"] += 1

    # Save matrices as CSV.
    for method, matrix in matrices.items():
        suffix = "baseline" if method == "PointPillars baseline" else "camera_supported"
        matrix.to_csv(EVALUATION_ROOT / f"class_agnostic_confusion_matrix_{suffix}.csv")

    # Save row-normalized matrices as well, useful for comparing class distributions.
    for method, matrix in matrices.items():
        row_sums = matrix.sum(axis=1).replace(0, np.nan)
        normalized = matrix.div(row_sums, axis=0).fillna(0)
        suffix = "baseline" if method == "PointPillars baseline" else "camera_supported"
        normalized.to_csv(
            EVALUATION_ROOT / f"class_agnostic_confusion_matrix_{suffix}_normalized.csv"
        )

    # Plot raw counts side by side.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (method, matrix) in zip(axes, matrices.items()):
        values = matrix.to_numpy()
        im = ax.imshow(values, cmap="Blues")
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha="right")
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Ground-truth class")
        ax.set_title(method)
        threshold = values.max() * 0.5 if values.size else 0
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(
                    j, i, f"{values[i, j]:,}",
                    ha="center", va="center",
                    color="white" if values[i, j] > threshold else "black"
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Class-agnostic spatial matching (BEV IoU ≥ {IOU_THRESHOLD:.2f})")
    fig.tight_layout()
    fig.savefig(
        EVALUATION_ROOT / "class_agnostic_confusion_matrices.png",
        dpi=200, bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)

    print("Rows = ground-truth class; columns = predicted class.")
    print("Off-diagonal cells indicate cross-class matches under class-agnostic spatial matching.")
    print("Background row = unmatched predictions (FP); background column = unmatched GT (FN).")
    for method, matrix in matrices.items():
        print(f"\n{method}\n{matrix}")

    print("\nSaved files:")
    for name in [
        "class_agnostic_confusion_matrix_baseline.csv",
        "class_agnostic_confusion_matrix_camera_supported.csv",
        "class_agnostic_confusion_matrix_baseline_normalized.csv",
        "class_agnostic_confusion_matrix_camera_supported_normalized.csv",
        "class_agnostic_confusion_matrices.png",
    ]:
        print(EVALUATION_ROOT / name)


if __name__ == "__main__":
    main()
