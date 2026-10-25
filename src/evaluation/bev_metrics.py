
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


IOU_THRESHOLDS = (0.25, 0.50, 0.70)


def bev_corners(cx, cy, sx, sy, heading):
    """Return the four corners of an oriented BEV box."""
    local = np.array([
        [-sx / 2, -sy / 2],
        [sx / 2, -sy / 2],
        [sx / 2, sy / 2],
        [-sx / 2, sy / 2],
    ], dtype=np.float32)

    c, s = np.cos(heading), np.sin(heading)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)

    return local @ rotation.T + np.array([cx, cy], dtype=np.float32)


def bev_iou(pred_box, gt_box):
    """
    Calculate oriented BEV IoU.

    Each box is [center_x, center_y, size_x, size_y, heading].
    """
    pred_corners = bev_corners(*pred_box)
    gt_corners = bev_corners(*gt_box)

    area_pred = float(pred_box[2] * pred_box[3])
    area_gt = float(gt_box[2] * gt_box[3])

    intersection, _ = cv2.intersectConvexConvex(
        pred_corners, gt_corners
    )

    union = area_pred + area_gt - float(intersection)
    return float(intersection / union) if union > 0 else 0.0


def evaluate_frame_boxes(pred_boxes, gt_boxes, iou_threshold):
    """
    Match predictions to ground truth using maximum-IoU
    Hungarian assignment, accepting matches above the threshold.

    Boxes have shape (N, 5): [x, y, size_x, size_y, heading].
    """
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 5)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 5)

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []

    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))

    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = bev_iou(pred_box, gt_box)

    cost = 1.0 - iou_matrix
    cost[iou_matrix < iou_threshold] = 1e6

    rows, cols = linear_sum_assignment(cost)

    return [
        (int(i), int(j), float(iou_matrix[i, j]))
        for i, j in zip(rows, cols)
        if iou_matrix[i, j] >= iou_threshold
    ]


def calculate_metrics(tp, total_predictions, total_gt,
                      matched_ious, center_errors, iou_threshold):
    """Calculate the metrics used in the original evaluation."""
    precision = tp / total_predictions if total_predictions else 0.0
    recall = tp / total_gt if total_gt else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    return {
        "BEV_IoU_threshold": iou_threshold,
        "true_positive_matches": tp,
        "predictions": total_predictions,
        "ground_truth_vehicles": total_gt,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "mean_matched_BEV_IoU": (
            float(np.mean(matched_ious)) if matched_ious else np.nan
        ),
        "mean_center_error_m": (
            float(np.mean(center_errors)) if center_errors else np.nan
        ),
    }
