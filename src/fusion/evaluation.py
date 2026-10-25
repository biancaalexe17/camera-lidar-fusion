
import numpy as np


def summarize_fusion(
    lidar_detections,
    projected_boxes,
    camera_candidates,
    iou_matrix,
    matches,
    threshold=0.30,
):
    """Summarize projection and class-aware association results."""
    total_lidar = len(lidar_detections)
    total_projected = sum(box is not None for box in projected_boxes)
    total_camera = len(camera_candidates)

    eligible_pairs = int(np.count_nonzero(iou_matrix >= threshold))
    matched_ious = [float(match[2]) for match in matches]
    accepted_matches = len(matches)

    return {
        "lidar_detections": total_lidar,
        "projected_lidar_boxes": total_projected,
        "camera_candidates": total_camera,
        "eligible_pairs_before_one_to_one_assignment": eligible_pairs,
        "accepted_matches": accepted_matches,
        "unmatched_camera_candidates": total_camera - len({
            j for _, j, _ in matches
        }),
        "unmatched_projected_lidar_boxes": total_projected - len({
            i for i, _, _ in matches
        }),
        "mean_matched_iou": (
            float(np.mean(matched_ious)) if matched_ious else None
        ),
        "matched_ious": matched_ious,
    }
