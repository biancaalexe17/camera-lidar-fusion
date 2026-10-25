
import numpy as np

from src.fusion.matching import box_iou


MATCH_IOU_THRESHOLD = 0.30

# Preserve the original compatibility rules and extend them for the
# four-class detector. YOLO's COCO label for a sign is "stop sign".
COMPATIBLE_CAMERA_CLASSES = {
    "vehicle": {"car", "truck", "bus"},
    "cyclist": {"bicycle", "motorcycle"},
    "pedestrian": {"person"},
    "sign": {"stop sign"},
}


def associate_detections(
    projected_boxes,
    lidar_labels,
    class_names,
    camera_candidates,
    threshold=MATCH_IOU_THRESHOLD,
):
    """
    Class-aware image-plane IoU association.

    Returns:
        iou_matrix: (num_lidar, num_camera) array
        matches: list of (lidar_index, camera_index, iou), sorted
                 by descending IoU, with one-to-one assignments.
    """
    n_lidar = len(projected_boxes)
    n_camera = len(camera_candidates)
    iou_matrix = np.zeros((n_lidar, n_camera), dtype=float)

    for i, projected_box in enumerate(projected_boxes):
        if projected_box is None:
            continue

        lidar_class = class_names[int(lidar_labels[i])]
        allowed = COMPATIBLE_CAMERA_CLASSES.get(lidar_class, set())

        for j, candidate in enumerate(camera_candidates):
            if not (set(candidate["labels"]) & allowed):
                continue

            iou_matrix[i, j] = box_iou(
                projected_box, candidate["box"]
            )

    # Same greedy one-to-one strategy as the original experiment.
    possible_pairs = [
        (float(iou_matrix[i, j]), i, j)
        for i in range(n_lidar)
        for j in range(n_camera)
        if iou_matrix[i, j] >= threshold
    ]
    possible_pairs.sort(reverse=True)

    used_lidar = set()
    used_camera = set()
    matches = []

    for overlap, i, j in possible_pairs:
        if i in used_lidar or j in used_camera:
            continue
        matches.append((i, j, overlap))
        used_lidar.add(i)
        used_camera.add(j)

    return iou_matrix, matches
