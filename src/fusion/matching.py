
import numpy as np

def box_iou(box_a, box_b):
    """IoU for image boxes in [x1, y1, x2, y2] format."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0

def merge_camera_boxes(raw_boxes, raw_scores, raw_labels, names, threshold=0.80):
    """Merge highly overlapping camera boxes, including class disagreements."""
    if len(raw_boxes) == 0:
        return []

    raw_boxes = np.asarray(raw_boxes)
    raw_scores = np.asarray(raw_scores)
    raw_labels = np.asarray(raw_labels)
    order = np.argsort(raw_scores)[::-1]
    groups = []

    for idx in order:
        target = next(
            (
                group for group in groups
                if box_iou(raw_boxes[idx], raw_boxes[group[0]]) >= threshold
            ),
            None,
        )
        if target is None:
            groups.append([int(idx)])
        else:
            target.append(int(idx))

    candidates = []
    for group in groups:
        boxes = raw_boxes[group]
        member_names = sorted({names[int(raw_labels[k])] for k in group})
        candidates.append({
            "box": np.array([
                boxes[:, 0].min(), boxes[:, 1].min(),
                boxes[:, 2].max(), boxes[:, 3].max(),
            ], dtype=float),
            "score": float(raw_scores[group].max()),
            "labels": member_names,
            "members": group,
        })

    return candidates

def greedy_match(projected_lidar_boxes, camera_candidates, min_iou=0.0):
    """
    Greedy one-to-one image-plane association.
    projected_lidar_boxes: list of dicts containing 'box' and 'label'.
    camera_candidates: output from merge_camera_boxes().
    """
    pairs = []
    used_lidar = set()
    used_camera = set()
    possible = []

    for li, lidar in enumerate(projected_lidar_boxes):
        for ci, camera in enumerate(camera_candidates):
            overlap = box_iou(lidar["box"], camera["box"])
            if overlap >= min_iou:
                possible.append((overlap, li, ci))

    for overlap, li, ci in sorted(possible, reverse=True):
        if li in used_lidar or ci in used_camera:
            continue
        used_lidar.add(li)
        used_camera.add(ci)
        pairs.append({
            "lidar_index": li,
            "camera_index": ci,
            "image_iou": overlap,
        })

    return pairs
