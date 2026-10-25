
import numpy as np


def bev_corners(box):
    x, y, _, dx, dy, _, heading = map(float, box)
    c, s = np.cos(heading), np.sin(heading)
    local = np.array([
        [-dx/2, -dy/2], [dx/2, -dy/2],
        [dx/2, dy/2], [-dx/2, dy/2],
    ])
    return local @ np.array([[c, -s], [s, c]]).T + [x, y]


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) -
                     np.dot(y, np.roll(x, 1)))


def clip_polygon(subject, clip):
    """Sutherland-Hodgman clipping for convex polygons."""
    output = subject.tolist()
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        inp, output = output, []
        if not inp:
            break
        for j, p in enumerate(inp):
            q = np.asarray(inp[(j + 1) % len(inp)])
            p = np.asarray(p)
            cross_q = np.cross(b - a, q - a)
            cross_p = np.cross(b - a, p - a)
            q_in, p_in = cross_q >= -1e-9, cross_p >= -1e-9
            if q_in != p_in:
                d = q - p
                edge = b - a
                denom = np.cross(d, edge)
                if abs(denom) > 1e-12:
                    t = np.cross(a - p, edge) / denom
                    output.append((p + t * d).tolist())
            if q_in:
                output.append(q.tolist())
    return np.asarray(output, dtype=float).reshape(-1, 2)


def bev_iou(box_a, box_b):
    a, b = bev_corners(box_a), bev_corners(box_b)
    intersection = polygon_area(clip_polygon(a, b))
    area_a = float(box_a[3] * box_a[4])
    area_b = float(box_b[3] * box_b[4])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def summarize_overlaps(detections, threshold=0.5):
    boxes = [np.asarray(d["box"], dtype=float) for d in detections]
    pairs = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if detections[i]["label"] != detections[j]["label"]:
                continue
            iou = bev_iou(boxes[i], boxes[j])
            if iou >= threshold:
                pairs.append((i, j, float(iou)))
    return pairs
