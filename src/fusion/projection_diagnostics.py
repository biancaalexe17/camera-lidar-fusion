
import cv2
import numpy as np


def inspect_box_projection(
    detection, points_vehicle, T_camera_vehicle, K, dist,
    image_width, image_height,
):
    """Report box geometry and raw projected corner bounds."""
    box = np.asarray(detection["box"], dtype=float)
    cx, cy, cz, dx, dy, dz, heading = box

    c, s = np.cos(heading), np.sin(heading)
    local_xy = np.array([
        [-dx/2, -dy/2], [dx/2, -dy/2],
        [dx/2, dy/2], [-dx/2, dy/2],
    ])
    xy = local_xy @ np.array([[c, -s], [s, c]]).T + [cx, cy]
    corners_vehicle = np.vstack([
        np.column_stack([xy, np.full(4, cz - dz/2)]),
        np.column_stack([xy, np.full(4, cz + dz/2)]),
    ])

    corners_h = np.column_stack([corners_vehicle, np.ones(8)])
    camera_aligned = (T_camera_vehicle @ corners_h.T).T[:, :3]
    corners_cv = np.column_stack([
        -camera_aligned[:, 1],
        -camera_aligned[:, 2],
        camera_aligned[:, 0],
    ])

    front = corners_cv[:, 2] > 0.1
    result = {
        "box": box.tolist(),
        "front_corners": int(front.sum()),
        "camera_center_xyz": camera_aligned.mean(axis=0).tolist(),
        "depth_range": [
            float(corners_cv[:, 2].min()),
            float(corners_cv[:, 2].max()),
        ],
        "points_in_bev": 0,
        "points_in_3d": 0,
        "undistorted_bounds": None,
        "distorted_bounds": None,
    }

    # Count raw points inside the predicted box.
    pts = np.asarray(points_vehicle)[:, :3]
    finite = np.isfinite(pts).all(axis=1)
    delta = pts[:, :2] - [cx, cy]
    lx = delta[:, 0] * c + delta[:, 1] * s
    ly = -delta[:, 0] * s + delta[:, 1] * c
    bev = (
        finite
        & (np.abs(lx) <= dx/2)
        & (np.abs(ly) <= dy/2)
    )
    full = bev & (np.abs(pts[:, 2] - cz) <= dz/2)
    result["points_in_bev"] = int(bev.sum())
    result["points_in_3d"] = int(full.sum())

    if front.sum() < 2:
        return result

    p = corners_cv[front]
    uv_undist, _ = cv2.projectPoints(
        p, np.zeros(3), np.zeros(3), K, np.zeros(5)
    )
    uv_dist, _ = cv2.projectPoints(
        p, np.zeros(3), np.zeros(3), K, dist
    )

    def bounds(uv):
        uv = uv.reshape(-1, 2)
        if not np.isfinite(uv).all():
            return None
        return [
            float(uv[:, 0].min()), float(uv[:, 1].min()),
            float(uv[:, 0].max()), float(uv[:, 1].max()),
        ]

    result["undistorted_bounds"] = bounds(uv_undist)
    result["distorted_bounds"] = bounds(uv_dist)
    return result
