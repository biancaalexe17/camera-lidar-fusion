
import cv2
import numpy as np

def project_lidar_box(
    box, points_vehicle, T_camera_vehicle, K, dist,
    image_width, image_height, min_points=10
):
    """Project a 3D vehicle-frame box into a distorted camera image."""
    x, y, cz, sx, sy, dz, heading = map(float, box)
    if sx <= 0 or sy <= 0 or dz <= 0 or len(points_vehicle) == 0:
        return None

    # Require LiDAR support in the oriented BEV footprint.
    pts = np.asarray(points_vehicle)
    finite = np.isfinite(pts[:, :3]).all(axis=1)
    c, s = np.cos(heading), np.sin(heading)
    dxp, dyp = pts[:, 0] - x, pts[:, 1] - y
    lx = dxp * c + dyp * s
    ly = -dxp * s + dyp * c
    inside = finite & (np.abs(lx) <= sx / 2) & (np.abs(ly) <= sy / 2)
    if inside.sum() < min_points:
        return None

    # Construct the eight corners in vehicle coordinates.
    local_xy = np.array([
        [-sx/2, -sy/2], [sx/2, -sy/2],
        [sx/2, sy/2], [-sx/2, sy/2],
    ])
    xy = local_xy @ np.array([[c, -s], [s, c]]).T + [x, y]
    corners = np.vstack([
        np.column_stack([xy, np.full(4, cz - dz/2)]),
        np.column_stack([xy, np.full(4, cz + dz/2)]),
    ])
    corners_h = np.column_stack([corners, np.ones(8)])
    cam = (T_camera_vehicle @ corners_h.T).T[:, :3]

    # Convert camera-aligned coordinates to OpenCV optical coordinates.
    cv = np.column_stack([-cam[:, 1], -cam[:, 2], cam[:, 0]])

    # Twelve edges of the box.
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    # Image-frustum planes in optical coordinates, with positive
    # signed distance meaning the point is inside the plane.
    near = 0.1
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    planes = [
        np.array([0., 0., 1., -near]),
        np.array([fx, 0., cx, 0.]),
        np.array([-fx, 0., image_width - cx, 0.]),
        np.array([0., fy, cy, 0.]),
        np.array([0., -fy, image_height - cy, 0.]),
    ]

    # Keep corners inside the undistorted frustum and add edge-plane
    # intersections. This handles boxes crossing the image boundary.
    candidates = []
    for point in cv:
        if all(np.dot(plane[:3], point) + plane[3] >= 0 for plane in planes):
            candidates.append(point)

    for i, j in edges:
        a, b = cv[i], cv[j]
        for plane in planes:
            da = np.dot(plane[:3], a) + plane[3]
            db = np.dot(plane[:3], b) + plane[3]
            if da * db < 0:
                t = da / (da - db)
                p = a + t * (b - a)
                if all(
                    np.dot(other[:3], p) + other[3] >= -1e-7
                    for other in planes
                ):
                    candidates.append(p)

    if len(candidates) < 2:
        return None

    candidates = np.asarray(candidates)
    projected, _ = cv2.projectPoints(
        candidates, np.zeros(3), np.zeros(3), K, dist
    )
    uv = projected.reshape(-1, 2)
    if not np.isfinite(uv).all():
        return None

    x1 = float(np.clip(uv[:, 0].min(), 0, image_width))
    y1 = float(np.clip(uv[:, 1].min(), 0, image_height))
    x2 = float(np.clip(uv[:, 0].max(), 0, image_width))
    y2 = float(np.clip(uv[:, 1].max(), 0, image_height))

    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def load_camera_geometry(calibration):
    """Build the same camera matrices as the original fusion experiment."""
    intr = calibration["intrinsic"]

    K = np.array([
        [intr["f_u"], 0.0, intr["c_u"]],
        [0.0, intr["f_v"], intr["c_v"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    dist = np.array([
        intr["k1"], intr["k2"], intr["p1"], intr["p2"], intr["k3"],
    ], dtype=np.float64)

    T_vehicle_camera = np.asarray(
        calibration["extrinsic"], dtype=np.float64
    )
    T_camera_vehicle = np.linalg.inv(T_vehicle_camera)

    return {
        "K": K,
        "dist": dist,
        "T_camera_vehicle": T_camera_vehicle,
        "image_width": int(calibration["width"]),
        "image_height": int(calibration["height"]),
    }
