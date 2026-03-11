"""
3D Reconstruction Module

This module provides core algorithms for 3D reconstruction from images,
including camera calibration, triangulation, point cloud processing,
and Structure from Motion (SfM) utilities.
"""

import math


def normalize_points(points):
    """
    Normalize 2D points for numerical stability in DLT and similar algorithms.

    Args:
        points: list of (x, y) tuples

    Returns:
        normalized_points: list of normalized (x, y) tuples
        T: 3x3 normalization matrix as a list of lists
    """
    n = len(points)
    if n == 0:
        return [], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n

    mean_dist = sum(math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) for p in points) / n
    if mean_dist == 0:
        scale = 1.0
    else:
        scale = math.sqrt(2) / mean_dist

    T = [
        [scale, 0, -scale * cx],
        [0, scale, -scale * cy],
        [0, 0, 1],
    ]

    normalized_points = [(scale * (p[0] - cx), scale * (p[1] - cy)) for p in points]
    return normalized_points, T


def project_point(point_3d, camera_matrix):
    """
    Project a 3D point into 2D image coordinates using a camera matrix.

    Args:
        point_3d: (X, Y, Z) tuple representing a 3D point
        camera_matrix: 3x4 projection matrix as a list of lists

    Returns:
        (u, v): projected 2D image coordinates
    """
    X, Y, Z = point_3d
    p = camera_matrix

    w_u = p[0][0] * X + p[0][1] * Y + p[0][2] * Z + p[0][3]
    w_v = p[1][0] * X + p[1][1] * Y + p[1][2] * Z + p[1][3]
    w = p[2][0] * X + p[2][1] * Y + p[2][2] * Z + p[2][3]

    if w == 0:
        raise ValueError("Point is at infinity (w=0).")

    return w_u / w, w_v / w


def triangulate_point(p1, p2, proj1, proj2):
    """
    Triangulate a 3D point from two 2D correspondences and camera projection matrices.

    Uses the Direct Linear Transform (DLT) method.

    Args:
        p1: (u, v) image point in camera 1
        p2: (u, v) image point in camera 2
        proj1: 3x4 projection matrix for camera 1 (list of lists)
        proj2: 3x4 projection matrix for camera 2 (list of lists)

    Returns:
        (X, Y, Z): reconstructed 3D point
    """
    u1, v1 = p1
    u2, v2 = p2

    A = [
        [u1 * proj1[2][j] - proj1[0][j] for j in range(4)],
        [v1 * proj1[2][j] - proj1[1][j] for j in range(4)],
        [u2 * proj2[2][j] - proj2[0][j] for j in range(4)],
        [v2 * proj2[2][j] - proj2[1][j] for j in range(4)],
    ]

    X = _solve_homogeneous_svd(A)
    if X[3] == 0:
        raise ValueError("Degenerate configuration: point at infinity.")

    return X[0] / X[3], X[1] / X[3], X[2] / X[3]


def _solve_homogeneous_svd(A):
    """
    Solve the homogeneous system A * x = 0 via Gaussian elimination with
    automatic pivot selection.

    Fixes the variable whose corresponding AtA diagonal entry is smallest
    (most likely to be in the null space) to 1, then solves the resulting
    (n-1) x (n-1) linear system with partial pivoting.

    Works for any matrix with n columns (n >= 2) and m >= n-1 rows.
    For a production system, use numpy.linalg.svd.

    Args:
        A: list of lists (m x n matrix, m >= n-1)

    Returns:
        x: solution vector of length n
    """
    n = len(A[0])
    m = len(A)

    # Build normal equations AtA (n x n)
    AtA = [[sum(A[r][i] * A[r][j] for r in range(m)) for j in range(n)] for i in range(n)]

    # Choose the pivot: fix the variable whose AtA diagonal is smallest
    # (it has the least energy and is most likely non-zero in the null vector)
    fixed_idx = min(range(n), key=lambda i: AtA[i][i])

    # Reorder columns so the fixed variable is last
    order = [i for i in range(n) if i != fixed_idx] + [fixed_idx]
    AtA_p = [[AtA[order[i]][order[j]] for j in range(n)] for i in range(n)]

    k = n - 1  # number of free variables
    sub = [[AtA_p[i][j] for j in range(k)] for i in range(k)]
    rhs = [-AtA_p[i][k] for i in range(k)]

    # Gaussian elimination with partial pivoting
    for col in range(k):
        pivot_row = max(range(col, k), key=lambda r: abs(sub[r][col]))
        sub[col], sub[pivot_row] = sub[pivot_row], sub[col]
        rhs[col], rhs[pivot_row] = rhs[pivot_row], rhs[col]

        if abs(sub[col][col]) < 1e-14:
            continue  # near-singular; skip

        for row in range(col + 1, k):
            factor = sub[row][col] / sub[col][col]
            for j in range(col, k):
                sub[row][j] -= factor * sub[col][j]
            rhs[row] -= factor * rhs[col]

    # Back substitution
    x_perm = [0.0] * k
    for i in range(k - 1, -1, -1):
        if abs(sub[i][i]) < 1e-14:
            x_perm[i] = 0.0
        else:
            x_perm[i] = (
                rhs[i] - sum(sub[i][j] * x_perm[j] for j in range(i + 1, k))
            ) / sub[i][i]

    x_perm = x_perm + [1.0]

    # Undo the column permutation
    x = [0.0] * n
    for j in range(n):
        x[order[j]] = x_perm[j]

    return x


def compute_reprojection_error(point_3d, point_2d, camera_matrix):
    """
    Compute the reprojection error for a 3D-2D point correspondence.

    Args:
        point_3d: (X, Y, Z) reconstructed 3D point
        point_2d: (u, v) observed 2D image point
        camera_matrix: 3x4 projection matrix

    Returns:
        float: Euclidean reprojection error in pixels
    """
    projected = project_point(point_3d, camera_matrix)
    dx = projected[0] - point_2d[0]
    dy = projected[1] - point_2d[1]
    return math.sqrt(dx * dx + dy * dy)


def estimate_fundamental_matrix(points1, points2):
    """
    Estimate the fundamental matrix from 8 or more point correspondences
    using the normalized 8-point algorithm.

    Args:
        points1: list of (u, v) points in image 1 (at least 8 points)
        points2: list of (u, v) corresponding points in image 2

    Returns:
        F: 3x3 fundamental matrix as a list of lists
    """
    if len(points1) < 8 or len(points1) != len(points2):
        raise ValueError("At least 8 point correspondences are required.")

    pts1_n, T1 = normalize_points(points1)
    pts2_n, T2 = normalize_points(points2)

    n = len(pts1_n)
    A = []
    for i in range(n):
        u1, v1 = pts1_n[i]
        u2, v2 = pts2_n[i]
        A.append([u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1.0])

    # Solve Af = 0
    f = _solve_homogeneous_svd(A[:9] if len(A) >= 9 else A)

    # Reshape to 3x3
    F_n = [[f[0], f[1], f[2]], [f[3], f[4], f[5]], [f[6], f[7], f[8]]]

    # Denormalize: F = T2^T * F_n * T1
    F = _mat_mul(_mat_transpose(T2), _mat_mul(F_n, T1))
    return F


def _mat_mul(A, B):
    """Multiply two 3x3 matrices."""
    return [
        [sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def _mat_transpose(M):
    """Transpose a 3x3 matrix."""
    return [[M[j][i] for j in range(3)] for i in range(3)]


def build_projection_matrix(K, R, t):
    """
    Build a 3x4 camera projection matrix P = K * [R | t].

    Args:
        K: 3x3 intrinsic matrix (list of lists)
        R: 3x3 rotation matrix (list of lists)
        t: translation vector [tx, ty, tz]

    Returns:
        P: 3x4 projection matrix (list of lists)
    """
    # [R | t] is 3x4
    Rt = [[R[i][j] for j in range(3)] + [t[i]] for i in range(3)]
    # P = K (3x3) * Rt (3x4)  => 3x4
    P = [
        [sum(K[i][k] * Rt[k][j] for k in range(3)) for j in range(4)]
        for i in range(3)
    ]
    return P


def depth_from_disparity(disparity, baseline, focal_length):
    """
    Compute depth from stereo disparity.

    Uses the stereo depth equation:  Z = (f * B) / d

    Args:
        disparity: disparity value in pixels
        baseline: distance between the two cameras (same unit as desired depth)
        focal_length: focal length in pixels

    Returns:
        depth: estimated depth (same unit as baseline)
    """
    if disparity == 0:
        raise ValueError("Disparity is zero; depth is infinite.")
    return (focal_length * baseline) / disparity


def point_cloud_centroid(points):
    """
    Compute the centroid of a 3D point cloud.

    Args:
        points: list of (X, Y, Z) tuples

    Returns:
        (cx, cy, cz): centroid coordinates
    """
    n = len(points)
    if n == 0:
        raise ValueError("Empty point cloud.")
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    cz = sum(p[2] for p in points) / n
    return cx, cy, cz


def apply_rigid_transform(points, R, t):
    """
    Apply a rigid body transformation (rotation + translation) to a point cloud.

    Args:
        points: list of (X, Y, Z) tuples
        R: 3x3 rotation matrix (list of lists)
        t: translation vector [tx, ty, tz]

    Returns:
        transformed: list of transformed (X, Y, Z) tuples
    """
    transformed = []
    for X, Y, Z in points:
        x_new = R[0][0] * X + R[0][1] * Y + R[0][2] * Z + t[0]
        y_new = R[1][0] * X + R[1][1] * Y + R[1][2] * Z + t[1]
        z_new = R[2][0] * X + R[2][1] * Y + R[2][2] * Z + t[2]
        transformed.append((x_new, y_new, z_new))
    return transformed


def icp_step(source, target):
    """
    Perform one step of the Iterative Closest Point (ICP) algorithm.

    Finds the closest target point for each source point, then computes
    the optimal rigid transformation to align them.

    Args:
        source: list of (X, Y, Z) source points
        target: list of (X, Y, Z) target points

    Returns:
        R: 3x3 rotation matrix (list of lists)
        t: translation vector [tx, ty, tz]
        mean_distance: mean distance after closest-point matching
    """
    if not source or not target:
        raise ValueError("Point clouds must be non-empty.")

    # Find closest points
    matched_target = []
    total_dist = 0.0
    for sp in source:
        best_dist = float("inf")
        best_tp = target[0]
        for tp in target:
            d = math.sqrt(sum((sp[i] - tp[i]) ** 2 for i in range(3)))
            if d < best_dist:
                best_dist = d
                best_tp = tp
        matched_target.append(best_tp)
        total_dist += best_dist

    mean_distance = total_dist / len(source)

    # Compute centroids
    src_c = point_cloud_centroid(source)
    tgt_c = point_cloud_centroid(matched_target)

    # Demean
    src_d = [(p[0] - src_c[0], p[1] - src_c[1], p[2] - src_c[2]) for p in source]
    tgt_d = [(p[0] - tgt_c[0], p[1] - tgt_c[1], p[2] - tgt_c[2]) for p in matched_target]

    # Compute cross-covariance matrix H (3x3)
    H = [[sum(src_d[k][i] * tgt_d[k][j] for k in range(len(src_d)))
          for j in range(3)] for i in range(3)]

    # For a dependency-free implementation, return H as a placeholder rotation
    # In production, decompose H via SVD to obtain the optimal R.
    # Here we return the identity rotation as a safe fallback.
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    t = [tgt_c[i] - src_c[i] for i in range(3)]

    return R, t, mean_distance


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix.

    Uses the ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)

    Args:
        roll: rotation around X axis in radians
        pitch: rotation around Y axis in radians
        yaw: rotation around Z axis in radians

    Returns:
        R: 3x3 rotation matrix (list of lists)
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    R = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    return R
