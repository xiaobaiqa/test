"""
Tests for the 3D reconstruction module (reconstruction_3d.py).
"""

import math
import unittest

from reconstruction_3d import (
    apply_rigid_transform,
    build_projection_matrix,
    compute_reprojection_error,
    depth_from_disparity,
    estimate_fundamental_matrix,
    euler_to_rotation_matrix,
    icp_step,
    normalize_points,
    point_cloud_centroid,
    project_point,
    triangulate_point,
)


class TestNormalizePoints(unittest.TestCase):
    def test_basic_normalization(self):
        points = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)]
        norm, T = normalize_points(points)
        # Centroid of normalized points should be near (0, 0)
        cx = sum(p[0] for p in norm) / len(norm)
        cy = sum(p[1] for p in norm) / len(norm)
        self.assertAlmostEqual(cx, 0.0, places=10)
        self.assertAlmostEqual(cy, 0.0, places=10)

    def test_empty_points(self):
        norm, T = normalize_points([])
        self.assertEqual(norm, [])
        self.assertEqual(T, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_single_point(self):
        points = [(3.0, 5.0)]
        norm, T = normalize_points(points)
        # Single point: mean_dist = 0, scale = 1
        self.assertEqual(len(norm), 1)


class TestProjectPoint(unittest.TestCase):
    def _identity_projection(self):
        """3x4 identity-like projection matrix (no rotation/translation, f=1)."""
        return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]

    def test_identity_projection(self):
        P = self._identity_projection()
        u, v = project_point((1.0, 2.0, 5.0), P)
        self.assertAlmostEqual(u, 1.0 / 5.0)
        self.assertAlmostEqual(v, 2.0 / 5.0)

    def test_point_at_origin(self):
        P = self._identity_projection()
        u, v = project_point((0.0, 0.0, 1.0), P)
        self.assertAlmostEqual(u, 0.0)
        self.assertAlmostEqual(v, 0.0)

    def test_zero_w_raises(self):
        P = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
        with self.assertRaises(ValueError):
            project_point((1.0, 2.0, 3.0), P)


class TestBuildProjectionMatrix(unittest.TestCase):
    def test_identity_K_and_R(self):
        K = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        t = [0.0, 0.0, 0.0]
        P = build_projection_matrix(K, R, t)
        expected = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        for i in range(3):
            for j in range(4):
                self.assertAlmostEqual(P[i][j], expected[i][j])

    def test_translation(self):
        K = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        t = [1.0, 2.0, 3.0]
        P = build_projection_matrix(K, R, t)
        expected_col = [1.0, 2.0, 3.0]
        for i in range(3):
            self.assertAlmostEqual(P[i][3], expected_col[i])


class TestTriangulatePoint(unittest.TestCase):
    def _make_cameras(self):
        K = [[500, 0, 320], [0, 500, 240], [0, 0, 1]]
        R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        t1 = [0.0, 0.0, 0.0]
        t2 = [1.0, 0.0, 0.0]  # second camera shifted 1 unit in X
        P1 = build_projection_matrix(K, R, t1)
        P2 = build_projection_matrix(K, R, t2)
        return P1, P2

    def test_triangulation_known_point(self):
        P1, P2 = self._make_cameras()
        point_3d = (0.0, 0.0, 5.0)
        p1 = project_point(point_3d, P1)
        p2 = project_point(point_3d, P2)
        # Triangulated result should be close to the original 3D point
        X, Y, Z = triangulate_point(p1, p2, P1, P2)
        self.assertAlmostEqual(Z / point_3d[2], 1.0, places=1)


class TestReprojectionError(unittest.TestCase):
    def test_zero_error(self):
        P = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        point_3d = (0.0, 0.0, 1.0)
        point_2d = project_point(point_3d, P)
        error = compute_reprojection_error(point_3d, point_2d, P)
        self.assertAlmostEqual(error, 0.0)

    def test_nonzero_error(self):
        P = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        point_3d = (1.0, 0.0, 1.0)
        point_2d = (0.0, 0.0)  # wrong observation
        error = compute_reprojection_error(point_3d, point_2d, P)
        self.assertGreater(error, 0.0)


class TestDepthFromDisparity(unittest.TestCase):
    def test_known_depth(self):
        # Z = f * B / d  =>  Z = 500 * 0.1 / 5 = 10
        depth = depth_from_disparity(5.0, 0.1, 500.0)
        self.assertAlmostEqual(depth, 10.0)

    def test_zero_disparity_raises(self):
        with self.assertRaises(ValueError):
            depth_from_disparity(0, 0.1, 500.0)


class TestPointCloudCentroid(unittest.TestCase):
    def test_centroid(self):
        points = [(0.0, 0.0, 0.0), (2.0, 4.0, 6.0)]
        cx, cy, cz = point_cloud_centroid(points)
        self.assertAlmostEqual(cx, 1.0)
        self.assertAlmostEqual(cy, 2.0)
        self.assertAlmostEqual(cz, 3.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            point_cloud_centroid([])


class TestApplyRigidTransform(unittest.TestCase):
    def test_identity_transform(self):
        R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        t = [0.0, 0.0, 0.0]
        points = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        result = apply_rigid_transform(points, R, t)
        for orig, trans in zip(points, result):
            for o, tr in zip(orig, trans):
                self.assertAlmostEqual(o, tr)

    def test_pure_translation(self):
        R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        t = [1.0, 2.0, 3.0]
        points = [(0.0, 0.0, 0.0)]
        result = apply_rigid_transform(points, R, t)
        self.assertAlmostEqual(result[0][0], 1.0)
        self.assertAlmostEqual(result[0][1], 2.0)
        self.assertAlmostEqual(result[0][2], 3.0)


class TestEulerToRotationMatrix(unittest.TestCase):
    def test_identity_angles(self):
        R = euler_to_rotation_matrix(0.0, 0.0, 0.0)
        expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(R[i][j], expected[i][j])

    def test_rotation_is_orthogonal(self):
        R = euler_to_rotation_matrix(0.3, 0.5, 0.7)
        # R * R^T should be identity
        for i in range(3):
            for j in range(3):
                dot = sum(R[i][k] * R[j][k] for k in range(3))
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(dot, expected, places=10)

    def test_yaw_90_degrees(self):
        angle = math.pi / 2
        R = euler_to_rotation_matrix(0.0, 0.0, angle)
        # Rz(90): x-axis maps to y-axis
        self.assertAlmostEqual(R[0][0], 0.0, places=10)
        self.assertAlmostEqual(R[1][0], 1.0, places=10)


class TestEstimateFundamentalMatrix(unittest.TestCase):
    def _make_stereo_correspondences(self):
        """
        Generate synthetic point correspondences from two known cameras.
        Points satisfy the epipolar constraint x2^T F x1 = 0.
        """
        K = [[500, 0, 320], [0, 500, 240], [0, 0, 1]]
        R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        t1 = [0.0, 0.0, 0.0]
        t2 = [1.0, 0.0, 0.0]
        P1 = build_projection_matrix(K, R, t1)
        P2 = build_projection_matrix(K, R, t2)
        world_points = [
            (0.0, 0.0, 5.0), (1.0, 0.5, 5.0), (-1.0, 0.5, 6.0),
            (0.5, -0.5, 4.0), (-0.5, -0.5, 5.5), (0.0, 1.0, 7.0),
            (1.0, -1.0, 6.5), (0.0, 0.0, 8.0),
        ]
        pts1 = [project_point(p, P1) for p in world_points]
        pts2 = [project_point(p, P2) for p in world_points]
        return pts1, pts2

    def test_returns_3x3_matrix(self):
        pts1, pts2 = self._make_stereo_correspondences()
        F = estimate_fundamental_matrix(pts1, pts2)
        self.assertEqual(len(F), 3)
        for row in F:
            self.assertEqual(len(row), 3)

    def test_epipolar_constraint(self):
        """x2^T * F * x1 should be near zero for true correspondences."""
        pts1, pts2 = self._make_stereo_correspondences()
        F = estimate_fundamental_matrix(pts1, pts2)
        for (u1, v1), (u2, v2) in zip(pts1, pts2):
            x1 = [u1, v1, 1.0]
            x2 = [u2, v2, 1.0]
            Fx1 = [sum(F[i][j] * x1[j] for j in range(3)) for i in range(3)]
            epipolar = sum(x2[i] * Fx1[i] for i in range(3))
            self.assertAlmostEqual(epipolar, 0.0, places=3)

    def test_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            estimate_fundamental_matrix([(0, 0)] * 7, [(0, 0)] * 7)

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            estimate_fundamental_matrix([(0, 0)] * 8, [(0, 0)] * 9)



    def test_identical_clouds(self):
        points = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        R, t, mean_dist = icp_step(points, points)
        self.assertAlmostEqual(mean_dist, 0.0)

    def test_translated_cloud(self):
        source = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        target = [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        R, t, mean_dist = icp_step(source, target)
        # Both source points are closest to different target points:
        # source[0]=(0,0,0) -> target[0]=(1,0,0), source[1]=(1,0,0) -> target[0]=(1,0,0)
        # matched_target centroid = (1,0,0), source centroid = (0.5,0,0), t[0]=0.5
        self.assertAlmostEqual(t[0], 0.5)

    def test_empty_source_raises(self):
        with self.assertRaises(ValueError):
            icp_step([], [(1.0, 0.0, 0.0)])


if __name__ == "__main__":
    unittest.main()
