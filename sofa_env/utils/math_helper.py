from typing import Union, Tuple, Optional
from numba import njit
import numpy as np


def is_in(x, low, high) -> bool:
    """Checks if a value x is in the interval of [low, high]."""
    return low <= x <= high


def farthest_point_sampling(points: np.ndarray, num_samples: int, starting_point_index: Optional[int] = None, return_indices: bool = False) -> np.ndarray:
    """Sample num_samples points from points using the farthest point sampling strategy.

    Eldar, Yuval, Michael Lindenbaum, Moshe Porat, and Yehoshua Y. Zeevi. 'The farthest point strategy for progressive image sampling.' IEEE Transactions on Image Processing 6, no. 9 (1997): 1305-1315.

    Args:
        points (np.ndarray): (N, 3) array of points
        num_samples (int): number of samples to take
        starting_point_index (Optional[int]): index of the point to start with. If None, a random point is chosen.
        return_indices (bool): if True, return the indices of the sampled points instead of the points themselves

    Returns:
         np.ndarray: (num_samples, 3) array of sampled points or (num_samples,) array of indices of sampled points
    """

    if num_samples > points.shape[0]:
        raise ValueError("Cannot sample more points than are available.")

    if starting_point_index is None:
        # Randomly sample the first point
        starting_point_index = np.random.randint(points.shape[0])
    else:
        # Select the first point
        if starting_point_index >= points.shape[0]:
            raise ValueError("Starting_point_index must be smaller than the number of points.")

    sampled_points = points[starting_point_index, :][np.newaxis, :]
    sampled_indices = np.array([starting_point_index])

    # Sample the remaining points
    for _ in range(num_samples - 1):
        dists = np.linalg.norm(points - sampled_points[:, np.newaxis], axis=-1)
        farthest_point_index = np.argmax(np.min(dists, axis=0))
        next_point = points[farthest_point_index, :][np.newaxis, :]
        sampled_points = np.concatenate([sampled_points, next_point], axis=0)
        sampled_indices = np.concatenate([sampled_indices, np.array([farthest_point_index])], axis=0)

    if return_indices:
        return sampled_indices
    else:
        return sampled_points


def rotation_matrix_from_vectors(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Find the rotation matrix that rotates a reference vector into a target vector.

    Args:
        reference (np.ndarray): Reference vector.
        target (np.ndarray): Target vector.

    Returns:
        rotation_matrix (np.ndarray): Rotation matrix that rotates the reference vector into the target vector.
    """
    # Normalize both vectors
    a, b = (reference / np.linalg.norm(reference)).reshape(3), (target / np.linalg.norm(target)).reshape(3)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    # Construct rotation matrix
    # See https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix


def quaternion_from_vectors(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Find the quaternion that rotates a reference vector into a target vector.

    Args:
        reference (np.ndarray): Reference vector.
        target (np.ndarray): Target vector.

    Returns:
        quaternion (np.ndarray): Quaternion that rotates the reference vector into the target vector.
    """
    rotation_matrix = rotation_matrix_from_vectors(reference, target)

    # Get quaternion
    # See https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    qw = np.sqrt(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]) / 2
    denominator = 4 * qw
    qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / denominator
    qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / denominator
    qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / denominator

    quaternion = np.array([qx, qy, qz, qw])

    return quaternion


def multiply_quaternions(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply two quaternions."""
    return np.array(
        [
            left[0] * right[3] + left[1] * right[2] - left[2] * right[1] + left[3] * right[0],
            -left[0] * right[2] + left[1] * right[3] + left[2] * right[0] + left[3] * right[1],
            left[0] * right[1] - left[1] * right[0] + left[2] * right[3] + left[3] * right[2],
            -left[0] * right[0] - left[1] * right[1] - left[2] * right[2] + left[3] * right[3],
        ],
    )


def conjugate_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Conjugate a quaternion by flipping the sign of the imaginary parts."""
    quaternion[:-1] = -quaternion[:-1]

    return quaternion


def rotated_z_axis(q: np.ndarray) -> np.ndarray:
    """Rotated Z-Axis around quaternion q.

    Note:
        Simplified version of point_rotation_by_quaternion
    """

    x = 2 * q[0] * q[2] + 2 * q[3] * q[1]
    y = -2 * q[3] * q[0] + 2 * q[1] * q[2]
    z = q[3] * q[3] - q[0] * q[0] - q[1] * q[1] + q[2] * q[2]
    return np.array([x, y, z])


def rotated_x_axis(q: np.ndarray) -> np.ndarray:
    """Rotated X-Axis around quaternion q.

    Note:
        Simplified version of point_rotation_by_quaternion
    """

    x = q[0] * q[0] + q[3] * q[3] - q[1] * q[1] - q[2] * q[2]
    y = 2 * q[3] * q[2] + 2 * q[0] * q[1]
    z = -2 * q[3] * q[1] + 2 * q[0] * q[2]
    return np.array([x, y, z])


def rotated_y_axis(q: np.ndarray) -> np.ndarray:
    """Rotated Y-Axis around quaternion q.

    Note:
        Simplified version of point_rotation_by_quaternion
    """

    x = 2 * q[0] * q[1] - 2 * q[3] * q[2]
    y = q[3] * q[3] - q[0] * q[0] + q[1] * q[1] - q[2] * q[2]
    z = 2 * q[3] * q[0] + 2 * q[1] * q[2]

    return np.array([x, y, z])


def point_rotation_by_quaternion(point: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotates a point by quaternion q.

    Note:
        From http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/transforms/derivations/vectors/index.htm

        For Rotation: P2=q * P1 * q'

        Which gives:

        P2.x = x*(qx*qx+qw*qw-qy*qy- qz*qz) + y*(2*qx*qy- 2*qw*qz) + z*(2*qx*qz+ 2*qw*qy)
        P2.y = x*(2*qw*qz + 2*qx*qy) + y*(qw*qw - qx*qx+ qy*qy - qz*qz)+ z*(-2*qw*qx+ 2*qy*qz)
        P2.z = x*(-2*qw*qy+ 2*qx*qz) + y*(2*qw*qx+ 2*qy*qz)+ z*(qw*qw - qx*qx- qy*qy+ qz*qz)

        Where:

        P2 = output vector
        P1 = input vector
        q = quaternion representing rotation
    """

    x = point[0] * (q[0] * q[0] + q[3] * q[3] - q[1] * q[1] - q[2] * q[2]) + point[1] * (2 * q[0] * q[1] - 2 * q[3] * q[2]) + point[2] * (2 * q[0] * q[2] + 2 * q[3] * q[1])
    y = point[0] * (2 * q[3] * q[2] + 2 * q[0] * q[1]) + point[1] * (q[3] * q[3] - q[0] * q[0] + q[1] * q[1] - q[2] * q[2]) + point[2] * (-2 * q[3] * q[0] + 2 * q[1] * q[2])
    z = point[0] * (-2 * q[3] * q[1] + 2 * q[0] * q[2]) + point[1] * (2 * q[3] * q[0] + 2 * q[1] * q[2]) + point[2] * (q[3] * q[3] - q[0] * q[0] - q[1] * q[1] + q[2] * q[2])

    return np.array([x, y, z])


def euler_to_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:
    """Computes the rotation matrix for XYZ euler angles.

    Args:
        euler_angles (np.ndarray): Rotations around [X, Y, Z] in degrees for rotation order XYZ.

    Returns:
        rotation_matrix (np.ndarray): The 3x3 rotation matrix.
    """

    c1 = np.cos(euler_angles[0] * np.pi / 180.0)
    c2 = np.cos(euler_angles[1] * np.pi / 180.0)
    c3 = np.cos(euler_angles[2] * np.pi / 180.0)
    s1 = np.sin(euler_angles[0] * np.pi / 180.0)
    s2 = np.sin(euler_angles[1] * np.pi / 180.0)
    s3 = np.sin(euler_angles[2] * np.pi / 180.0)

    rotation_matrix = np.array(
        [
            [c2 * c3, -c2 * s3, s2],
            [c1 * s3 + s1 * s2 * c3, c1 * c3 - s1 * s2 * s3, -s1 * c2],
            [s1 * s3 - c1 * s2 * c3, s1 * c3 + c1 * s2 * s3, c1 * c2],
        ]
    )

    return rotation_matrix


def rotation_matrix_to_euler(rotation_matrix: np.ndarray):
    """Computes the XYZ euler angles from a rotation matrix.

    Args:
        rotation_matrix (np.ndarray): The 3x3 rotation matrix.

    Returns:
        euler_angles (np.ndarray): Rotations around [X, Y, Z] in degrees for rotation order XYZ.
    """
    r11, r12, r13 = rotation_matrix[0]
    _, _, r23 = rotation_matrix[1]
    _, _, r33 = rotation_matrix[2]

    theta1 = np.arctan2(-r23, r33)
    theta2 = np.arctan(r13 * np.cos(theta1) / r33)
    theta3 = np.arctan2(-r12, r11)

    theta1 = theta1 * 180 / np.pi
    theta2 = theta2 * 180 / np.pi
    theta3 = theta3 * 180 / np.pi

    return (theta1, theta2, theta3)


def homogeneous_transform_to_pose(homogeneous_transform: np.ndarray) -> np.ndarray:
    """Extracts a 7x1 pose from a 4x4 homogeneous matrix.

    Args:
        homogeneous_transform (np.ndarray): A 4x4 homogeneous transformation matrix.

    Returns:
        quaternion (np.ndarray): The equivalent 7x1 pose consisting of Cartesian position and quaternion: [x, y, z, a, b, c, w].
    """

    quaternion = np.empty((7,))

    # position x, y, z
    quaternion[:3] = homogeneous_transform[:3, 3]

    # orientation as [a, b, c, w] quaternion
    quaternion[3:] = rotation_matrix_to_quaternion(homogeneous_transform[:3, :3])

    return quaternion


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    """Converts a 3x3 rotation matrix into the equivalent quaternion.

    Note:
        See https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    Args:
        rotation_matrix (np.ndarray): A 3x3 rotation matrix.

    Returns:
        quaternion (np.ndarray): The equivalent 7x1 quaternion [x, y, z, a, b, c, w].
    """

    four_x_squared_minus_1 = rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]
    four_y_squared_minus_1 = rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]
    four_z_squared_minus_1 = rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]
    four_w_squared_minus_1 = rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]

    biggest_index = 0

    four_biggest_squared_minus_1 = four_w_squared_minus_1
    if four_x_squared_minus_1 > four_biggest_squared_minus_1:
        four_biggest_squared_minus_1 = four_x_squared_minus_1
        biggest_index = 1

    if four_y_squared_minus_1 > four_biggest_squared_minus_1:
        four_biggest_squared_minus_1 = four_y_squared_minus_1
        biggest_index = 2

    if four_z_squared_minus_1 > four_biggest_squared_minus_1:
        four_biggest_squared_minus_1 = four_z_squared_minus_1
        biggest_index = 3

    biggest_value = np.sqrt(four_biggest_squared_minus_1 + 1.0) * 0.5
    multiplier = 0.25 / biggest_value

    if biggest_index == 0:
        return np.array(
            [
                (rotation_matrix[2][1] - rotation_matrix[1][2]) * multiplier,
                (rotation_matrix[0][2] - rotation_matrix[2][0]) * multiplier,
                (rotation_matrix[1][0] - rotation_matrix[0][1]) * multiplier,
                biggest_value,
            ]
        )
    if biggest_index == 1:
        return np.array(
            [
                biggest_value,
                (rotation_matrix[1][0] + rotation_matrix[0][1]) * multiplier,
                (rotation_matrix[0][2] + rotation_matrix[2][0]) * multiplier,
                (rotation_matrix[2][1] - rotation_matrix[1][2]) * multiplier,
            ]
        )
    if biggest_index == 2:
        return np.array(
            [
                (rotation_matrix[1][0] + rotation_matrix[0][1]) * multiplier,
                biggest_value,
                (rotation_matrix[2][1] + rotation_matrix[1][2]) * multiplier,
                (rotation_matrix[0][2] - rotation_matrix[2][0]) * multiplier,
            ]
        )
    if biggest_index == 3:
        return np.array(
            [
                (rotation_matrix[0][2] + rotation_matrix[2][0]) * multiplier,
                (rotation_matrix[2][1] + rotation_matrix[1][2]) * multiplier,
                biggest_value,
                (rotation_matrix[1][0] - rotation_matrix[0][1]) * multiplier,
            ]
        )
    else:
        raise RuntimeError("Mathematical error in converting to quaternion.")


def quaternion_to_euler_angles(quat: np.ndarray) -> np.ndarray:
    """Converts a unit quaternion into the corresponding euler angles (in degrees).

    Args:
        quat (np.ndarray): The given quaternion

    Returns:
        euler (np.ndarray): XYZ euler angles in degrees

    Note:
        Basic idea: Convert the quaternion to a rotation matrix and compare its values with the rotation matrix from
        ``euler_to_rotation_matrix``. For the quaternion-matrix conversion, see
        https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Conversion_to_and_from_the_matrix_representation
    """

    b, c, d, a = quat

    theta_3 = np.arctan2(2.0 * (a * d - b * c), a ** 2 + b ** 2 - c ** 2 - d ** 2)
    theta_1 = np.arctan2(2.0 * (a * b - c * d), a ** 2 - b ** 2 - c ** 2 + d ** 2)
    theta_2 = np.arcsin(2.0 * (a * c + b * d))

    return (180.0 / np.pi) * np.array([theta_1, theta_2, theta_3])


def euler_angles_to_quaternion(euler_angles: np.ndarray) -> np.ndarray:
    """Converts XYZ euler angles into a quaternion.

    Args:
        euler_angles (np.ndarray): The euler angles in order XYZ in degrees.

    Returns:
        quat (np.ndarray): The quaternion that represents the same rotation as the euler angles.
    """

    quat = np.empty(4)
    pts = (np.pi / 360) * euler_angles[:3]
    cpan, ctilt, cspin = np.cos(pts)
    span, stilt, sspin = np.sin(pts)

    quat[3] = cpan * ctilt * cspin - span * stilt * sspin
    quat[0] = cpan * stilt * sspin + span * ctilt * cspin
    quat[1] = cpan * stilt * cspin - span * ctilt * sspin
    quat[2] = cpan * ctilt * sspin + span * stilt * cspin

    return quat


def distance_between_line_segments(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    clamp_segments: bool = False,
) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None], float]:

    """Given two lines defined by (a0,a1,b0,b1), return the closest points on each segment and their distance.

    Args:
        a0 (np.ndarray): First point on line A.
        a1 (np.ndarray): Second point on line A.
        b0 (np.ndarray): First point on line B.
        b1 (np.ndarray): Second point on line B.
        clamp_segments (bool): If set to True, the lines will be treated as segments, ending at points a0 and a1, and b0 and b1.

    Returns:
        pA (np.ndarray): The closest point on line/segment A.
        pB (np.ndarray): The closest point on line/segment B.
        d (float): The closest distance betwenn the lines/segments.
    """

    a0 = np.asarray(a0, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    b0 = np.asarray(b0, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    _A = A / magA
    _B = B / magB

    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross) ** 2

    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A, (b0 - a0))

        # Overlap only possible with clamping
        if clamp_segments:
            d1 = np.dot(_A, (b1 - a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if np.absolute(d0) < np.absolute(d1):
                    return a0.copy(), b0.copy(), float(np.linalg.norm(a0 - b0))
                return a0.copy(), b1.copy(), float(np.linalg.norm(a0 - b1))

            # Is segment B after A?
            elif d0 >= magA <= d1:
                if np.absolute(d0) < np.absolute(d1):
                    return a1.copy(), b0.copy(), float(np.linalg.norm(a1 - b0))
                return a1, b1, float(np.linalg.norm(a1 - b1))

        # Segments overlap, return distance between parallel segments
        return None, None, float(np.linalg.norm(((d0 * _A) + a0) - b0))

    # Lines criss-cross: Calculate the projected closest points
    t = b0 - a0
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA / denom
    t1 = detB / denom

    pA = a0 + (_A * t0)  # Projected closest point on segment A
    pB = b0 + (_B * t1)  # Projected closest point on segment B

    # Clamp projections
    if clamp_segments:
        if t0 < 0:
            pA = a0
        elif t0 > magA:
            pA = a1

        if t1 < 0:
            pB = b0
        elif t1 > magB:
            pB = b1

        # Clamp projection A
        if (t0 < 0) or (t0 > magA):
            dot = np.dot(_B, (pA - b0))
            if dot < 0:
                dot = 0
            elif dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (t1 < 0) or (t1 > magB):
            dot = np.dot(_A, (pB - a0))
            if dot < 0:
                dot = 0
            elif dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    return pA, pB, float(np.linalg.norm(pA - pB))


@njit
def anorm2(X):
    """Compute euclidean norm along axis 1"""
    return np.sqrt(np.sum(X ** 2, axis=1))


@njit
def adet(X, Y, Z):
    """Compute 3x3 determinant along axis 1"""
    ret = np.multiply(np.multiply(X[:, 0], Y[:, 1]), Z[:, 2])
    ret += np.multiply(np.multiply(Y[:, 0], Z[:, 1]), X[:, 2])
    ret += np.multiply(np.multiply(Z[:, 0], X[:, 1]), Y[:, 2])
    ret -= np.multiply(np.multiply(Z[:, 0], Y[:, 1]), X[:, 2])
    ret -= np.multiply(np.multiply(Y[:, 0], X[:, 1]), Z[:, 2])
    ret -= np.multiply(np.multiply(X[:, 0], Z[:, 1]), Y[:, 2])
    return ret


@njit
def is_inside_mesh(triangles: np.ndarray, X: np.ndarray) -> np.ndarray:
    """ Checks if an array of points is inside a triangular mesh.

    Args:
        triangles (np.ndarray): Array of triangles, shape (n, 3, 3).
        X (np.ndarray): Array of points, shape (m, 3).

    Returns:
        np.ndarray: Array of booleans, shape (m,).

    Notes:
        From https://github.com/marmakoide/inside-3d-mesh based on https://igl.ethz.ch/projects/winding-number/
    """

    # One generalized winding number per input vertex
    ret = np.zeros(X.shape[0], dtype=X.dtype)

    # Accumulate generalized winding number for each triangle
    for U, V, W in triangles:
        A, B, C = U - X, V - X, W - X
        omega = adet(A, B, C)

        a, b, c = anorm2(A), anorm2(B), anorm2(C)
        k = a * b * c
        k += c * np.sum(np.multiply(A, B), axis=1)
        k += a * np.sum(np.multiply(B, C), axis=1)
        k += b * np.sum(np.multiply(C, A), axis=1)

        ret += np.arctan2(omega, k)

    # Introduced buffer value of -0.001 as proposed in https://github.com/marmakoide/inside-3d-mesh/issues/2.
    # Increased accuracy from 79% to 99%.
    return ret >= 2 * np.pi - 0.001


class cubic_interp1d:
    """
    Interpolate a 1-D function using cubic splines.
    `x` and `y` are arrays of values used to approximate some function f:
    ``y = f(x)``.

    Args:
        x (np.ndarray): A 1-D array of real/complex values.
        y (np.ndarray): A 1-D array of real values. The length of y along the interpolation axis must be equal to the length of x.

    Note:
        This was copied from: https://stackoverflow.com/a/48085583
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        x = np.asfarray(x)
        y = np.asfarray(y)

        # remove non finite values
        # indexes = np.isfinite(x)
        # x = x[indexes]
        # y = y[indexes]

        # check if sorted
        if np.any(np.diff(x) < 0):
            indexes = np.argsort(x)
            x = x[indexes]
            y = y[indexes]

        size = len(x)

        xdiff = np.diff(x)
        ydiff = np.diff(y)

        # allocate buffer matrices
        Li = np.empty(size)
        Li_1 = np.empty(size - 1)
        z = np.empty(size)

        # fill diagonals Li and Li-1 and solve [L][y] = [B]
        Li[0] = np.sqrt(2 * xdiff[0])
        Li_1[0] = 0.0
        B0 = 0.0  # natural boundary
        z[0] = B0 / Li[0]

        for i in range(1, size - 1, 1):
            Li_1[i] = xdiff[i - 1] / Li[i - 1]
            Li[i] = np.sqrt(2 * (xdiff[i - 1] + xdiff[i]) - Li_1[i - 1] * Li_1[i - 1])
            Bi = 6 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])
            z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

        i = size - 1
        Li_1[i - 1] = xdiff[-1] / Li[i - 1]
        Li[i] = np.sqrt(2 * xdiff[-1] - Li_1[i - 1] * Li_1[i - 1])
        Bi = 0.0  # natural boundary
        z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

        # solve [L.T][x] = [y]
        i = size - 1
        z[i] = z[i] / Li[i]
        for i in range(size - 2, -1, -1):
            z[i] = (z[i] - Li_1[i - 1] * z[i + 1]) / Li[i]

        self.size = size
        self.x = x
        self.y = y
        self.z = z

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # find index and clip to bounds if outside
        index = np.clip(np.searchsorted(a=self.x, v=x), 1, self.size - 1)
        # calculate cubic f(x)
        xi1, xi0 = self.x[index], self.x[index - 1]
        yi1, yi0 = self.y[index], self.y[index - 1]
        zi1, zi0 = self.z[index], self.z[index - 1]
        hi1 = xi1 - xi0
        fx = zi0 / (6 * hi1) * (xi1 - x) ** 3 + zi1 / (6 * hi1) * (x - xi0) ** 3 + (yi1 / hi1 - zi1 * hi1 / 6) * (x - xi0) + (yi0 / hi1 - zi0 * hi1 / 6) * (xi1 - x)
        return fx
