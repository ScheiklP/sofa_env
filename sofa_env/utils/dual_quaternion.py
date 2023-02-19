from functools import reduce
import numpy as np
from sofa_env.utils.math_helper import conjugate_quaternion, multiply_quaternions, point_rotation_by_quaternion


def axis_angle_to_dquat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Converts a given rotation axis and angle into a dual quaternion

    Args:
        axis (np.ndarray): The rotation axis
        angle (float): The rotation angle

    Returns:
        q (np.ndarray): The dual quaternion that represents the given rotation
    """
    q = np.zeros(8)
    q[:3] = np.sin(angle / 2.0) * axis
    q[3] = np.cos(angle / 2.0)
    return q


def dquat_translate(translation: np.ndarray) -> np.ndarray:
    """
    Returns a dual quaternion that represents the given translation

    Args:
        translation (np.ndarray): The three-dimensional translation vector

    Returns:
        dquat (np.ndarray): The dual quaternion that represents the given translation
    """
    return point_to_dquat(translation / 2.0)


def dquat_apply(dquat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Applies the transform represented by a dual quaternion on a given vector

    Args:
        dquat (np.ndarray): The dual quaternion that should be applied on the vector
        vector (np.ndarray): The vector that should be transformed

    Returns:
        transformed (np.ndarray): The transformed three-dimensional vector

    """
    transformed = dquat_prod(dquat, np.hstack((np.zeros(3), 1.0, vector, 0.0)), dquat_conjugate(dquat))[4:7]
    return transformed


def fast_dquat_apply(dquat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    rotated = point_rotation_by_quaternion(vector, dquat[:4])
    return rotated + 2 * multiply_quaternions(dquat[4:], conjugate_quaternion(np.copy(dquat[:4])))[:3]


def dquat_apply_on_pose(dquat: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Applies a dual quaternion on a given pose (position and orientation quaternion)

    Args:
        dquat (np.ndarray): The dual quaternion that should be applied on the pose
        pose (np.ndarray): [x, y, z, a, b, c, w] The given pose: Position (x, y, z) and quaternion (a, b, c, w)

    Returns:
        transformed (np.ndarray): The transformed pose
    """
    position = dquat_apply(dquat, pose[:3])
    orientation = multiply_quaternions(dquat[:4], pose[3:])
    return np.hstack((position, orientation))


def point_to_dquat(point: np.ndarray) -> np.ndarray:
    """Converts a given three-dimensional point into a dual quaternion.

    Args:
        point (np.ndarray): The point that should be converted

    Returns:
        dquat (np.ndarray): The dual quaternion that represents the given point
    """
    dquat = np.zeros(8)
    dquat[3] = 1
    dquat[4:7] = point
    return dquat


def dquat_rotate_and_translate(rotation_quat: np.ndarray, translation_vec: np.ndarray, translate_first: bool = False) -> np.ndarray:
    """Calculates the dual quaternion that represents a rotation and a translation.

    If not specified otherwise (see ``translate_first``), the rotation will be applied before the translation.

    Args:
        rotation_quat (np.ndarray): The quaternion that represents the rotation
        translation_vec (np.ndarray): The translation vector
        translate_first (bool): Determines if the translation should be applied before the rotation. ``False`` by default.

    Returns:
        dquat (np.ndarray): The dual quaternion that represents the given transformations

    """
    t = np.zeros(4)
    t[:3] = translation_vec / 2

    dquat = np.empty(8)
    dquat[:4] = rotation_quat
    if translate_first:
        dquat[4:] = multiply_quaternions(rotation_quat, t)
    else:
        dquat[4:] = multiply_quaternions(t, rotation_quat)

    return dquat


def dquat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiplies two dual quaternions.

    Args:
        q1 (np.ndarray): The first dual quaternion.
        q2 (np.ndarray): The second dual quaternion.

    Returns:
        q (np.ndarray): The dual quaternion that represents the multiplication of the two given dual quaternions.
    """
    return np.hstack((multiply_quaternions(q1[:4], q2[:4]), multiply_quaternions(q1[:4], q2[4:]) + multiply_quaternions(q1[4:], q2[:4])))


def dquat_prod(*qs: np.ndarray) -> np.ndarray:
    """Calculates the product of dual quaternions.

    Args:
        qs (np.ndarray): The dual quaternions.

    Returns:
        q (np.ndarray): The dual quaternion that represents the product of the given dual quaternions.
    """
    return reduce(dquat_mul, qs)


def dquat_conjugate(dquat: np.ndarray) -> np.ndarray:
    """Returns the conjugate of a dual quaternion.

    Note:
        This function is not applied implace, but returns a new array.

    Args:
        dquat (np.ndarray): The dual quaternion that should be conjugated.

    Returns:
        dquat_ (np.ndarray): The conjugate of ``dquat``.
        In this case, the conjugate of a dual quaternion
            q = r + εt where r and t are quaternions, the conjugate q* is defined as:
            q* = r* - εt*
    """
    dquat_ = np.copy(dquat)
    return np.hstack((conjugate_quaternion(dquat_[:4]), -conjugate_quaternion(dquat_[4:])))


def dquat_is_invertible(dquat: np.ndarray) -> bool:
    """Checks whether the dual quaternion is invertible or not.

    Args:
        dquat (np.ndarray): The dual quaternion that should be checked for invertibility.

    Returns:
        is_invertible (bool): ``True`` if the dual quaternion is invertible, ``False`` otherwise
    """
    return any(dquat[:4] != 0)


def dquat_inverse(dquat: np.ndarray) -> np.ndarray:
    """Calculates the inverse of a dual quaternion.

    Warning:
        Raises a RuntimeError if the dual quaternion is not invertible.

    Args:
        dquat (np.ndarray): The dual quaternion that should be inverted.

    Returns:
        dq_inverse (np.ndarray): The inverse of the dual quaternion.
    """

    if not dquat_is_invertible(dquat):
        raise RuntimeError(f"Dual quaternion {dquat} can not be inverted.")

    # Assume dquat = r + εt
    r = dquat[:4]
    r_conj = conjugate_quaternion(np.copy(r))
    r_inverse = r_conj / multiply_quaternions(r, r_conj)[3]

    t = dquat[4:]

    dq_inverse = np.empty(8)
    dq_inverse[:4] = r_inverse
    dq_inverse[4:] = -multiply_quaternions(multiply_quaternions(r_inverse, t), r_inverse)

    return dq_inverse
