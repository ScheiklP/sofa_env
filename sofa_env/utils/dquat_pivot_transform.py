from typing import Callable
import numpy as np

from sofa_env.utils.dual_quaternion import dquat_apply, dquat_rotate_and_translate, fast_dquat_apply
from sofa_env.utils.math_helper import conjugate_quaternion, multiply_quaternions, euler_angles_to_quaternion, point_rotation_by_quaternion, rotated_z_axis


def dquat_ptsd_to_pose(ptsd: np.ndarray, rcm_pose: np.ndarray) -> np.ndarray:
    """Calculates the 7x1 pose from state of a tool and a remote center of motion.

    Notes:
        Transformation assumes that the tool axis is aligned with the z axis, the x axis points to the left, and the y axis points up.

    Args:
        ptsd (np.ndarray): [pan, tilt, spin, depth] describing the horizontal, vertical, axis rotation and the insertion depth.
        rcm_pose (np.ndarray): [x, y, z, X, Y, Z] Cartesian position and XYZ euler angles of the remote center of motion.

    Returns:
        pose (np.ndarray): Global [x, y, z, a, b, c, w] pose of the tool described by Cartesian position and quaternion.
    """

    # Turn the rotations into quaternions.
    rcm_quat = euler_angles_to_quaternion(rcm_pose[3:])
    pan, tilt, spin, depth = ptsd
    pts_quat = euler_angles_to_quaternion(np.array([tilt, pan, spin]))

    # Multiply the rotations.
    rotation = multiply_quaternions(rcm_quat, pts_quat)

    # Calculate the dual quaternion that rotates by the determined rotation and then translates
    # to the position of the remote center of motion.
    q = dquat_rotate_and_translate(rotation, rcm_pose[:3])
    position = fast_dquat_apply(q, np.asarray([0.0, 0.0, depth]))
    return np.hstack((position, rotation))


def quat_ptsd_to_pose(ptsd: np.ndarray, rcm_pose: np.ndarray) -> np.ndarray:
    """Calculates the 7x1 pose from state of a tool and a remote center of motion.

    Notes:
        Transformation assumes that the tool axis is aligned with the z axis, the x axis points to the left, and the y axis points up.

    Args:
        ptsd (np.ndarray): [pan, tilt, spin, depth] describing the horizontal, vertical, axis rotation and the insertion depth.
        rcm_pose (np.ndarray): [x, y, z, X, Y, Z] Cartesian position and XYZ euler angles of the remote center of motion.

    Returns:
        pose (np.ndarray): Global [x, y, z, a, b, c, w] pose of the tool described by Cartesian position and quaternion.
    """
    # Turn the rotations into quaternions.
    rcm_quat = euler_angles_to_quaternion(rcm_pose[3:])
    pan, tilt, spin, depth = ptsd
    pts_quat = euler_angles_to_quaternion(np.array([tilt, pan, spin]))

    # Multiply the rotations.
    rotation = multiply_quaternions(rcm_quat, pts_quat)

    # Calculate the dual quaternion that rotates by the determined rotation and then translates
    # to the position of the remote center of motion.
    position = point_rotation_by_quaternion(np.asarray([0.0, 0.0, depth]), rotation) + rcm_pose[:3]
    return np.hstack((position, rotation))


def quat_ptsd_to_pose_with_offset(ptsd: np.ndarray, rcm_pose: np.ndarray, offset: np.ndarray) -> np.ndarray:
    # Turn the rotations into quaternions.
    rcm_quat = euler_angles_to_quaternion(rcm_pose[3:])
    pan, tilt, spin, depth = ptsd
    pts_quat = euler_angles_to_quaternion(np.array([tilt, pan, spin]))

    # Multiply the rotations.
    rotation = multiply_quaternions(rcm_quat, pts_quat)

    # Calculate the dual quaternion that rotates by the determined rotation and then translates
    # to the position of the remote center of motion.
    position = point_rotation_by_quaternion(np.asarray([0.0, 0.0, depth]) + offset, rotation) + rcm_pose[:3]
    return np.hstack((position, rotation))


def generate_dquat_ptsd_to_pose(rcm_pose: np.ndarray) -> Callable:
    """Parametrizes ``dquat_ptsd_to_pose`` with a fixed remote center of motion.

    Args:
        rcm_pose (np.ndarray): [x, y, z, X, Y, Z] Cartesian position and XYZ euler angles of the remote center of motion.

    Returns:
        ptsd_to_pose (Callable): A parametrized ``dquat_ptsd_to_pose`` function with a fixed remote center of motion.
    """
    return lambda ptsd: dquat_ptsd_to_pose(ptsd, rcm_pose)


def dquat_oblique_viewing_endoscope_ptsd_to_pose(ptsd: np.ndarray, rcm_pose: np.ndarray, viewing_angle: float) -> np.ndarray:
    """Calculates the 7x1 pose from state of an oblique viewing endoscope and a remote center of motion.

    Notes:
        Transformation assumes that the endoscope axis is aligned with the negative z axis, the x axis points to the right, and the y axis points up.

    Args:
        ptsd (np.ndarray): [pan, tilt, spin, depth] describing the horizontal, vertical, axis rotation and the insertion depth.
        rcm_pose (np.ndarray): [x, y, z, X, Y, Z] Cartesian position and XYZ euler angles of the remote center of motion.
        viewing_angle (float): Angle between the endoscope axis and the viewing directon in degrees. Rotation around X.

    Returns:
        pose (np.ndarray): Global [x, y, z, a, b, c, w] pose of the endoscope described by Cartesian position and quaternion.
    """

    # Turn the rotations into quaternions.
    rcm_quat = euler_angles_to_quaternion(rcm_pose[3:])
    pan, tilt, spin, depth = ptsd
    pts_quat = euler_angles_to_quaternion(np.array([tilt, pan, spin]))

    # Multiply the rotations.
    rotation = multiply_quaternions(rcm_quat, pts_quat)

    # Calculate the dual quaternion that rotates by the determined rotation and then translates
    # to the position of the remote center of motion.
    q = dquat_rotate_and_translate(rotation, rcm_pose[:3])

    v = np.asarray([0.0, 0.0, depth])
    position = dquat_apply(q, v)

    # Apply OpenGL rotation.
    rotation = multiply_quaternions(rotation, np.array([0.0, 1.0, 0.0, 0.0]))
    # Apply viewing angle.
    va = viewing_angle * np.pi / 180
    rotation = multiply_quaternions(rotation, np.array([np.sin(-va / 2), 0.0, 0.0, np.cos(-va / 2)], dtype=float))

    return np.hstack((position, rotation))


def generate_dquat_oblique_viewing_endoscope_ptsd_to_pose(rcm_pose: np.ndarray, viewing_angle: float) -> Callable:
    """Parametrizes ``dquat_oblique_viewing_endoscope_ptsd_to_pose`` with a fixed remote center of motion and fixed viewing_angle.

    Args:
        rcm_pose (np.ndarray): [x, y, z, X, Y, Z] Cartesian position and XYZ euler angles of the remote center of motion.
        viewing_angle (float): Angle between the endoscope axis and the viewing directon in degrees. Rotation around X.

    Returns:
        oblique_viewing_endoscope_ptsd_to_pose (Callable): A parametrized ``dquat_oblique_viewing_endoscope_ptsd_to_pose`` function with a fixed remote center of motion and fixed viewing_angle.
    """
    return lambda ptsd: dquat_oblique_viewing_endoscope_ptsd_to_pose(ptsd, rcm_pose, viewing_angle)
