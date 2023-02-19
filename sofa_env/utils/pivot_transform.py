import numpy as np
from typing import Callable

from sofa_env.utils.math_helper import euler_to_rotation_matrix, homogeneous_transform_to_pose


def ptsd_to_pose(ptsd: np.ndarray, rcm_pose: np.ndarray) -> np.ndarray:
    """Calculates the 7x1 pose from state of a tool and a remote center of motion.

    Notes:
        Transformation assumes that the tool axis is aligned with the z axis, the x axis points to the left, and the y axis points up.

    Args:
        ptsd (np.ndarray): [pan, tilt, spin, depth] describing the horizontal, vertical, axis rotation and the insertion depth.
        rcm_pose (np.ndarray): [x, y, z, X, Y, Z] Cartesian position and XYZ euler angles of the remote center of motion.

    Returns:
        pose (np.ndarray): Global [x, y, z, a, b, c, w] pose of the tool described by Cartesian position and quaternion.
    """

    rcm_transform = np.eye(4)
    rcm_transform[:3, :3] = euler_to_rotation_matrix(rcm_pose[3:])
    rcm_transform[:3, 3] = rcm_pose[:3]

    transform = rcm_transform

    tool_rotation = np.eye(4)
    pan = ptsd[0]
    tilt = ptsd[1]
    spin = ptsd[2]
    tool_euler_angles = np.array([tilt, pan, spin])
    tool_rotation[:3, :3] = euler_to_rotation_matrix(tool_euler_angles)

    transform = transform @ tool_rotation

    tool_translation = np.eye(4)
    tool_translation[:3, 3] = np.array([0.0, 0.0, ptsd[3]])

    transform = transform @ tool_translation

    return homogeneous_transform_to_pose(transform)


def generate_ptsd_to_pose(rcm_pose: np.ndarray) -> Callable:
    """Parametrizes ``ptsd_to_pose`` with a fixed remote center of motion.

    Args:
        rcm_pose (np.ndarray): [x, y, z, X, Y, Z] Cartesian position and XYZ euler angles of the remote center of motion.

    Returns:
        ptsd_to_pose (Callable): A parametrized ``ptsd_to_pose`` function with a fixed remote center of motion.
    """

    def ptsd_to_pose(ptsd: np.ndarray) -> np.ndarray:

        rcm_transform = np.eye(4)
        rcm_transform[:3, :3] = euler_to_rotation_matrix(rcm_pose[3:])
        rcm_transform[:3, 3] = rcm_pose[:3]

        transform = rcm_transform

        tool_rotation = np.eye(4)
        pan = ptsd[0]
        tilt = ptsd[1]
        spin = ptsd[2]
        tool_euler_angles = np.array([tilt, pan, spin])
        tool_rotation[:3, :3] = euler_to_rotation_matrix(tool_euler_angles)

        transform = transform @ tool_rotation

        tool_translation = np.eye(4)
        tool_translation[:3, 3] = np.array([0.0, 0.0, ptsd[3]])

        transform = transform @ tool_translation

        return homogeneous_transform_to_pose(transform)

    return ptsd_to_pose


def oblique_viewing_endoscope_ptsd_to_pose(ptsd: np.ndarray, rcm_pose: np.ndarray, viewing_angle: float) -> np.ndarray:
    """Calculates the 7x1 pose from state of an oblique viewing endoscope and a remote center of motion.

    Notes:
        Transformation assumes that the endoscope axis is aligned with the negative z axis, the x axis points to the right, and the y axis points up.

    Args:
        ptsd (np.ndarray): [pan, tilt, spin, depth] describing the horizontal, vertical, axis rotation and the insertion depth.
        rcm_pose (np.ndarray): [x, y, z, X, Y, Z] Cartesian position and XYZ euler angles of the remote center of motion.
        viewing_angle (float): Angle between the endoscope axis and the viewing directon. Rotation around X.

    Returns:
        pose (np.ndarray): Global [x, y, z, a, b, c, w] pose of the endoscope described by Cartesian position and quaternion.
    """

    rcm_transform = np.eye(4)
    rcm_transform[:3, :3] = euler_to_rotation_matrix(rcm_pose[3:])
    rcm_transform[:3, 3] = rcm_pose[:3]

    transform = rcm_transform

    endoscope_rotation = np.eye(4)
    pan = ptsd[0]
    tilt = ptsd[1]
    spin = ptsd[2]
    endoscope_euler_angles = np.array([tilt, pan, spin])
    endoscope_rotation[:3, :3] = euler_to_rotation_matrix(endoscope_euler_angles)

    transform = transform @ endoscope_rotation

    endoscope_translation = np.eye(4)
    endoscope_translation[:3, 3] = np.array([0.0, 0.0, ptsd[3]])

    transform = transform @ endoscope_translation

    opengl_rotation = np.eye(4)
    opengl_euler_angles = np.array([0.0, 180.0, 0.0])
    opengl_rotation[:3, :3] = euler_to_rotation_matrix(opengl_euler_angles)

    transform = transform @ opengl_rotation

    viewing_angle_rotation = np.eye(4)
    viewing_angle_euler_angles = np.array([viewing_angle, 0.0, 0.0])
    viewing_angle_rotation[:3, :3] = euler_to_rotation_matrix(viewing_angle_euler_angles)

    transform = transform @ viewing_angle_rotation

    return homogeneous_transform_to_pose(transform)


def generate_oblique_viewing_endoscope_ptsd_to_pose(rcm_pose: np.ndarray, viewing_angle: float) -> Callable:
    """Parametrizes ``oblique_viewing_endoscope_ptsd_to_pose`` with a fixed remote center of motion and fixed viewing_angle.

    Args:
        rcm_pose (np.ndarray): [x, y, z, X, Y, Z] Cartesian position and XYZ euler angles of the remote center of motion.
        viewing_angle (float): Angle between the endoscope axis and the viewing directon. Rotation around X.

    Returns:
        oblique_viewing_endoscope_ptsd_to_pose (Callable): A parametrized ``oblique_viewing_endoscope_ptsd_to_pose`` function with a fixed remote center of motion and fixed viewing_angle.
    """

    def oblique_viewing_endoscope_ptsd_to_pose(ptsd: np.ndarray, rcm_pose=rcm_pose, viewing_angle=viewing_angle) -> np.ndarray:
        rcm_transform = np.eye(4)
        rcm_transform[:3, :3] = euler_to_rotation_matrix(rcm_pose[3:])
        rcm_transform[:3, 3] = rcm_pose[:3]

        transform = rcm_transform

        endoscope_rotation = np.eye(4)
        pan = ptsd[0]
        tilt = ptsd[1]
        spin = ptsd[2]
        endoscope_euler_angles = np.array([tilt, pan, spin])
        endoscope_rotation[:3, :3] = euler_to_rotation_matrix(endoscope_euler_angles)

        transform = transform @ endoscope_rotation

        endoscope_translation = np.eye(4)
        endoscope_translation[:3, 3] = np.array([0.0, 0.0, ptsd[3]])

        transform = transform @ endoscope_translation

        opengl_rotation = np.eye(4)
        opengl_euler_angles = np.array([0, 180.0, 180.0])
        opengl_rotation[:3, :3] = euler_to_rotation_matrix(opengl_euler_angles)

        transform = transform @ opengl_rotation

        viewing_angle_rotation = np.eye(4)
        viewing_angle_euler_angles = np.array([-viewing_angle, 0.0, 0.0])
        viewing_angle_rotation[:3, :3] = euler_to_rotation_matrix(viewing_angle_euler_angles)

        transform = transform @ viewing_angle_rotation

        return homogeneous_transform_to_pose(transform)

    return oblique_viewing_endoscope_ptsd_to_pose


def sofa_pose_to_camera_pose(pose: np.ndarray) -> np.ndarray:
    """Converts a pose from the SOFA convention to the camera convention (z towards camera, y up).

    Args:
        pose (np.ndarray): [x, y, z, a, b, c, w] pose of the endoscope described by Cartesian position and quaternion.

    Returns:
        camera_pose (np.ndarray): [x, y, z, a, b, c, w] pose of the endoscope described by Cartesian position and quaternion.
    """

    camera_pose = pose.copy()

    # Like multiply_quaternions([1.0, 0.0, 0.0, 0.0], pose[3:])
    orientation = pose[3:]
    camera_pose[3:] = [orientation[3], orientation[2], -orientation[1], -orientation[0]]

    return camera_pose


def sofa_orientation_to_camera_orientation(sofa_quaternion: np.ndarray) -> np.ndarray:
    """Converts a quaterion from the SOFA convention to the camera convention (z towards camera, y up).

    Args:
        sofa_quaterion (np.ndarray): [a, b, c, w] orientation of the endoscope described by a quaternion.

    Returns:
        camera_quaternion (np.ndarray): [a, b, c, w] pose of the endoscope described by a quaternion.
    """

    # Like multiply_quaternions([1.0, 0.0, 0.0, 0.0], pose[3:])
    return np.array([sofa_quaternion[3], sofa_quaternion[2], -sofa_quaternion[1], -sofa_quaternion[0]])
