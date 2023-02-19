import numpy as np

from typing import Union

from sofa_env.sofa_templates.camera import Camera

from sofa_env.utils.dquat_inverse_pivot_transform import pose_to_ptsd
from sofa_env.utils.dquat_pivot_transform import dquat_ptsd_to_pose
from sofa_env.utils.math_helper import point_rotation_by_quaternion


def move_relative_to_camera(object_pose: np.ndarray, motion_vector: np.ndarray, rcm_pose: np.ndarray, camera: Union[Camera, np.ndarray]):
    """Moves an object relative to the view of the camera.

    Args:
        object_pose (np.ndarray): [x, y, z, a, b, c, w] The object's pose current pose.
        motion_vector (np.ndarray): three-dimensional vector that describes the movement of the object relative to the camera's view.
        rcm_pose (np.ndarray): [x, y, z, X, Y, Z] Cartesian position and XYZ euler angles of the remote center of motion.
        camera (Union[Camera, np.ndarray]): The camera observing the scene.
        Either as a ``sofa_env.sofa_templates.camera.Camera`` object or as a [x, y, z, a, b, c, w] pose of position and quaternion.

    Returns:
        pose (np.ndarray): Resulting pose after applying the vector
    """

    if isinstance(camera, Camera):
        camera_pose = camera.get_pose()
    else:
        camera_pose = camera

    camera_orientation = camera_pose[3:]

    translation = point_rotation_by_quaternion(motion_vector, camera_orientation)

    target_pose = np.copy(object_pose)
    target_pose[:3] = target_pose[:3] + translation

    ptsd = pose_to_ptsd(target_pose, rcm_pose)
    pose = dquat_ptsd_to_pose(ptsd, rcm_pose)

    return pose
