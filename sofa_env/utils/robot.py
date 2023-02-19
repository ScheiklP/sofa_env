import numpy as np
from typing import Tuple, Union, Callable

from sofa_env.utils.math_helper import quaternion_from_vectors


def get_main_link_pose_transformation(
    base_vector: Union[np.ndarray, Tuple[float, float, float]],
    remote_center_of_motion: Union[np.ndarray, Tuple[float, float, float]],
    link_offset: Union[np.ndarray, Tuple[float, float, float]],
) -> Callable:
    """Parametrizes the function that transforms the pose of the gripper part of the PSM to the pose of the PSM main link.

    Args:
        base_vector (Union[np.ndarray, Tuple[float, float, float]]): Unit vector that describes the axis of the main link model (e.g. [0, 1, 0]).
        remote_center_of_motion (Union[np.ndarray, Tuple[float, float, float]]): XYZ coordinates of the remote center of motion.
        link_offset (Union[np.ndarray, Tuple[float, float, float]]): XYZ offset between the center of the gripper and main link models.

    Returns:
        transform_main_link_pose (Callable): The transform function that returns the pose of the PSM main link, given the pose of the gripper.

    """

    base_vector = np.asarray(base_vector)
    remote_center_of_motion = np.asarray(remote_center_of_motion)
    link_offset = np.asarray(link_offset)

    def transform_main_link_pose(gripper_position: np.ndarray) -> np.ndarray:
        """Calculates the pose of a PSM main link given a PSM gripper position.

        Args:
            gripper_position (np.ndarray): XYZ position of the gripper.

        Returns:
            main_link_pose (np.ndarray): Pose of the PSM main link, given the pose of the gripper.
        """

        reference_point = gripper_position + link_offset

        from_gripper_to_remote_center_of_motion = remote_center_of_motion - reference_point

        quaternion = quaternion_from_vectors(base_vector, from_gripper_to_remote_center_of_motion)

        main_link_pose = np.append(reference_point, quaternion)

        return main_link_pose

    return transform_main_link_pose
