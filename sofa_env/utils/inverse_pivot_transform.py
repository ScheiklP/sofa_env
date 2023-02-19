import numpy as np

from sofa_env.utils.math_helper import euler_to_rotation_matrix, quaternion_to_euler_angles, rotation_matrix_to_euler, rotation_matrix_from_vectors


def spin_matrix_close_to(m: np.ndarray, target: np.ndarray) -> np.ndarray:
    _, _, spin_target = quaternion_to_euler_angles(target)
    _, _, spin_m = rotation_matrix_to_euler(m)

    angle = (spin_target - spin_m) * np.pi / 180
    spinner = np.eye(3)
    c = np.cos(angle)
    s = np.sin(angle)
    spinner[:2, :2] = np.array([[c, -s], [s, c]])

    return m @ spinner


def target_pose_to_ptsd(target_pose: np.ndarray, rcm_pose: np.ndarray, error_on_unreachable: bool = False, link_offset: np.ndarray = np.zeros(3), base_vector: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:

    # Calculate the inverse RCM matrix
    rcm_rotation = euler_to_rotation_matrix(rcm_pose[3:])
    rcm_rotation_T = rcm_rotation.transpose()

    length = np.linalg.norm(target_pose[:3] - rcm_pose[:3])
    min_offset = link_offset[0] ** 2 + link_offset[1] ** 2
    if min_offset <= length ** 2:
        depth = np.sqrt(length ** 2 - min_offset)
        depth = depth - link_offset[2]
    elif error_on_unreachable:
        raise RuntimeError(f"Target position {target_pose[:3]} in pose {target_pose} is unreachable with given offset of {link_offset}.")
    else:
        depth = -min_offset

    target_rotation = rotation_matrix_from_vectors(depth * base_vector + link_offset, target_pose[:3] - rcm_pose[:3])
    target_rotation = spin_matrix_close_to(target_rotation, target_pose[3:])
    pts_matrix = rcm_rotation_T @ target_rotation
    tilt, pan, spin = rotation_matrix_to_euler(pts_matrix)

    return np.array([pan, tilt, spin, depth])
