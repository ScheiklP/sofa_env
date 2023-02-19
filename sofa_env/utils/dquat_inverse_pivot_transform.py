import numpy as np

from sofa_env.utils.math_helper import conjugate_quaternion, multiply_quaternions, quaternion_from_vectors, rotated_z_axis, euler_angles_to_quaternion, quaternion_to_euler_angles


def spin_quaternion_close_to(quat: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotates a quaternion around the z axis so that it is close to the target quaternion.

    Note:
        We use this function to reorient a target pose such that the tip of the end-effector is still
        at the same location, but rotate the instrument around its main axis to find a pose that is
        as close as possible to the given target.

    Args:
        quat (np.ndarray): The quaternion to be modified.
        target_quat (np.ndarray): The quaternion that ``quat`` is supposed to be close to.

    Returns:
        q_ (np.ndarray): The rotated version of ``quat``.

    Example:
        >>> target = np.array([0, 0, np.sqrt(.75), np.sqrt(.25)]) # spin=120°
        >>> quat = np.array([0, np.sqrt(.25), np.sqrt(.25), np.sqrt(.5)]) # pan=-45°, tilt=45°, spin=90°
        >>> spin_quaternion_close_to(quat, target) = np.array([0.12940952 0.48296291 0.66597562 0.55360318]) # pan=-45°, tilt=45°, spin=120°
    """
    _, _, spin_target = quaternion_to_euler_angles(target)
    _, _, spin_q = quaternion_to_euler_angles(quat)

    angle = (spin_target - spin_q) * np.pi / 180
    spinner = np.empty(4)
    spinner[:3] = np.sin(angle / 2) * rotated_z_axis(quat)
    spinner[3] = np.cos(angle / 2)

    return multiply_quaternions(spinner, quat)


def pose_to_ptsd(target_pose: np.ndarray, rcm_pose: np.ndarray, error_on_unreachable: bool = False, link_offset: np.ndarray = np.zeros(3)) -> np.ndarray:
    """Calculates the necessary pan, tilt, spin and depth so that the end effector will reach a target pose.

    Note:
        If the target pose is unreachable, there are two things that can happen:
        1) There will be an error
        2) The function returns a ptsd which allow the position (XYZ) to be reached but not the orientation. However,
            the spin will be the same as in the target orientation.

    Args:
        target_pose (np.ndarray): [x, y, z, a, b, c, w] The target position and orientation (as a unit quaternion).
        rcm_pose (np.ndarray): [x, y, z, X, Y, Z] Cartesian position and XYZ euler angles of the remote center of motion.
        error_on_unreachable (bool): When set to ``True``, the function will return an error if the target position is unreachable.
        link_offset (np.ndarray): XYZ offset of the end-effector to the actually controlled point in the end-effector's coordinate system.

    Returns:
        ptsd (np.ndarray): pan, tilt, spin and depth to reach the target pose.

    Example:
        >>> target_pose = np.array([1.0, 1.0, 0.0, np.sqrt(.5), 0.0, 0.0, np.sqrt(.5)])
        >>> rcm_pose = np.array([0.0, 0.0, 4.0, 0.0, 90.0, 0.0])
        >>> ptsd_state = pose_to_ptsd(target_pose, rcm_pose)
        >>> ptsd_to_pose(ptsd_state, rcm_pose) = np.array(
                [0.9999999999999981, 0.9999999999999978, -8.881784197001252e-16,
                0.9854918641895205, -0.014501944948706718, 0.11780083059524492, -0.1213195924821651])
        >>> instrument.set_state(ptsd_state)
    """

    # Calculate the inverse RCM transform dual quaternion
    rcm_rotation_quaternion = euler_angles_to_quaternion(rcm_pose[3:])
    q_rcm_rotation_inverse = conjugate_quaternion(np.copy(rcm_rotation_quaternion))

    # Convert the target into a dual quaternion
    # 1) Calculate the depth: || [0, 0, depth] + link_offset|| must be equal to || target - rcm ||
    length = np.linalg.norm(target_pose[:3] - rcm_pose[:3])
    min_offset = link_offset[0] ** 2 + link_offset[1] ** 2
    if min_offset <= length ** 2:
        depth = np.sqrt(length ** 2 - min_offset)
        depth = depth - link_offset[2]
    elif error_on_unreachable:
        raise RuntimeError(f"Target position {target_pose[:3]} in pose {target_pose} is unreachable with given offset of {link_offset}.")
    else:
        depth = -min_offset

    # 2) Use the depth to obtain the rotation
    q_target_rotation = quaternion_from_vectors(np.array([0, 0, depth]) + link_offset, target_pose[:3] - rcm_pose[:3])
    q_target_rotation = spin_quaternion_close_to(q_target_rotation, target_pose[3:])

    if error_on_unreachable and not np.allclose(q_target_rotation, target_pose[3:], atol=1e-5):
        raise RuntimeError(f"Target pose {target_pose} is unreachable.")

    # Calculate the quaternion that describes pan, tilt and spin
    q_pts = multiply_quaternions(q_rcm_rotation_inverse, q_target_rotation)
    tilt, pan, spin = quaternion_to_euler_angles(q_pts)

    return np.array([pan, tilt, spin, depth])
