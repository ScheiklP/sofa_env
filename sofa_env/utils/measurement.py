import time
from typing import Callable, TypeVar

import numpy as np
from sofa_env.utils.dquat_inverse_pivot_transform import pose_to_ptsd

from sofa_env.utils.dquat_pivot_transform import dquat_ptsd_to_pose, quat_ptsd_to_pose
from sofa_env.utils.inverse_pivot_transform import target_pose_to_ptsd
from sofa_env.utils.math_helper import point_rotation_by_quaternion, quaternion_to_euler_angles
from sofa_env.utils.pivot_transform import ptsd_to_pose
from sofa_env.utils.relative_camera_movement import move_relative_to_camera

T = TypeVar("T")


def compare_complexity(*fs: Callable[[T], any], generator: Callable[[], T], iterations: int = 10 ** 5) -> list[tuple[float, float]]:
    """
    Generates a new parameter sample for each iteration and runs every given function on that sample. Measures the computation time of
    each function and returns the mean and standard deviation of every function.

    Args:
        *fs (Callable[[T], any]): The functions to examine. They should all require the same type of parameters.
        generator (Callable[[], T]): Generator function for the parameters. For each call, it should generate a new valid random parameter sample.
        iterations (int): The number of samples to be generated. It should be a large number for meaningful results.
    """
    lists: list[list[float]] = [[] for _ in fs]
    for _ in range(iterations):
        parameters = generator()

        for index, f in enumerate(fs):
            t = time.perf_counter()
            f(*parameters)
            dt = time.perf_counter() - t
            lists[index].append(dt)

    return [(np.mean(l), np.std(l)) for l in lists]


def calculate_accuracy(fwd: Callable, inv: Callable, iterations: int = 10 ** 5) -> tuple[tuple[float, float], tuple[float, float]]:
    position_errors = []
    orientation_errors = []
    scale = 1.0
    for _ in range(iterations):
        rcm = np.random.rand(6)
        rcm[:3] *= scale
        rcm[3:] *= 360.0

        quat = 2.0 * np.random.rand(4) - 1.0
        quat /= np.linalg.norm(quat)
        position = point_rotation_by_quaternion(np.array([0.0, 0.0, scale]), quat) + rcm[:3]
        pose = np.hstack((position, quat))

        ptsd = inv(pose, rcm)
        pose_ = fwd(ptsd, rcm)
        position_errors.append(np.linalg.norm(pose[:3] - pose_[:3]))
        orientation_errors.append(1.0 - np.abs(np.dot(pose[3:], pose_[3:])))
    return ((np.mean(position_errors), np.std(position_errors)), (np.mean(orientation_errors), np.std(orientation_errors)))


def random_pose():
    random_quat = np.random.rand(4)
    random_quat /= np.linalg.norm(random_quat)
    return np.hstack((np.random.rand(3), random_quat))


def __measure_forward_kinematics():
    ptsd_to_pose_generator = lambda: (360 * np.random.rand(4), 360 * np.random.rand(6))
    values = compare_complexity(dquat_ptsd_to_pose, quat_ptsd_to_pose, ptsd_to_pose, generator=ptsd_to_pose_generator, iterations=10 ** 3)

    for m, std in values:
        print(f"Mean: {m}, Standard deviation: {std}")

    for m, std in values:
        print(f"Mean: {m / values[2][0]}, Standard deviation: {std / values[2][1]}")


def __measure_inverse_kinematics():
    values = compare_complexity(target_pose_to_ptsd, pose_to_ptsd, generator=lambda: (random_pose(), np.random.rand(6)), iterations=100)
    print(values)


def __measure_relative_camera():
    rel_cam_generator = lambda: (random_pose(), np.random.rand(3), np.random.rand(6), random_pose())
    print(compare_complexity(move_relative_to_camera, generator=rel_cam_generator))


if __name__ == "__main__":
    __measure_forward_kinematics()
    """
    # Numerical accuracy of forward and inverse kinematics
    print(calculate_accuracy(dquat_ptsd_to_pose, pose_to_ptsd, 10**3))

    rcm = np.random.rand(6)
    rcm[3:] *= 360.0

    quat = 2.0 * np.random.rand(4) - 1.0
    quat /= np.linalg.norm(quat)
    position = point_rotation_by_quaternion(np.array([0.0, 0.0, 1]), quat) + rcm[:3]
    pose = np.hstack((position, quat))

    offset = np.random.rand(3) * 5.0

    print(pose_to_ptsd(pose, rcm, link_offset=offset))
    print(target_pose_to_ptsd(pose, rcm, link_offset=offset))
    """
