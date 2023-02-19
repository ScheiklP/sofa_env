import numpy as np
from typing import Union, List

from sofa_env.utils.dquat_pivot_transform import quat_ptsd_to_pose


def create_linear_motion_plan(
    target_state: np.ndarray,
    current_state: np.ndarray,
    dt: float,
    velocity: Union[float, np.ndarray],
) -> List[np.ndarray]:
    """Creates a list of waypoints from current to target state discretized along a linear path with velocity and time step dt.

    Args:
        target_state (np.ndarray): The final point in the motion plan.
        current_state (np.ndarray): The first point in the motion plan.
        dt (float): Time step discretization to correctly scale velocity.
        velocity (Union[float, np.ndarray]): Uniform or per dimension velocity of the motion.

    Returns:
        motion_plan (List[np.ndarray]): Waypoints along linear path.
    """

    state_delta = target_state - current_state

    state_delta_per_step = velocity * dt

    motion_steps = int(np.ceil(np.max(np.abs(state_delta / state_delta_per_step))))

    motion_plan = np.linspace(current_state, target_state, num=motion_steps)

    return list(motion_plan)


def create_ptsd_motion_plan(
    target_ptsd: np.ndarray,
    current_ptsd: np.ndarray,
    dt: float,
    velocity: Union[float, np.ndarray],
) -> List[np.ndarray]:
    """Creates a list of [X, Y, Z, *quaternion] waypoints from current to target ptsd state discretized along a linear path with velocity and time step dt.

    Args:
        target_ptsd (np.ndarray): The final state in the motion plan.
        current_ptsd (np.ndarray): The first state in the motion plan.
        dt (float): Time step discretization to correctly scale velocity.
        velocity (Union[float, np.ndarray]): Uniform or per dimension velocity of the motion.

    Returns:
        motion_plan (List[np.ndarray]): Waypoints along linear path.
    """

    rcm_pose = np.zeros(6)  # any RCM works here
    target_point = quat_ptsd_to_pose(target_ptsd, rcm_pose)
    current_point = quat_ptsd_to_pose(current_ptsd, rcm_pose)
    motion_steps = int(np.ceil(np.max(np.abs((target_point - current_point) / velocity / dt))))
    motion_plan = np.linspace(current_ptsd, target_ptsd, num=motion_steps)
    return list(motion_plan)


def create_linear_motion_action_plan(
    target_state: np.ndarray,
    current_state: np.ndarray,
    dt: float,
    velocity: Union[float, np.ndarray],
) -> List[np.ndarray]:
    """Creates a list of actions from current to target state discretized along a linear path with velocity and time step dt.

    Args:
        target_state (np.ndarray): The final point in the motion plan.
        current_state (np.ndarray): The first point in the motion plan.
        dt (float): Time step discretization to correctly scale velocity.
        velocity (Union[float, np.ndarray]): Uniform or per dimension velocity of the motion.

    Returns:
        action_plan (List[np.ndarray]): Actions along linear path.
    """

    linear_motion_plan = create_linear_motion_plan(
        target_state,
        current_state,
        dt,
        velocity,
    )

    action_plan = []
    for i in range(len(linear_motion_plan) - 1):
        action_plan.append((linear_motion_plan[i + 1] - linear_motion_plan[i]))

    return action_plan
