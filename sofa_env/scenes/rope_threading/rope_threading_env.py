import gymnasium.spaces as spaces
import numpy as np

from collections import deque, defaultdict
from enum import Enum, unique
from pathlib import Path

from typing import Callable, Union, Tuple, Optional, List, Any, Dict

from sofa_env.base import SofaEnv, RenderMode, RenderFramework
from sofa_env.scenes.rope_threading.sofa_objects.eye import Eye, EyeStates
from sofa_env.scenes.rope_threading.sofa_objects.gripper import ArticulatedGripper
from sofa_env.scenes.rope_threading.sofa_objects.transfer_rope import TransferRope

from sofa_env.sofa_templates.camera import Camera
from sofa_env.utils.motion_planning import create_linear_motion_action_plan

HERE = Path(__file__).resolve().parent
SCENE_DESCRIPTION_FILE_PATH = HERE / "scene_description.py"


def follow_waypoints(waypoints: List, env) -> None:
    action = env.action_space.sample()

    for i in range(len(waypoints) - 1):
        for key in action:
            action[key][:] = 0.0

        instrument_change = waypoints[i + 1][0] != waypoints[i][0]

        if instrument_change:
            current_state = env.left_gripper.ptsd_state if waypoints[i + 1][0] == "left_gripper" else env.right_gripper.ptsd_state
        else:
            current_state = waypoints[i][1]

        linear_motion_action_plan = create_linear_motion_action_plan(target_state=waypoints[i + 1][1], current_state=current_state, velocity=env._maximum_state_velocity[:4], dt=env.time_step)
        linear_motion_action_plan /= env._maximum_state_velocity[:4] * env.time_step

        for planned_action in linear_motion_action_plan:
            action[waypoints[i + 1][0]][:4] = np.clip(planned_action, -np.ones_like(planned_action), np.ones_like(planned_action))
            env.step(action)


def change_grasp(left_to_right: bool, env) -> None:
    action = env.action_space.sample()
    for key in action:
        action[key][:] = 0.0

    if left_to_right:
        action["right_gripper"][4] = -20.0 / (env._maximum_state_velocity[-1] * env.time_step)
        env.step(action)

        action["right_gripper"][4] = 0.0
        for _ in range(30):
            env.step(action)

        action["left_gripper"][4] = 20.0 / (env._maximum_state_velocity[-1] * env.time_step)

        env.step(action)

        action["left_gripper"][4] = 0.0
        for _ in range(30):
            env.step(action)

    else:
        action["left_gripper"][4] = -20.0 / (env._maximum_state_velocity[-1] * env.time_step)

        env.step(action)

        action["left_gripper"][4] = 0.0
        for _ in range(30):
            env.step(action)

        action["right_gripper"][4] = 20.0 / (env._maximum_state_velocity[-1] * env.time_step)

        env.step(action)

        action["right_gripper"][4] = 0.0
        for _ in range(30):
            env.step(action)


def wait_for_n(n, env) -> None:
    action = env.action_space.sample()

    for key in action:
        action[key][:] = 0.0

    for _ in range(n):
        env.step(action)


@unique
class ObservationType(Enum):
    RGB = 0
    STATE = 1
    RGBD = 2


@unique
class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1
    VELOCITY = 2
    POSITION = 3


class RopeThreadingEnv(SofaEnv):
    """Rope Threading Environment

    Args:
        scene_path (Union[str, Path]): Path to the scene description script that contains this environment's ``createScene`` function.
        image_shape (Tuple[int, int]): Height and Width of the rendered images.
        observation_type (ObservationType): Whether to return RGB images or an array of states as the observation.
        time_step (float): size of simulation time step in seconds (default: 0.01).
        frame_skip (int): number of simulation time steps taken (call ``_do_action`` and advance simulation) each time step is called (default: 1).
        settle_steps (int): How many steps to simulate without returning an observation after resetting the environment.
        render_mode (RenderMode): create a window (``RenderMode.HUMAN``) or run headless (``RenderMode.HEADLESS``).
        render_framework (RenderFramework): choose between pyglet and pygame for rendering
        reward_amount_dict (dict): Dictionary to weigh the components of the reward function.
        maximum_state_velocity (Union[np.ndarray, float]): Velocity in deg/s for pts and angle, and mm/s for d in state space which are applied with a normalized action of value 1.
        fraction_of_rope_to_pass (float): Fraction of rope to pass through an eye to mark it as done.
        create_scene_kwargs (Optional[dict]): A dictionary to pass additional keyword arguments to the ``createScene`` function.
        on_reset_callbacks (Optional[List[Callable]]): Functions that are called as the last step of the ``env.reset()`` function.
        color_eyes (bool): Whether to color the eyes according to their state.
        individual_agents (bool): Whether the instruments are controlled individually, or the action is one large array.
        only_right_gripper (bool): Whether to remove the left gripper from the scene.
        action_type (ActionType): Discrete or continuous actions to define the action space of the environment.
        num_rope_tracking_points (int): Number of evenly spaced points on the rope to include in the state observation.
    """

    def __init__(
        self,
        scene_path: Union[str, Path] = SCENE_DESCRIPTION_FILE_PATH,
        image_shape: Tuple[int, int] = (124, 124),
        observation_type: ObservationType = ObservationType.RGB,
        time_step: float = 0.01,
        frame_skip: int = 3,
        settle_steps: int = 50,
        render_mode: RenderMode = RenderMode.HEADLESS,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        reward_amount_dict={
            "passed_eye": 1.0,
            "lost_eye": -2.0,  # more than passed_eye
            "successful_task": 10.0,
            "distance_to_active_eye": -0.0,
            "lost_grasp": -0.1,
            "collision": -0.01,
            "floor_collision": -0.01,
            "bimanual_grasp": 0.0,
            "distance_to_bimanual_grasp": -0.0,
            "delta_distance_to_bimanual_grasp": -0.0,
            "moved_towards_eye": 0.01,
            "moved_away_from_eye": -0.01,
            "workspace_violation": -0.01,
            "state_limit_violation": -0.01,
            "distance_to_lost_rope": -0.0,
            "delta_distance_to_lost_rope": -0.0,
            "fraction_rope_passed": 0.0,
            "delta_fraction_rope_passed": 0.0,
        },
        maximum_state_velocity: np.ndarray = np.array([12.0, 12.0, 25.0, 12.0, 5.0]),
        fraction_of_rope_to_pass: Optional[float] = None,
        create_scene_kwargs: Optional[dict] = None,
        on_reset_callbacks: Optional[List[Callable]] = None,
        color_eyes: bool = True,
        individual_agents: bool = True,
        only_right_gripper: bool = False,
        action_type: ActionType = ActionType.CONTINUOUS,
        num_rope_tracking_points: int = -1,
    ) -> None:
        # Pass image shape to the scene creation function
        if create_scene_kwargs is None:
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape
        create_scene_kwargs.setdefault("only_right_gripper", only_right_gripper)
        num_rope_points = create_scene_kwargs.setdefault("num_rope_points", 100)

        super().__init__(
            scene_path=scene_path,
            time_step=time_step,
            frame_skip=frame_skip,
            render_mode=render_mode,
            render_framework=render_framework,
            create_scene_kwargs=create_scene_kwargs,
        )

        # How many simulation steps to wait before starting the episode
        self._settle_steps = settle_steps

        # Whether to color the eyes based on task progress
        self.color_eyes = color_eyes

        # How much of the rope has to pass the eye, before it is marked as passed
        self.fraction_of_rope_to_pass = fraction_of_rope_to_pass

        ##############
        # Action Space
        ##############
        self.individual_agents = individual_agents
        self.single_agent = only_right_gripper

        self.action_type = action_type
        if action_type == ActionType.CONTINUOUS:
            action_space_limits = {
                "low": -1.0,
                "high": 1.0,
            }
            # Scale 1.0 to the maximum velocity
            self._maximum_state_velocity = maximum_state_velocity
            self._scale_action = self._scale_continuous_action
        elif action_type == ActionType.VELOCITY:
            action_space_limits = {
                "low": -maximum_state_velocity,
                "high": maximum_state_velocity,
            }
            # Do not scale the velocity, as it is already scaled
            self._maximum_state_velocity = 1.0
            self._scale_action = self._scale_continuous_action
        elif action_type == ActionType.POSITION:
            # Same as the state limits of the instruments
            action_space_limits = {
                "low": np.array([-90.0, -90.0, np.finfo(np.float16).min, 0.0, 0.0]),
                "high": np.array([90.0, 90.0, np.finfo(np.float16).max, 100.0, 60.0]),
            }
        else:
            raise NotImplementedError("RopeThreadingEnv currently only supports continuous actions.")

        if only_right_gripper:
            self.action_space = spaces.Box(low=action_space_limits["low"], high=action_space_limits["high"], shape=(5,), dtype=np.float32)
            self._do_action = self._do_action_array_single_agent
        else:
            if individual_agents:
                self.action_space = spaces.Dict(
                    {
                        "left_gripper": spaces.Box(low=action_space_limits["low"], high=action_space_limits["high"], shape=(5,), dtype=np.float32),
                        "right_gripper": spaces.Box(low=action_space_limits["low"], high=action_space_limits["high"], shape=(5,), dtype=np.float32),
                    }
                )
                self._do_action = self._do_action_dict
            else:
                if action_type == ActionType.CONTINUOUS:
                    self.action_space = spaces.Box(low=action_space_limits["low"], high=action_space_limits["high"], shape=(10,), dtype=np.float32)
                else:
                    self.action_space = spaces.Box(low=np.tile(action_space_limits["low"], 2), high=np.tile(action_space_limits["high"], 2), shape=(10,), dtype=np.float32)
                self._do_action = self._do_action_array

        ###################
        # Observation Space
        ###################
        # Identify the indices of points on the rope that should be used for describing the state of the rope.
        # If num_rope_tracking_points is -1, use all of them.
        match num_rope_tracking_points:
            case -1:
                self.num_rope_tracking_points = num_rope_points
                self.rope_tracking_point_indices = np.array(range(num_rope_points), dtype=np.int16)
            case 0:
                self.num_rope_tracking_points = 0
                self.rope_tracking_point_indices = None
            case n if 1 <= n <= num_rope_points:
                self.num_rope_tracking_points = n
                self.rope_tracking_point_indices = np.linspace(0, num_rope_points - 1, num=self.num_rope_tracking_points, endpoint=True, dtype=np.int16)
            case n if n > num_rope_points:
                raise ValueError(f"The number of rope tracking points ({num_rope_tracking_points}) is larger than the number of points on the rope ({num_rope_points}).")
            case _:
                raise ValueError(f"num_rope_tracking_points must be > 0 or == -1 (to use them all). Received {num_rope_tracking_points}.")

        # State observations
        if observation_type == ObservationType.STATE:
            # has_grasped -> 1 if single_agent else 2
            # ptsda_state -> 5 if single_agent else 10
            # gripper_pose -> 7 if single_agent else 14
            # position of rope tip -> 3
            # rope_tracking_point_positions -> num_rope_tracking_points * 3
            # active eye pose -> 4
            times = 1 if self.single_agent else 2
            observations_size = (1 + 5 + 7) * times + 3 + self.num_rope_tracking_points * 3 + 4
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observations_size,), dtype=np.float32)

        # Image observations
        elif observation_type == ObservationType.RGB:
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (3,), dtype=np.uint8)

        elif observation_type == ObservationType.RGBD:
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (4,), dtype=np.uint8)

        else:
            raise Exception(f"Please set observation_type to a value of ObservationType. Received {observation_type}.")

        self.observation_type = observation_type

        #########################
        # Episode specific values
        #########################

        # Distance to the active eye in the previous step
        self.distance_to_eye = np.inf

        # Current target eye
        self.active_index = 0

        # Infos per episode
        self.episode_info = defaultdict(float)

        # Infos from the reward
        self.reward_info = {}

        # All parameters of the reward function that are not passed default to 0.0
        self.reward_amount_dict = defaultdict(float) | reward_amount_dict

        # Callback functions called on reset
        if on_reset_callbacks is not None:
            self.on_reset_callbacks = on_reset_callbacks
        else:
            self.on_reset_callbacks = []

    def _init_sim(self) -> None:
        super()._init_sim()

        self.initial_rope_state: np.ndarray = self.scene_creation_result["rope"].get_state()
        self.left_gripper: Optional[ArticulatedGripper] = self.scene_creation_result["interactive_objects"]["left_gripper"]
        self.right_gripper: ArticulatedGripper = self.scene_creation_result["interactive_objects"]["right_gripper"]
        self.eyes: List[Eye] = self.scene_creation_result["eyes"]
        self.rope: TransferRope = self.scene_creation_result["rope"]
        self.contact_listeners: Dict[str, list] = self.scene_creation_result["contact_listeners"]
        self.camera: Camera = self.scene_creation_result["camera"]

        # Factor for normalizing the distances in the reward function.
        # Based on the workspace of the grippers.
        self._distance_normalization_factor = 1.0 / np.linalg.norm(self.right_gripper.cartesian_workspace["high"] - self.right_gripper.cartesian_workspace["low"])

    def constrained_value(self, value: float) -> float:
        """Constrains a value to lie within the workspace of the environment.

        In some cases, the simulation can become unstable, but not crash.
        In these cases, we want to constrain position values (that may be arbitrarily large) to lie within the workspace of the environment.

        Args:
            value (float): The value to constrain.

        Returns:
            The constrained value, clipped by the workspace.
        """
        if np.isnan(value):
            value = 1.0 / self._distance_normalization_factor
        value = np.clip(value, -1.0 / self._distance_normalization_factor, 1.0 / self._distance_normalization_factor)
        return value

    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        """Step function of the environment that applies the action to the simulation and returns observation, reward, done signal, and info."""
        maybe_rgb_observation = super().step(action)

        observation = self._get_observation(maybe_rgb_observation)
        reward = self._get_reward()
        terminated = self._get_done()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _do_action(self, action) -> None:
        """Only defined to satisfy ABC."""
        pass

    def _scale_continuous_action(self, action: np.ndarray) -> np.ndarray:
        """
        Policy is output is clipped to [-1, 1] e.g. [-0.3, 0.8, 1.0, -0.5, 0.7].
        We want to scale that to the maximum velocities defined
        in ``maximum_state_velocity`` in [angle, angle, angle, mm, angle] / step.
        and further to per second (because delta T is not 1 second).
        """
        return self.time_step * self._maximum_state_velocity * action

    def _do_action_dict(self, action: Dict[str, np.ndarray]) -> None:
        """Apply action to the simulation."""
        if self.action_type == ActionType.POSITION:
            self.left_gripper.set_articulated_state(action["left_gripper"])
            self.right_gripper.set_articulated_state(action["right_gripper"])
        else:
            self.left_gripper.set_articulated_state(self.left_gripper.get_articulated_state() + self._scale_action(action["left_gripper"]))
            self.right_gripper.set_articulated_state(self.right_gripper.get_articulated_state() + self._scale_action(action["right_gripper"]))

    def _do_action_array(self, action: np.ndarray) -> None:
        """Apply action to the simulation."""
        if self.action_type == ActionType.POSITION:
            self.left_gripper.set_articulated_state(action[:5])
            self.right_gripper.set_articulated_state(action[5:])
        else:
            self.left_gripper.set_articulated_state(self.left_gripper.get_articulated_state() + self._scale_action(action[:5]))
            self.right_gripper.set_articulated_state(self.right_gripper.get_articulated_state() + self._scale_action(action[5:]))

    def _do_action_array_single_agent(self, action: np.ndarray) -> None:
        """Apply action to the simulation."""
        if self.action_type == ActionType.POSITION:
            self.right_gripper.set_articulated_state(action)
        else:
            self.right_gripper.set_articulated_state(self.right_gripper.get_articulated_state() + self._scale_action(action))

    def _get_reward(self) -> float:
        """Calculate the reward for the Rope Threading task.

        Note:
            This task is very phase based with discrete steps that are to be followed in order to solve the task.
            Therefore, this task does not follow the preferred pattern of ``get_reward_features`` that can be easily extended
            and weighted.
        """

        reward = 0.0

        previously_lost_grasp = self.reward_info.get("lost_grasp", False)
        previous_fraction_rope_passed = self.reward_info.get("fraction_rope_passed", None)
        previous_distance_to_bimanual_grasp = self.reward_info.get("distance_to_bimanual_grasp", None)
        distance_to_lost_rope = self.reward_info.get("distance_to_lost_rope", None)
        rope_positions = self.rope.get_positions()

        self.reward_info = {
            "lost_grasp": False,
            "recovered_lost_grasp": False,
            "passed_eye": False,
            "lost_eye": 0,
            "distance_to_active_eye": None,
            "successful_task": False,
            "number_of_contacts": 0,
            "collisions_with_floor": 0,
            "delta_distance_to_eye": None,
            "distance_to_lost_rope": None,
            "delta_distance_to_lost_rope": None,
            "reward_from_distance_to_lost_rope": 0.0,
            "reward_from_delta_distance_to_lost_rope": 0.0,
            "reward_from_delta_distance": 0.0,
            "reward_from_absolute_distance": 0.0,
            "reward_from_losing_eyes": 0.0,
            "reward_from_losing_grasp": 0.0,
            "reward_from_collisions": 0.0,
            "reward_from_collisions_with_floor": 0,
            "reward_from_workspace_violations": 0.0,
            "reward_from_state_limit_violations": 0.0,
            "reward_from_passed_eyes": 0.0,
            "left_gripper_violated_workspace": False,
            "right_gripper_violated_workspace": False,
            "left_gripper_violated_state_limits": False,
            "right_gripper_violated_state_limits": False,
        }

        ######################################
        # Check if the episode is already done
        ######################################
        if self.eyes[-1].state == EyeStates.DONE:
            self.reward_info["successful_task"] = True
            return reward

        ############################################################
        # Check if the rope is still grasped by at least one gripper
        ############################################################
        rope_grasped = self.right_gripper.grasp_established
        if not self.single_agent:
            rope_grasped = rope_grasped or self.left_gripper.grasp_established
        if not rope_grasped:
            self.reward_info["lost_grasp"] = True
            reward += self.reward_amount_dict["lost_grasp"]
            self.reward_info["reward_from_losing_grasp"] = self.reward_amount_dict["lost_grasp"]

        self.reward_info["recovered_lost_grasp"] = previously_lost_grasp and not self.reward_info["lost_grasp"]

        ##########################################################
        # Check if the active loop is actually the valid next loop
        ##########################################################
        # This reward part is there to punish the following behaviors:
        # 1. pushing the rope through a loop to get rewards and then just going to the next without passing the rope through all of them in sequence
        # 2. letting go of the rope so that it slips out of the previous eyes
        # -> go back to the last eye with rope passin through

        if self.active_index == 0:
            # If the first eye was marked as in transition or done
            if self.eyes[0].get_state() in (EyeStates.TRANSITION, EyeStates.DONE):
                # and now has no rope points inside
                if not len(self.rope.get_indices_in_sphere(0)):
                    # mark it as lost
                    self.reward_info["lost_eye"] += 1
                    self.eyes[0].set_state(EyeStates.OPEN, self.color_eyes)

        if self.active_index > 0:
            # rope has n sphere ROIs for n eyes -> traverse back, until there is one, where rope indices are inside
            while not len(self.rope.get_indices_in_sphere(max(0, self.active_index - 1))):
                self.eyes[self.active_index].set_state(EyeStates.OPEN, self.color_eyes)
                self.active_index = max(0, self.active_index - 1)
                self.reward_info["lost_eye"] += 1
                if self.active_index == 0:
                    break

        if self.reward_info["lost_eye"] > 0:
            # If an eye was lost, recalculate the current distance to the active eye.
            self.distance_to_eye = np.linalg.norm(rope_positions[0] - self.eyes[self.active_index].center_pose[:3])

        # Subtract reward for every lost loop
        self.reward_info["reward_from_losing_eyes"] = self.reward_amount_dict["lost_eye"] * self.reward_info["lost_eye"]
        reward += self.reward_info["reward_from_losing_eyes"]

        # Set the active eye's state
        if self.eyes[self.active_index].get_state() == EyeStates.TRANSITION:
            # If the active eye is in transition, do not change the state. Otherwise passed_eye reward gets added in
            # bimanual_grasp part every step.
            pass
        else:
            self.eyes[self.active_index].set_state(EyeStates.NEXT, self.color_eyes)

        ########################
        # Distance based rewards
        ########################
        # Only give a distance reward, if the rope is not yet through the eye.
        if self.eyes[self.active_index].state == EyeStates.NEXT:
            ############################
            # Distance to the active eye
            ############################
            distance_to_active_eye = self.constrained_value(np.linalg.norm(rope_positions[0] - self.eyes[self.active_index].center_pose[:3]))
            self.reward_info["distance_to_active_eye"] = distance_to_active_eye
            self.reward_info["reward_from_absolute_distance"] = distance_to_active_eye * self.reward_amount_dict["distance_to_active_eye"] * self._distance_normalization_factor
            reward += self.reward_info["reward_from_absolute_distance"]

            ########################################
            # Moving the rope towards or away from the active eye
            ########################################
            delta_distance_to_eye = distance_to_active_eye - self.distance_to_eye
            self.reward_info["delta_distance_to_eye"] = delta_distance_to_eye
            self.distance_to_eye = distance_to_active_eye

            if delta_distance_to_eye < 0:
                self.reward_info["reward_from_delta_distance"] = self.reward_amount_dict["moved_towards_eye"] * abs(delta_distance_to_eye) * self._distance_normalization_factor
            elif delta_distance_to_eye > 0:
                self.reward_info["reward_from_delta_distance"] = self.reward_amount_dict["moved_away_from_eye"] * abs(delta_distance_to_eye) * self._distance_normalization_factor

            reward += self.reward_info["reward_from_delta_distance"]

        # Punish distance from the rope, if it is currently not grasped.
        # The distance is calculated to the index of the rope point set at the start of and episode
        # for starting grasped.
        if self.reward_info["lost_grasp"]:
            # Index on the rope used for starting grasped (estimate of a good position to grasp again)
            desired_grasp_position = rope_positions[self.right_gripper.grasp_index_pair[1]]
            if self.single_agent:
                gripper_position = self.right_gripper.get_grasp_center_position()
            else:
                # If both grippers are controlled, figure out which of the grippers should be used for grasping the rope
                right_gripper_position = self.right_gripper.get_grasp_center_position()
                left_gripper_position = self.left_gripper.get_grasp_center_position()
                # Check which of the ones is on the "right" side of the eye. If none or both are, use the right one.
                if self.eyes[self.active_index].points_are_right_of_eye(left_gripper_position) and not self.eyes[self.active_index].points_are_right_of_eye(right_gripper_position):
                    gripper_position = left_gripper_position
                else:
                    gripper_position = right_gripper_position
            self.reward_info["distance_to_lost_rope"] = self.constrained_value(np.linalg.norm(gripper_position - desired_grasp_position))
            self.reward_info["reward_from_distance_to_lost_rope"] = self.reward_info["distance_to_lost_rope"] * self.reward_amount_dict["distance_to_lost_rope"] * self._distance_normalization_factor
            reward += self.reward_info["reward_from_distance_to_lost_rope"]

            # Check if there already is a distance to the rope from the last step
            if distance_to_lost_rope is not None:
                self.reward_info["delta_distance_to_lost_rope"] = self.reward_info["distance_to_lost_rope"] - distance_to_lost_rope
                self.reward_info["reward_from_delta_distance_to_lost_rope"] = self.reward_info["delta_distance_to_lost_rope"] * self.reward_amount_dict["delta_distance_to_lost_rope"] * self._distance_normalization_factor
                reward += self.reward_info["reward_from_delta_distance_to_lost_rope"]

        ####################################################
        # Check if the rope is passed through the active eye
        ####################################################
        if len(indices_in_eye := self.rope.get_indices_in_sphere(self.active_index)):
            # The rope is correctly in the loop, if
            # 1. there are rope indices in the loop
            # 2. the smaller indices of the rope are left of the eye
            # 3. the larger indices of the rope are right of the eye
            start_and_end_sign = self.eyes[self.active_index].points_are_right_of_eye(rope_positions[indices_in_eye[[0, -1]]])
            rope_is_on_both_sides = start_and_end_sign[0] < 0 and start_and_end_sign[1] > 0

            if self.fraction_of_rope_to_pass is not None:
                fraction_passed = (indices_in_eye[0] + 1) / self.rope.num_points
                # If the rope is passed through the eye, remove the distance based rewards from the reward
                # because the distance will get bigger, when passing the rope further through the eye
                if rope_is_on_both_sides:
                    reward -= self.reward_info["reward_from_delta_distance"]
                    reward -= self.reward_info["reward_from_absolute_distance"]
                    self.reward_info["reward_from_absolute_distance"] = 0.0
                    self.reward_info["reward_from_delta_distance"] = 0.0
                passed_through = rope_is_on_both_sides and (fraction_passed >= self.fraction_of_rope_to_pass)

                self.reward_info["fraction_rope_passed"] = fraction_passed * rope_is_on_both_sides
                self.reward_info["reward_from_fraction_rope_passed"] = fraction_passed * self.reward_amount_dict["fraction_rope_passed"]
                reward += self.reward_info["reward_from_fraction_rope_passed"]

                if previous_fraction_rope_passed is not None:
                    self.reward_info["delta_fraction_rope_passed"] = fraction_passed * rope_is_on_both_sides - previous_fraction_rope_passed
                    self.reward_info["reward_from_delta_fraction_rope_passed"] = self.reward_info["delta_fraction_rope_passed"] * self.reward_amount_dict["delta_fraction_rope_passed"]
                    reward += self.reward_info["reward_from_delta_fraction_rope_passed"]
            else:
                passed_through = rope_is_on_both_sides

        else:
            passed_through = False

        ########################
        # Switch to the next eye
        ########################
        if (not self.single_agent) and passed_through and self.reward_amount_dict["bimanual_grasp"]:
            # If there is reward for bimanual_grasp, only switch over
            # if the grippers have grasped the rope on opposite sides of the eye.

            # Only give the passed_eye reward once per loop
            if self.eyes[self.active_index].state == EyeStates.NEXT:
                self.reward_info["reward_from_passed_eyes"] = self.reward_amount_dict["passed_eye"]
                reward += self.reward_info["reward_from_passed_eyes"]
                self.reward_info["passed_eye"] = True
                # Swith the state to TRANSITION
                self.eyes[self.active_index].set_state(EyeStates.TRANSITION, self.color_eyes)

            # Give rewards to the second gripper for moving closer to the rope tip -> incentive to grasp
            if self.left_gripper.grasp_established:
                gripper_position = self.right_gripper.get_physical_pose()[:3]
            else:
                gripper_position = self.left_gripper.get_physical_pose()[:3]

            self.reward_info["distance_to_bimanual_grasp"] = self.constrained_value(np.linalg.norm(gripper_position - rope_positions[0]))
            self.reward_info["reward_from_distance_to_bimanual_grasp"] = self.reward_info["distance_to_bimanual_grasp"] * self.reward_amount_dict["distance_to_bimanual_grasp"] * self._distance_normalization_factor
            reward += self.reward_info["reward_from_distance_to_bimanual_grasp"]

            if previous_distance_to_bimanual_grasp is not None:
                self.reward_info["delta_distance_to_bimanual_grasp"] = self.reward_info["distance_to_bimanual_grasp"] - previous_distance_to_bimanual_grasp
                self.reward_info["reward_from_delta_distance_to_bimanual_grasp"] = self.reward_info["delta_distance_to_bimanual_grasp"] * self.reward_amount_dict["delta_distance_to_bimanual_grasp"] * self._distance_normalization_factor
                reward += self.reward_info["reward_from_delta_distance_to_bimanual_grasp"]

            both_grasped = self.left_gripper.grasp_established and self.right_gripper.grasp_established
            grippers_on_opposite_sides = self.eyes[self.active_index].points_are_right_of_eye(self.left_gripper.get_physical_pose()[:3]) + self.eyes[self.active_index].points_are_right_of_eye(self.right_gripper.get_physical_pose()[:3]) == 0

            if both_grasped and grippers_on_opposite_sides:
                self.reward_info["reward_from_bimanual_grasp"] = self.reward_amount_dict["bimanual_grasp"]
                reward += self.reward_info["reward_from_bimanual_grasp"]
                self.reward_info["bimanual_grasp"] = True
                self.eyes[self.active_index].set_state(EyeStates.DONE, self.color_eyes)
                self.active_index += 1
                # When switching to the next eye, update the distance to the active eye with the next eye.
                # Otherwise there would be a large negative delta distance in the next step.
                if self.active_index < len(self.eyes):
                    self.distance_to_eye = self.constrained_value(np.linalg.norm(rope_positions[0] - self.eyes[self.active_index].center_pose[:3]))

        elif passed_through:
            # If there is no reward for bimanual_grasp, switch the state as soon as the rope
            # is passed through the loop.
            self.reward_info["reward_from_passed_eyes"] = self.reward_amount_dict["passed_eye"]
            reward += self.reward_info["reward_from_passed_eyes"]
            self.reward_info["passed_eye"] = True
            self.eyes[self.active_index].set_state(EyeStates.DONE, self.color_eyes)
            self.active_index += 1
            # When switching to the next eye, update the distance to the active eye with the next eye.
            # Otherwise there would be a large negative delta distance in the next step.
            if self.active_index < len(self.eyes):
                self.distance_to_eye = self.constrained_value(np.linalg.norm(rope_positions[0] - self.eyes[self.active_index].center_pose[:3]))

        ######################
        # Collision punishment
        ######################
        number_of_contacts = 0
        for gripper_name in ("left_gripper", "right_gripper"):
            for contact_listener in self.contact_listeners[gripper_name]:
                number_of_contacts += contact_listener.getNumberOfContacts()

        if number_of_contacts > 0:
            self.reward_info["number_of_contacts"] = number_of_contacts
            self.reward_info["reward_from_collisions"] = self.reward_amount_dict["collision"] * number_of_contacts
            reward += self.reward_info["reward_from_collisions"]

        #######################
        # Collisions with floor
        #######################
        right_collision_model_positions = self.right_gripper.get_collision_object_positions()
        floor_collisions = len(right_collision_model_positions[right_collision_model_positions[:, 2] <= 0])
        if not self.single_agent:
            left_collision_model_positions = self.left_gripper.get_collision_object_positions()
            floor_collisions += len(left_collision_model_positions[left_collision_model_positions[:, 2] <= 0])

        if floor_collisions > 0:
            self.reward_info["collisions_with_floor"] = floor_collisions
            self.reward_info["reward_from_collisions_with_floor"] = self.reward_amount_dict["floor_collision"] * floor_collisions
            reward += self.reward_info["reward_from_collisions_with_floor"]

        ############################
        # Workspace and state limits
        ############################
        self.reward_info["right_gripper_violated_workspace"] = self.right_gripper.last_set_state_violated_workspace_limits
        self.reward_info["right_gripper_violated_state_limits"] = self.right_gripper.last_set_state_violated_state_limits

        if self.single_agent:
            self.reward_info["left_gripper_violated_workspace"] = False
            self.reward_info["left_gripper_violated_state_limits"] = False
        else:
            self.reward_info["left_gripper_violated_workspace"] = self.left_gripper.last_set_state_violated_workspace_limits
            self.reward_info["left_gripper_violated_state_limits"] = self.left_gripper.last_set_state_violated_state_limits
        self.reward_info["reward_from_state_limit_violations"] = self.reward_amount_dict["state_limit_violation"] * (self.reward_info["left_gripper_violated_state_limits"] + self.reward_info["right_gripper_violated_state_limits"])
        self.reward_info["reward_from_workspace_violations"] = self.reward_amount_dict["workspace_violation"] * (self.reward_info["left_gripper_violated_workspace"] + self.reward_info["right_gripper_violated_workspace"])

        reward += self.reward_info["reward_from_state_limit_violations"]
        reward += self.reward_info["reward_from_workspace_violations"]

        ################
        # Win condition
        ################
        if self.eyes[-1].state == EyeStates.DONE:
            self.reward_info["successful_task"] = True
            reward += self.reward_amount_dict["successful_task"]

        # Paranoid nan and inf check
        if np.isnan(reward) or np.isinf(reward):
            print(f"[WARNING] Encounterd invalid reward value {reward} with reward info {self.reward_info}")
            reward = 0.0

        self.reward_info["reward"] = reward

        return reward

    def _get_done(self) -> bool:
        """Look up if the episode is finished."""
        return self.reward_info["successful_task"]

    def _get_observation(self, maybe_rgb_observation: Union[np.ndarray, None]) -> Union[np.ndarray, dict]:
        """Assembles the correct observation based on the ``ObservationType``."""

        if self.observation_type == ObservationType.RGB:
            observation = maybe_rgb_observation

        elif self.observation_type == ObservationType.RGBD:
            observation = self.observation_space.sample()
            observation[:, :, :3] = maybe_rgb_observation
            observation[:, :, 3:] = self.get_depth()

        else:
            state_dict = {}
            if not self.single_agent:
                state_dict["left_has_grasped"] = np.array(float(self.left_gripper.grasp_established))[None]  # make it a 1D array that can be concatenated
                state_dict["left_ptsda_state"] = self.left_gripper.get_articulated_state()
                state_dict["left_gripper_pose"] = self.left_gripper.get_physical_pose()

            state_dict["right_has_grasped"] = np.array(float(self.right_gripper.grasp_established))[None]  # make it a 1D array that can be concatenated
            state_dict["right_ptsda_state"] = self.right_gripper.get_articulated_state()
            state_dict["right_gripper_pose"] = self.right_gripper.get_physical_pose()

            rope_positions = self.rope.get_positions()
            state_dict["rope_tip_position"] = rope_positions[0].ravel()
            if self.rope_tracking_point_indices is not None:
                state_dict["rope_tracking_point_positions"] = rope_positions[self.rope_tracking_point_indices].ravel()

            state_dict["active_eye_pose"] = np.zeros(4)
            state_dict["active_eye_pose"][:3] = self.eyes[self.active_index].center_pose[:3]
            state_dict["active_eye_pose"][-1] = self.eyes[self.active_index].rotation
            observation = np.concatenate(tuple(state_dict.values()), dtype=self.observation_space.dtype)
            observation = np.where(np.isnan(observation), 1.0 / self._distance_normalization_factor, observation)
            # Clip observation values to 10 times the workspace size.
            observation = np.clip(observation, -10.0 / self._distance_normalization_factor, 10.0 / self._distance_normalization_factor)

        return observation

    def _get_info(self) -> dict:
        """Assemble the info dictionary."""

        self.info = {}
        self.episode_info["lost_grasps"] += self.reward_info["lost_grasp"]
        self.episode_info["recovered_lost_grasps"] += self.reward_info["recovered_lost_grasp"]
        self.episode_info["passed_eyes"] += self.reward_info["passed_eye"]
        self.episode_info["lost_eyes"] += self.reward_info["lost_eye"]
        self.episode_info["collisions"] += self.reward_info["number_of_contacts"]
        self.episode_info["floor_collisions"] += self.reward_info["collisions_with_floor"]
        self.episode_info["rew_passed_eyes"] += self.reward_info["reward_from_passed_eyes"]
        self.episode_info["rew_delta_distance"] += self.reward_info["reward_from_delta_distance"]
        self.episode_info["rew_absolute_distance"] += self.reward_info["reward_from_absolute_distance"]
        self.episode_info["rew_losing_eyes"] += self.reward_info["reward_from_losing_eyes"]
        self.episode_info["rew_losing_grasp"] += self.reward_info["reward_from_losing_grasp"]
        self.episode_info["rew_collisions"] += self.reward_info["reward_from_collisions"]
        self.episode_info["rew_floor_collisions"] += self.reward_info["reward_from_collisions_with_floor"]
        self.episode_info["rew_workspace_violation"] += self.reward_info["reward_from_workspace_violations"]
        self.episode_info["rew_state_limit_violation"] += self.reward_info["reward_from_state_limit_violations"]
        self.episode_info["rew_dist_to_lost_rope"] += self.reward_info["reward_from_distance_to_lost_rope"]
        self.episode_info["rew_delt_dist_to_lost_rope"] += self.reward_info["reward_from_delta_distance_to_lost_rope"]

        self.episode_info["rew_bimanual_grasp"] += self.reward_info.get("reward_from_bimanual_grasp", 0.0)
        self.episode_info["rew_dist_to_bimanual_grasp"] += self.reward_info.get("reward_from_distance_to_bimanual_grasp", 0.0)
        self.episode_info["rew_delt_dist_to_bimanual_grasp"] += self.reward_info.get("reward_from_delta_distance_to_bimanual_grasp", 0.0)

        self.episode_info["rew_fraction_passed"] += self.reward_info.get("reward_from_fraction_rope_passed", 0.0)
        self.episode_info["rew_delta_fraction_passed"] += self.reward_info.get("reward_from_delta_fraction_rope_passed", 0.0)

        return {**self.info, **self.reward_info, **self.episode_info}

    def reset(
        self,
        seed: Union[int, np.random.SeedSequence, None] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Union[np.ndarray, None], Dict]:
        super().reset(seed)

        # Seed the instruments
        if self.unconsumed_seed:
            seeds = self.seed_sequence.spawn(len(self.eyes) + 2)
            for index, eye in enumerate(self.eyes):
                eye.seed(seeds[index])
            self.right_gripper.seed(seed=seeds[-1])
            if not self.single_agent:
                self.left_gripper.seed(seed=seeds[-2])
            self.unconsumed_seed = False

        # Reset scene objects
        def reset_gripper(gripper_key: str, gripper: ArticulatedGripper):
            if options and gripper_key in options:
                gripper_reset_options = {k: options[gripper_key][k] if k in options[gripper_key] else None for k in ["rcm_pose", "state", "angle"]}
                gripper.reset_gripper(**gripper_reset_options)
            else:
                gripper.reset_gripper()

        reset_gripper("right_gripper", self.right_gripper)
        if not self.single_agent:
            reset_gripper("left_gripper", self.left_gripper)

        # TODO: This does not work (does nothing -> SOFA bug?)
        self.rope.set_state(self.initial_rope_state)

        # Reset the eye states and optionally colors
        new_waypoint_positions = []
        if options and "eye_xyzphi" in options:
            if not len(options["eye_xyzphi"]) == len(self.eyes):
                raise ValueError(f"The number of eye positions must match the number of eyes. Got {len(options['eye_xyzphi'])} positions for {len(self.eyes)} eyes.")

            for index, eye in enumerate(self.eyes):
                eye.set_state(EyeStates.OPEN, self.color_eyes)
                eye.set_xyzphi(options["eye_xyzphi"][index])
                new_waypoint_positions.append(eye.center_pose[:3])
        elif options and "eye_center_xyzphi" in options:
            if not len(options["eye_center_xyzphi"]) == len(self.eyes):
                raise ValueError(f"The number of eye positions must match the number of eyes. Got {len(options['eye_center_xyzphi'])} positions for {len(self.eyes)} eyes.")

            for index, eye in enumerate(self.eyes):
                eye.set_state(EyeStates.OPEN, self.color_eyes)
                eye.set_center_xyzphi(options["eye_center_xyzphi"][index])
                new_waypoint_positions.append(eye.center_pose[:3])
        else:
            for eye in self.eyes:
                eye.set_state(EyeStates.OPEN, self.color_eyes)
                eye.reset()
                new_waypoint_positions.append(eye.center_pose[:3])

        # Update the waypoints in the rope
        self.rope.sphere_roi_center = new_waypoint_positions

        self.active_index = 0
        self.eyes[0].set_state(EyeStates.NEXT, self.color_eyes)

        # Clear the episode info values
        for key in self.episode_info:
            self.episode_info[key] = 0.0

        # and the reward info dict
        self.reward_info = {}

        # Execute any callbacks passed to the env.
        for callback in self.on_reset_callbacks:
            callback(self)

        # Do not release grasped rope during settle_steps
        if self.right_gripper.start_grasped:
            original_grasp_angle = self.right_gripper.angle_to_grasp_threshold
            original_release_angle = self.right_gripper.angle_to_release_threshold
            self.right_gripper.angle_to_grasp_threshold = np.inf
            self.right_gripper.angle_to_release_threshold = np.inf

        # Animate several timesteps without actions until simulation settles
        for _ in range(self._settle_steps):
            self.sofa_simulation.animate(self._sofa_root_node, self._sofa_root_node.getDt())

        # Restore the original thresholds
        if self.right_gripper.start_grasped:
            self.right_gripper.angle_to_grasp_threshold = original_grasp_angle
            self.right_gripper.angle_to_release_threshold = original_release_angle

        # Initial distance between rope and the active eye
        self.distance_to_eye = self.constrained_value(np.linalg.norm(self.rope.get_positions()[0] - self.eyes[self.active_index].center_pose[:3]))

        return self._get_observation(self._maybe_update_rgb_buffer()), {}


if __name__ == "__main__":
    import pprint
    import time

    pp = pprint.PrettyPrinter()

    eye_config = [
        (60, 10, 0, 90),
        # (10, 10, 0, 90),
        # (10, 60, 0, -45),
        # (60, 60, 0, 90),
    ]

    env = RopeThreadingEnv(
        observation_type=ObservationType.STATE,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.VELOCITY,
        image_shape=(800, 800),
        frame_skip=10,
        time_step=0.01,
        settle_steps=20,
        reward_amount_dict={
            "passed_eye": 10.0,
            "lost_eye": -20.0,  # more than passed_eye
            "successful_task": 100.0,
            "lost_grasp": -0.1,
            "collision": -0.1,
            "floor_collision": -0.1,
            "moved_towards_eye": 200.0,
            "moved_away_from_eye": -200.0,
            "workspace_violation": -0.01,
            "state_limit_violation": -0.01,
            "bimanual_grasp": 100.0,
            "distance_to_bimanual_grasp": -0.0,
            "delta_distance_to_bimanual_grasp": -200.0,
        },
        create_scene_kwargs={
            "eye_config": eye_config,
            "eye_reset_noise": {
                "low": np.array([-20.0, -20.0, 0.0, -15]),
                "high": np.array([20.0, 20.0, 0.0, 15]),
            },
            # "eye_reset_noise": None,
            "randomize_gripper": False,
            "randomize_grasp_index": False,
            "start_grasped": True,
        },
        fraction_of_rope_to_pass=0.05,
        only_right_gripper=False,
        individual_agents=True,
        num_rope_tracking_points=10,
    )

    env.reset()
    done = False

    options = {"eye_xyzphi": [(60, 10, 0, 30)]}

    fps_list = deque(maxlen=100)
    counter = 0
    while not done:
        start = time.perf_counter()
        action = env.action_space.sample()
        for key, val in action.items():
            action[key] = 3.0
        previous_state = env.right_gripper.get_articulated_state().copy()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        current_state = env.right_gripper.get_articulated_state().copy()
        print(f"State Velocity: {(current_state - previous_state)/(env.time_step*env.frame_skip)}")
        end = time.perf_counter()
        fps = 1 / (end - start)
        fps_list.append(fps)

        print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")

        counter += 1
        if counter % 50 == 0:
            couter = 0
            env.reset(options=options)
