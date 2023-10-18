import gymnasium.spaces as spaces
import numpy as np

from collections import defaultdict, deque
from enum import Enum, unique
from pathlib import Path
from functools import reduce

from typing import Callable, Union, Tuple, Optional, List, Any, Dict

from sofa_env.base import SofaEnv, RenderMode, RenderFramework

from sofa_env.scenes.pick_and_place.scene_description import Peg
from sofa_env.scenes.rope_threading.sofa_objects.gripper import ArticulatedGripper
from sofa_env.sofa_templates.camera import Camera
from sofa_env.sofa_templates.rope import Rope

HERE = Path(__file__).resolve().parent
SCENE_DESCRIPTION_FILE_PATH = HERE / "scene_description.py"


@unique
class ObservationType(Enum):
    """Observation type specifies whether the environment step returns RGB images or a defined state"""

    RGB = 0
    STATE = 1
    RGBD = 2


@unique
class ActionType(Enum):
    """Action type specifies whether the actions are continuous values that are added to the gripper's state, or discrete actions that increment the gripper's state."""

    DISCRETE = 0
    CONTINUOUS = 1


@unique
class Phase(Enum):
    PICK = 0
    PLACE = 1
    ANY = 2


class PickAndPlaceEnv(SofaEnv):
    """Pick and Place Environment

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
        maximum_state_velocity (Union[np.ndarray, float]): Velocity in deg/s for pts and mm/s for d in state space which are applied with a normalized action of value 1.
        discrete_action_magnitude (Union[np.ndarray, float]): Discrete change in state space in deg/s for pts and mm/s for d.
        create_scene_kwargs (Optional[dict]): A dictionary to pass additional keyword arguments to the ``createScene`` function.
        on_reset_callbacks (Optional[List[Callable]]): Functions that are called as the last step of the ``env.reset()`` function.
        action_type (ActionType): Discrete or continuous actions to define the action space of the environment.
        num_active_pegs (int): Number of pegs that are active for placing the torus. The pegs are selected randomly.
        randomize_color (bool): Whether to randomize the color of the torus and pegs.
        num_torus_tracking_points (int): Number of points on the torus to add to state observations.
        start_grasped (bool): Whether the torus is grasped at the start of the episode.
        randomize_torus_position (bool): Whether to randomize the position of the torus at the start of the episode.
        only_learn_pick (bool): Whether to only learn the pick phase of the task.
        minimum_lift_height (float): Minimum height above the peg that the torus must be lifted to in the pick phase.
        block_done_when_torus_unstable (bool): Whether to block the done signal when the torus model is unstable.
    """

    def __init__(
        self,
        scene_path: Union[str, Path] = SCENE_DESCRIPTION_FILE_PATH,
        image_shape: Tuple[int, int] = (124, 124),
        observation_type: ObservationType = ObservationType.RGB,
        time_step: float = 0.01,
        frame_skip: int = 3,
        settle_steps: int = 20,
        render_mode: RenderMode = RenderMode.HEADLESS,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        reward_amount_dict={
            Phase.ANY: {
                "lost_grasp": -10.0,
                "grasped_torus": 0.0,
                "gripper_jaw_peg_collisions": -0.0,
                "gripper_jaw_floor_collisions": -0.0,
                "unstable_deformation": -0.0,
                "torus_velocity": -0.0,
                "gripper_velocity": -0.0,
                "torus_dropped_off_board": -0.0,
                "action_violated_state_limits": -0.0,
                "action_violated_cartesian_workspace": -0.0,
                "successful_task": 50.0,
            },
            Phase.PICK: {
                "established_grasp": 10.0,
                "gripper_distance_to_torus_center": -0.0,
                "delta_gripper_distance_to_torus_center": -0.0,
                "gripper_distance_to_torus_tracking_points": -0.0,
                "delta_gripper_distance_to_torus_tracking_points": -0.0,
                "distance_to_minimum_pick_height": -0.0,
                "delta_distance_to_minimum_pick_height": -0.0,
            },
            Phase.PLACE: {
                "torus_distance_to_active_pegs": -0.0,
                "delta_torus_distance_to_active_pegs": -1.0,
            },
        },
        maximum_state_velocity: Union[np.ndarray, float] = np.array([20.0, 20.0, 20.0, 20.0, 5.0]),
        discrete_action_magnitude: Union[np.ndarray, float] = np.array([15.0, 15.0, 15.0, 10.0, 2.0]),
        create_scene_kwargs: Optional[dict] = None,
        on_reset_callbacks: Optional[List[Callable]] = None,
        action_type: ActionType = ActionType.CONTINUOUS,
        num_active_pegs: int = 1,
        randomize_color: bool = False,
        num_torus_tracking_points: int = -1,
        start_grasped: bool = False,
        randomize_torus_position: bool = False,
        only_learn_pick: bool = False,
        minimum_lift_height: float = 30.0,
        block_done_when_torus_unstable: bool = False,
    ) -> None:
        # Pass image shape to the scene creation function
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape
        create_scene_kwargs["start_grasped"] = start_grasped

        if randomize_torus_position and start_grasped:
            raise ValueError("If you want to skip the pick phase, and just place, the torus position should match the initial gripper position, thus randomize_torus_position and start_grasped should not both be True at the same time.\n{start_grasped=}\n{randomize_torus_position=}")

        # Whether to colorize the torus and pegs
        self.randomize_color = randomize_color

        # How high to lift the torus' center of mass to switch from Pick to Place phase
        self.minimum_lift_height = minimum_lift_height
        self.only_learn_pick = only_learn_pick
        self.start_grasped = start_grasped

        # Check if the position of the torus should be randomized on reset
        self.randomize_torus_position = randomize_torus_position

        if only_learn_pick and start_grasped:
            raise Exception(f"Environment was asked to skip the pick phase ({start_grasped=}) and also only learn to pick ({only_learn_pick=}).")

        # Whether to block the done signal, if the torus is unstable while in the Place phase
        self.block_done_when_torus_unstable = block_done_when_torus_unstable

        self.colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

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

        ##############
        # Action Space
        ##############
        action_dimensionality = 5
        self.action_type = action_type
        if action_type == ActionType.CONTINUOUS:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dimensionality,), dtype=np.float32)
            self._maximum_state_velocity = maximum_state_velocity
            self._scale_action = self._scale_continuous_action
        else:
            self.action_space = spaces.Discrete(action_dimensionality * 2 + 1)
            self._scale_action = self._scale_discrete_action

            if isinstance(discrete_action_magnitude, np.ndarray):
                if not len(discrete_action_magnitude) == action_dimensionality:
                    raise ValueError("If you want to use individual discrete action step sizes per action dimension, please pass an array of length {action_dimensionality} as discrete_action_magnitude. Received {discrete_action_magnitude=} with lenght {len(discrete_action_magnitude)}.")

            # [step, 0, 0, ...], [-step, 0, 0, ...], [0, step, 0, ...], [0, -step, 0, ...]
            action_list = []
            for i in range(action_dimensionality * 2):
                action = [0.0] * action_dimensionality
                step_size = discrete_action_magnitude if isinstance(discrete_action_magnitude, float) else discrete_action_magnitude[int(i / 2)]
                action[int(i / 2)] = (1 - 2 * (i % 2)) * step_size
                action_list.append(action)

            # Noop action
            action_list.append([0.0] * action_dimensionality)

            self._discrete_action_lookup = np.array(action_list)
            self._discrete_action_lookup *= self.time_step
            self._discrete_action_lookup.flags.writeable = False

        ###################
        # Observation Space
        ###################
        # State observations
        self.num_torus_tracking_points = num_torus_tracking_points
        self.torus_tracking_point_indices: Union[np.ndarray, None]  # set in self._init_sim

        if observation_type == ObservationType.STATE:
            # has_grasped -> 1
            # ptsda_state -> 5
            # gripper_pose -> 7
            # torus_center_of_mass_position -> 3
            # torus_tracking_point_positions -> num_torus_tracking_points * 3
            # active_peg_positions (XZ) -c num_active_pegs*2
            observations_size = 1 + 5 + 7 + 3 + num_torus_tracking_points * 3 + num_active_pegs * 2
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observations_size,), dtype=np.float32)

        # Image observations
        elif observation_type == ObservationType.RGB:
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (3,), dtype=np.uint8)

        elif observation_type == ObservationType.RGBD:
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (4,), dtype=np.uint8)

        else:
            raise ValueError(f"Please set observation_type to a value of ObservationType. Received {observation_type}.")

        self.observation_type = observation_type

        #########################
        # Episode specific values
        #########################

        # Where to place the torus
        if num_active_pegs < 1:
            raise ValueError(f"num_active_pegs must be greater than 0. Received {num_active_pegs}.")
        self.num_active_pegs = num_active_pegs
        self.active_target_positions = np.empty((num_active_pegs, 3))

        # Infos per episode
        self.episode_info = defaultdict(float)

        # Infos from the reward
        self.reward_info = {}
        self.reward_features = {}

        # All parameters of the reward function that are not passed default to 0.0
        self.reward_amount_dict = defaultdict(dict) | {phase: defaultdict(float, **reward_dict) for phase, reward_dict in reward_amount_dict.items()}

        # Callback functions called on reset
        self.on_reset_callbacks = on_reset_callbacks if on_reset_callbacks is not None else []

    def _init_sim(self) -> None:
        super()._init_sim()

        self.gripper: ArticulatedGripper = self.scene_creation_result["gripper"]
        self.torus: Rope = self.scene_creation_result["torus"]
        self.camera: Camera = self.scene_creation_result["camera"]
        self.pegs: List[Peg] = self.scene_creation_result["pegs"]
        self.target_positions: np.ndarray = self.scene_creation_result["target_positions"]
        self.contact_listeners = self.scene_creation_result["contact_listeners"]

        # Factor for normalizing the distances in the reward function.
        # Based on the workspace of the gripper.
        self._distance_normalization_factor = 1.0 / np.linalg.norm(self.gripper.cartesian_workspace["high"] - self.gripper.cartesian_workspace["low"])

        # Save the gripper position for velocity calculation
        self.previous_gripper_position = self.gripper.get_pose()[:3].copy()

        # Identify the indices of points on the torus that should be used for describing the state of the rope. If num_torus_tracking_points is -1, use all of them.
        if self.num_torus_tracking_points > 0:
            if self.torus.num_points < self.num_torus_tracking_points:
                raise ValueError(f"The number of torus tracking points ({self.num_torus_tracking_points}) is larger than the number of points on the torus ({self.torus.num_points}).")
            self.torus_tracking_point_indices = np.linspace(0, self.torus.num_points, num=self.num_torus_tracking_points, endpoint=False, dtype=np.int16)
        elif self.num_torus_tracking_points == -1:
            self.torus_tracking_point_indices = np.array(range(self.torus.num_points), dtype=np.int16)
        elif self.num_torus_tracking_points == 0:
            self.torus_tracking_point_indices = None
        else:
            raise ValueError(f"num_torus_tracking_points must be > 0 or == -1 (to use them all). Received {self.num_torus_tracking_points}.")

    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        """Step function of the environment that applies the action to the simulation and returns observation, reward, done signal, and info."""

        maybe_rgb_observation = super().step(action)

        observation = self._get_observation(maybe_rgb_observation=maybe_rgb_observation)
        reward = self._get_reward()
        terminated = self._get_done()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _scale_continuous_action(self, action: np.ndarray) -> np.ndarray:
        """
        Policy is output is clipped to [-1, 1] e.g. [-0.3, 0.8, 1].
        We want to scale that to the maximum velocities defined
        in ``maximum_state_velocity`` in [angle, angle, angle, mm, angle] / step.
        and further to per second (because delta T is not 1 second).
        """
        return self.time_step * self._maximum_state_velocity * action

    def _scale_discrete_action(self, action: int) -> np.ndarray:
        """Maps action indices to a motion delta."""
        return self._discrete_action_lookup[action]

    def _do_action(self, action: Union[int, np.ndarray]) -> None:
        """Apply action to the simulation."""

        self.gripper.set_articulated_state(self.gripper.get_articulated_state() + self._scale_action(action))

    def _get_reward_features(self, previous_reward_features) -> dict:
        """Get the features that may be used to assemble the reward function

        Features:
            - grasped_torus (bool): Whether the gripper is grasping the torus.
            - established_grasp (bool): Whether the gripper has established a new grasp on the torus in this step.
            - lost_grasp (bool): Whether the gripper has lost its grasp on the torus in this step.
            - torus_distance_to_active_pegs (np.ndarray): The distance to the active pegs (one distance per active peg).
            - delta_torus_distance_to_active_pegs (np.ndarray): The change in distance to the active pegs (one distance per active peg) (since last step).
            - gripper_distance_to_torus_center (float): The distance between the middle of the gripper jaw and the torus' center of mass.
            - delta_gripper_distance_to_torus_center (float): The change in distance between the middle of the gripper jaw and the torus' center of mass (since last step).
            - gripper_distance_to_torus_tracking_points (np.ndarray): The distance between the middle of the gripper jaw and the torus' tracking points (one distance per key frame).
            - delta_gripper_distance_to_torus_tracking_points (np.ndarray): The change in distance between the middle of the gripper jaw and the torus' tracking points (one distance per key frame) (since last step).
            - gripper_jaw_peg_collisions (int): Number of collisions between the gripper jaw collision models and the pegs.
            - gripper_jaw_floor_collisions (int): Number of collisions between the gripper jaw collision models and the floor (approximated by checking if any points are below y=0).
            - unstable_deformation (bool): An approximation of whether the torus deformation might be unstable. Checked by looking at the velocities of the torus.
            - torus_velocity (float): The mean Cartesian velocity of all points of the torus.
            - gripper_velocity (float): The Cartesian velocity of the gripper's jaw joint.
            - torus_dropped_off_board (bool): Whether the torus is not on the board any more.
            - action_violated_state_limits (bool): Whether the last action would have violated the state limits of the gripper.
            - action_violated_cartesian_workspace (bool): Whether the last action would have violated the Cartesian workspace limits of the gripper.
            - successful_task (bool): Whether the torus was placed on one of the active pegs.
            - distance_to_minimum_pick_height (float): The distance in height between the torus' center of mass and the ``minimum_lift_height``.
            - delta_distance_to_minimum_pick_height (float): The change in distance in height between the torus' center of mass and the ``minimum_lift_height``.
        """

        reward_features = {}

        torus_center_of_mass = self.torus.get_center_of_mass()
        torus_positions = self.torus.get_positions()
        gripper_grasp_center_position = self.gripper.get_grasp_center_position()

        # Is the torus grasped by the gripper?
        reward_features["grasped_torus"] = self.gripper.grasp_established

        # Was there a grasp established in this step?
        reward_features["established_grasp"] = self.gripper.grasp_established and not previous_reward_features["grasped_torus"]

        # Was there a grasp lost in this step?
        reward_features["lost_grasp"] = previous_reward_features["grasped_torus"] and not self.gripper.grasp_established

        # Distances between torus and the active pegs
        reward_features["torus_distance_to_active_pegs"] = np.linalg.norm(torus_center_of_mass - self.active_target_positions, axis=1)

        # Change in distances between torus and the active pegs
        reward_features["delta_torus_distance_to_active_pegs"] = reward_features["torus_distance_to_active_pegs"] - previous_reward_features["torus_distance_to_active_pegs"]

        # Distance between gripper and torus
        reward_features["gripper_distance_to_torus_center"] = np.linalg.norm(gripper_grasp_center_position - torus_center_of_mass)

        # Change in distance between gripper and torus
        reward_features["delta_gripper_distance_to_torus_center"] = reward_features["gripper_distance_to_torus_center"] - previous_reward_features["gripper_distance_to_torus_center"]

        if self.torus_tracking_point_indices is not None:
            # Distance between gripper and torus tracking points
            reward_features["gripper_distance_to_torus_tracking_points"] = np.linalg.norm(gripper_grasp_center_position - torus_positions[self.torus_tracking_point_indices], axis=1)
            # Change in distance between gripper and torus tracking points
            reward_features["delta_gripper_distance_to_torus_tracking_points"] = reward_features["gripper_distance_to_torus_tracking_points"] - previous_reward_features["gripper_distance_to_torus_tracking_points"]
        else:
            reward_features["gripper_distance_to_torus_tracking_points"] = 0.0
            reward_features["delta_gripper_distance_to_torus_tracking_points"] = 0.0

        # Are there collisions between the gripper's jaws and the pegs / the floor?
        reward_features["gripper_jaw_peg_collisions"] = sum([listener.getNumberOfContacts() for listener in self.contact_listeners])
        reward_features["gripper_jaw_floor_collisions"] = self.gripper.get_grasp_center_position()[1] < 0.0

        # Is the deformable model of the torus unstable?
        # The torus is probably unstable, if the mean velocity is larger than 150% of the gripper speed, and larger than 25 mm/s
        mean_torus_velocity = np.mean(np.linalg.norm(self.torus.mechanical_object.velocity.array()[:, :3], axis=1))
        gripper_position = self.gripper.get_pose()[:3]
        gripper_velocity = np.linalg.norm(self.previous_gripper_position - gripper_position) / (self.time_step * self.frame_skip)
        torus_probably_unstable = mean_torus_velocity > 1.5 * gripper_velocity and mean_torus_velocity > 25.0
        self.previous_gripper_position[:] = gripper_position
        reward_features["unstable_deformation"] = torus_probably_unstable

        # Cartesian speed of gripper and torus center
        reward_features["torus_velocity"] = mean_torus_velocity
        reward_features["gripper_velocity"] = gripper_velocity

        # Task is successful, if the torus is on an active peg
        # X and Z distance between torus center of mass and pegs smaller than the peg radius -> center of mass in peg in XZ plane
        # peg radius: 5.0 mm
        peg_in_torus_xz = np.any(np.linalg.norm(self.active_target_positions[:, [0, 2]] - torus_center_of_mass[[0, 2]], axis=1) < 7.0)
        # All points of the torus below half of the peg height
        # peg height: 20.0 mm and raius of torus: 3.5
        peg_in_torus_y = np.all(torus_positions[:, 1] < 10.0)

        if self.block_done_when_torus_unstable:
            reward_features["successful_task"] = peg_in_torus_xz and peg_in_torus_y and not torus_probably_unstable
        else:
            reward_features["successful_task"] = peg_in_torus_y and peg_in_torus_xz

        # Did the torus drop off the board?
        reward_features["torus_dropped_off_board"] = np.mean(torus_positions[:, 1]) < 3.0 and (np.abs(np.mean(torus_positions[:, 0])) > 80.0 or abs(np.mean(torus_positions[:, 2])) > 80.0)

        # Did the action violate state or workspace limits?
        reward_features["action_violated_cartesian_workspace"] = self.gripper.last_set_state_violated_workspace_limits
        reward_features["action_violated_state_limits"] = self.gripper.last_set_state_violated_state_limits

        # Is the torus' center of mass above the pick threshold?
        # Masked by whether the torus is actually grasped
        if reward_features["grasped_torus"]:
            reward_features["distance_to_minimum_pick_height"] = max(self.minimum_lift_height - torus_center_of_mass[1], 0.0)
            if "distance_to_minimum_pick_height" in previous_reward_features:
                reward_features["delta_distance_to_minimum_pick_height"] = reward_features["distance_to_minimum_pick_height"] - previous_reward_features["distance_to_minimum_pick_height"]

        return reward_features

    def _get_reward(self) -> float:
        """Retrieve the reward features and scale them with the ``reward_amount_dict``."""
        reward = 0.0
        self.reward_info = {}

        reward_features = self._get_reward_features(self.reward_features)

        # Figuring out the current phase of the task
        # If we come from the PICK phase, and lift the torus high enough, switch over to PLACE
        if self.active_phase == Phase.PICK:
            if reward_features["grasped_torus"] and max(self.minimum_lift_height - self.torus.get_center_of_mass()[1], 0.0) <= 0.0:
                self.active_phase = Phase.PLACE
        # If we come from the PLACE phase, and drop the torus, switch back to the PICK phase
        elif self.active_phase == Phase.PLACE and not reward_features["grasped_torus"]:
            self.active_phase = Phase.PICK

        # If the environment was set to end after the PICK task, check if that is already done
        if self.only_learn_pick and self.active_phase == Phase.PLACE:
            reward_features["successful_task"] = True

        # Dictionary of the active phase overwrites the values of the ANY phase (original dict is not changed)
        # right hand of | has higher priority -> if key is in both, right side will be used
        reward_amount_dict = self.reward_amount_dict[Phase.ANY] | self.reward_amount_dict[self.active_phase]

        # These values are arrays -> reduce them to a scalar
        if self.torus_tracking_point_indices is not None:
            reward_features["gripper_distance_to_torus_tracking_points"] = np.min(reward_features["gripper_distance_to_torus_tracking_points"])
            reward_features["delta_gripper_distance_to_torus_tracking_points"] = np.min(reward_features["delta_gripper_distance_to_torus_tracking_points"])
        if self.num_active_pegs > 1:
            reward_features["torus_distance_to_active_pegs"] = np.min(reward_features["torus_distance_to_active_pegs"])
            reward_features["delta_torus_distance_to_active_pegs"] = np.min(reward_features["delta_torus_distance_to_active_pegs"])

        for key, value in reward_features.items():
            # Normalize distance and velocity features with the size of the workspace
            # and clip them to the size of the workspace (to catch invalid values that come from an unstable simulation)
            if "distance" in key or "velocity" in key:
                if np.isnan(value):
                    value = 1.0 / self._distance_normalization_factor
                value = np.clip(value, -1.0 / self._distance_normalization_factor, 1.0 / self._distance_normalization_factor)
                value = self._distance_normalization_factor * value

            # Aggregate the features with their specific factors
            self.reward_info[f"reward_{key}"] = reward_amount_dict[key] * value

            # Add them to the reward
            reward += self.reward_info[f"reward_{key}"]

        self.reward_info["reward"] = reward

        # we change the values of the dict -> do a copy (deepcopy not necessary, because the value itself is not manipulated)
        self.reward_features = reward_features.copy()

        return float(reward)

    def _get_done(self) -> bool:
        """Look up if the episode is finished."""
        return self.reward_features["successful_task"]

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
            state_dict["has_grasped"] = np.array(float(self.gripper.grasp_established))[None]  # make it a 1D array that can be concatenated
            state_dict["ptsda_state"] = self.gripper.get_articulated_state()
            state_dict["gripper_pose"] = self.gripper.get_pose()
            state_dict["torus_center_of_mass_position"] = self.torus.get_center_of_mass()
            if self.torus_tracking_point_indices is not None:
                state_dict["torus_tracking_point_positions"] = self.torus.get_positions()[self.torus_tracking_point_indices].ravel()
            state_dict["active_peg_positions"] = self.active_target_positions[:, [0, 2]].ravel()
            observation = np.concatenate(tuple(state_dict.values()))
            # Overwrite possible NaNs with the maximum distance in the workspace
            observation = np.where(np.isnan(observation), 1.0 / self._distance_normalization_factor, observation)

        return observation

    def _get_info(self) -> dict:
        """Assemble the info dictionary."""

        self.info = {"phase": self.active_phase.value}

        # These keys might not be in the reward features (masked out).
        # Add them here manually, to satisfy the logger that expects them to be there.
        if "reward_distance_to_minimum_pick_height" not in self.reward_info:
            self.reward_info["reward_distance_to_minimum_pick_height"] = 0.0
        if "reward_delta_distance_to_minimum_pick_height" not in self.reward_info:
            self.reward_info["reward_delta_distance_to_minimum_pick_height"] = 0.0
        if "distance_to_minimum_pick_height" not in self.reward_features:
            self.reward_features["distance_to_minimum_pick_height"] = self.minimum_lift_height - self.torus.get_center_of_mass()[1]

        for key, value in self.reward_info.items():
            # shortens 'reward_delta_gripper_distance_to_torus_tracking_points'
            # to 'ret_del_gri_dis_to_tor_tra_poi'
            words = key.split("_")[1:]
            shortened_key = reduce(lambda x, y: x + "_" + y[:3], words, "ret")
            self.episode_info[shortened_key] += value

        return {**self.info, **self.reward_info, **self.episode_info, **self.reward_features}

    def reset(self, seed: Union[int, np.random.SeedSequence, None] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Union[np.ndarray, None], Dict]:
        """Reset the state of the environment and return the initial observation."""

        if self._initialized:
            # If the gripper does not start already having grasped the torus, we can randomize the torus' starting position.
            # This has to happen before resetting the SOFA simulation, because setting the state of the MO in Rope
            # currently does not work -> we have to change the reset position -> SOFA will place the torus on reset.
            if self.randomize_torus_position:
                # Get a new position randomly from the board
                new_center_of_mass = self.rng.uniform([-80, 40, -80], [80, 40, 80])
                # Translation to the new position from the old one
                translation_offset = new_center_of_mass - self.torus.get_reset_center_of_mass()
                new_states = self.torus.get_reset_state().copy()
                new_states[:, :3] += translation_offset
                self.torus.set_reset_state(new_states)

        # Reset from parent class -> calls the simulation's reset function
        super().reset(seed)

        # Seed the instruments
        if self.unconsumed_seed:
            seeds = self.seed_sequence.spawn(1)
            self.gripper.seed(seed=seeds[0])
            self.unconsumed_seed = False

        # Reset the gripper
        self.gripper.reset_gripper()
        self.previous_gripper_position[:] = self.gripper.get_pose()[:3]

        # Chose new active pegs
        active_indices = self.rng.choice(len(self.pegs), size=self.num_active_pegs, replace=False)
        self.active_target_positions[:] = self.target_positions[active_indices]
        self.torus_distance_to_active_pegs = np.linalg.norm(self.torus.get_center_of_mass() - self.active_target_positions, axis=1)

        # Randomize colors if required
        if self.randomize_color:
            active_color_index, inactive_color_index = self.rng.choice(len(self.colors) - 1, size=2, replace=False)
            active_color = self.colors[active_color_index]
            inactive_color = self.colors[inactive_color_index]
        else:
            active_color = (255, 0, 0)
            inactive_color = (0, 0, 255)

        # Set colors of the pegs and torus
        for peg in self.pegs:
            peg.set_color(inactive_color)
        for index in active_indices:
            self.pegs[index].set_color(active_color)
        self.torus.set_color(active_color)

        # Clear the episode info values
        for key in self.episode_info:
            self.episode_info[key] = 0.0

        # and the reward info dict
        self.reward_info = {}

        # and fill the first values used as previous_reward_features
        self.reward_features = {}
        self.reward_features["grasped_torus"] = self.gripper.grasp_established
        self.reward_features["torus_distance_to_active_pegs"] = np.linalg.norm(self.torus.get_center_of_mass() - self.active_target_positions, axis=1)
        self.reward_features["gripper_distance_to_torus_center"] = np.linalg.norm(self.gripper.get_grasp_center_position() - self.torus.get_center_of_mass())
        if self.torus_tracking_point_indices is not None:
            self.reward_features["gripper_distance_to_torus_tracking_points"] = np.linalg.norm(self.gripper.get_grasp_center_position() - self.torus.get_positions()[self.torus_tracking_point_indices], axis=1)

        # Execute any callbacks passed to the env.
        for callback in self.on_reset_callbacks:
            callback(self)

        # Animate several timesteps without actions until simulation settles
        for _ in range(self._settle_steps):
            self.sofa_simulation.animate(self._sofa_root_node, self._sofa_root_node.getDt())

        # If the torus is immediately unstable after reset, do reset again
        # The torus is probably unstable, if the mean velocity is larger than 25 mm/s
        mean_torus_velocity = np.mean(np.linalg.norm(self.torus.mechanical_object.velocity.array()[:, :3], axis=1))
        torus_probably_unstable = mean_torus_velocity > 25.0
        if torus_probably_unstable:
            print("Reset again, because simulation was unstable!")
            self.reset()

        # Set the current phase after reset
        self.active_phase = Phase.PLACE if self.start_grasped else Phase.PICK

        return self._get_observation(maybe_rgb_observation=self._maybe_update_rgb_buffer()), {}


if __name__ == "__main__":
    import pprint
    import time

    pp = pprint.PrettyPrinter()

    env = PickAndPlaceEnv(
        observation_type=ObservationType.STATE,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.CONTINUOUS,
        image_shape=(600, 600),
        frame_skip=10,
        time_step=0.01,
        settle_steps=50,
        create_scene_kwargs={
            "gripper_randomization": {
                "angle_reset_noise": 0.0,
                "ptsd_reset_noise": np.array([10.0, 10.0, 40.0, 5.0]),
                "rcm_reset_noise": np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0]),
            },
        },
        num_active_pegs=3,
        randomize_color=True,
        num_torus_tracking_points=5,
        start_grasped=True,
        only_learn_pick=False,
        minimum_lift_height=50.0,
        randomize_torus_position=False,
    )

    env.reset()
    done = False

    # Get position of grasp index of torus
    # Plan linear motion towards that Point (using Yannic's IK)
    # Grasp
    # Lift up

    counter = 0
    fps_list = deque(maxlen=100)
    while not done:
        for _ in range(500):
            start = time.perf_counter()
            action = env.action_space.sample()
            action[:] = 0.0
            action[3] = -1.0
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            end = time.perf_counter()
            fps = 1 / (end - start)
            fps_list.append(fps)
            # pp.pprint(done)
            # pp.pprint(info["successful_task"])
            pp.pprint(info["distance_to_minimum_pick_height"])
            pp.pprint(info["grasped_torus"])

            # for key in info.keys():
            #     print(key)

            # print("\n\n")

            # pp.pprint({key: val for key, val in info.items() if not np.all(np.array(val) == 0)})
            # print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")
        env.reset()
