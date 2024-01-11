import gymnasium.spaces as spaces
import numpy as np

from collections import defaultdict, deque
from enum import Enum, unique
from pathlib import Path
from functools import reduce
from typing import Callable, Union, Tuple, Optional, List, Any, Dict

from sofa_env.base import SofaEnv, RenderMode, RenderFramework

from sofa_env.scenes.ligating_loop.sofa_objects.gripper import ArticulatedGripper
from sofa_env.scenes.ligating_loop.sofa_objects.cavity import Cavity
from sofa_env.scenes.ligating_loop.sofa_objects.loop import LigatingLoop
from sofa_env.scenes.ligating_loop.sofa_objects.loop import ActionType as LoopActionType
from sofa_env.scenes.rope_threading.sofa_objects.camera import ControllableCamera

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
    """Action type specifies whether the actions are continuous values that are added to the state, or discrete actions that increment the state."""

    DISCRETE = 0
    CONTINUOUS = 1
    POSITION = 2


class LigatingLoopEnv(SofaEnv):
    """Ligating Loop Environment

    The goal of this environment is to put a loop over a cavity up to a visual marking and then close the loop
    to constrict the cavity.

    Args:
        scene_path (Union[str, Path]): Path to the scene description script that contains this environment's ``createScene`` function.
        image_shape (Tuple[int, int]): Height and Width of the rendered images.
        observation_type (ObservationType): Whether to return RGB images or an array of states as the observation.
        action_type (ActionType): Discrete or continuous actions to define the action space of the environment.
        time_step (float): size of simulation time step in seconds (default: 0.01).
        frame_skip (int): number of simulation time steps taken (call ``_do_action`` and advance simulation) each time step is called (default: 1).
        settle_steps (int): How many steps to simulate without returning an observation after resetting the environment.
        render_mode (RenderMode): Create a window (``RenderMode.HUMAN``), run headless (``RenderMode.HEADLESS``), or do not create a render buffer at all (``RenderMode.NONE``).
        render_framework (RenderFramework): choose between pyglet and pygame for rendering
        reward_amount_dict (dict): Dictionary to weigh the components of the reward function.
        maximum_state_velocity (Union[np.ndarray, float]): Velocity in deg/s for pts and mm/s for d in state space which are applied with a normalized action of value 1.
        discrete_action_magnitude (Union[np.ndarray, float]): Discrete change in state space in deg/s for pts and mm/s for d.
        create_scene_kwargs (Optional[dict]): A dictionary to pass additional keyword arguments to the ``createScene`` function.
        on_reset_callbacks (Optional[List[Callable]]): Functions that are called as the last step of the ``env.reset()`` function.
        num_tracking_points_cavity (int): Number of evenly spaced points on the cavity to include in the state observation.
        num_tracking_points_marking (int): Number of evenly spaced points on the cavity marking to include in the state observation.
        num_tracking_points_loop (int): Number of evenly spaced points on the loop to include in the state observation.
        target_loop_closed_ratio (float): How far the loop has to closed to consider the episode done.
        target_loop_overlap_ratio (float): How much of potential overlap between loop and marking are to be reached to consider the episode done.
        randomize_marking_position (bool): Whether to change the location of the marking on the cavity on each reset.
        with_gripper (bool): Whether to include a gripper as a second instrument.
        individual_agents (bool): Whether the instruments are controlled individually, or the action is one large array.
        band_width (float): Width of the marking in millimeters.
        disable_in_cavity_checks (bool): Whether to disable the reward features instrument_not_in_cavity and loop_center_in_cavity.
    """

    def __init__(
        self,
        scene_path: Union[str, Path] = SCENE_DESCRIPTION_FILE_PATH,
        image_shape: Tuple[int, int] = (124, 124),
        observation_type: ObservationType = ObservationType.RGB,
        action_type: ActionType = ActionType.CONTINUOUS,
        time_step: float = 0.01,
        frame_skip: int = 10,
        settle_steps: int = 50,
        render_mode: RenderMode = RenderMode.HEADLESS,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        reward_amount_dict={
            "distance_loop_to_marking_center": -0.0,
            "delta_distance_loop_to_marking_center": -0.0,
            "loop_center_in_cavity": 0.0,
            "instrument_not_in_cavity": -0.0,
            "instrument_shaft_collisions": -0.0,
            "loop_marking_overlap": 0.0,
            "loop_closed_around_marking": 0.0,
            "loop_closed_in_thin_air": -0.0,
            "successful_task": 0.0,
        },
        maximum_state_velocity: Union[np.ndarray, float] = np.array([20.0, 20.0, 20.0, 20.0, 1.0]),
        discrete_action_magnitude: Union[np.ndarray, float] = np.array([15.0, 15.0, 15.0, 10.0, 1.0]),
        create_scene_kwargs: Optional[dict] = None,
        on_reset_callbacks: Optional[List[Callable]] = None,
        num_tracking_points_cavity: Dict[str, int] = {
            "height": 3,
            "radius": 1,
            "angle": 3,
        },
        num_tracking_points_marking: Dict[str, int] = {
            "height": 1,
            "radius": 1,
            "angle": 3,
        },
        num_tracking_points_loop: int = 6,
        target_loop_closed_ratio: float = 0.5,
        target_loop_overlap_ratio: float = 0.1,
        randomize_marking_position: bool = True,
        with_gripper: bool = False,
        individual_agents: bool = False,
        band_width: float = 6.0,
        disable_in_cavity_checks: bool = False,
        randomize_loop_state: bool = True,
        loop_ptsd_reset_noise: np.ndarray = np.array([20.0, 20.0, 0.0, 10.0]),
        loop_closure_reset_interval: np.ndarray = np.array([0.0, 1.0]),
    ) -> None:
        # Pass image shape to the scene creation function
        if create_scene_kwargs is None:
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape
        create_scene_kwargs["with_gripper"] = with_gripper
        create_scene_kwargs["band_width"] = band_width

        if action_type == ActionType.POSITION:
            loop_action_type = LoopActionType.POSITION
        elif action_type == ActionType.CONTINUOUS:
            loop_action_type = LoopActionType.CONTINUOUS
        else:
            loop_action_type = LoopActionType.DISCRETE
        create_scene_kwargs["loop_action_type"] = loop_action_type

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

        self.disable_in_cavity_checks = disable_in_cavity_checks

        self.individual_agents = individual_agents
        self.with_gripper = with_gripper
        self.maximum_loop_state_velocity = maximum_state_velocity.copy() if isinstance(maximum_state_velocity, np.ndarray) else np.array([maximum_state_velocity] * 5)
        self.maximum_loop_state_velocity[-1] = 1.0 / self.time_step
        self.maximum_gripper_state_velocity = maximum_state_velocity

        self.num_tracking_points_cavity = num_tracking_points_cavity
        self.num_tracking_points_marking = num_tracking_points_marking
        self.num_tracking_points_loop = num_tracking_points_loop
        self.target_loop_closed_ratio = target_loop_closed_ratio
        self.target_loop_overlap_ratio = target_loop_overlap_ratio
        self.randomize_marking_position = randomize_marking_position
        self.band_width = band_width

        self.randomize_loop_state = randomize_loop_state
        self.loop_ptsd_reset_noise = loop_ptsd_reset_noise
        self.loop_closure_reset_interval = loop_closure_reset_interval

        ##############
        # Action Space
        ##############
        min_angle = 0.0
        max_angle = 60.0

        action_dimensionality = 5
        self.action_type = action_type
        if action_type == ActionType.POSITION:
            action_space_limits = {
                "low": np.array([-90.0, -90.0, np.finfo(np.float16).min, 0.0]),
                "high": np.array([90.0, 90.0, np.finfo(np.float16).max, 100.0]),
            }
            self.min_angle = min_angle
            self.max_angle = max_angle
            if with_gripper:
                if individual_agents:
                    self._do_action = self._do_action_dict
                    self.action_space = spaces.Dict(
                        {
                            "ligating_loop": spaces.Box(
                                low=np.append(action_space_limits["low"], 0.0),
                                high=np.append(action_space_limits["high"], 1.0),
                                shape=(action_dimensionality,),
                                dtype=np.float32,
                            ),
                            "gripper": spaces.Box(
                                low=np.append(action_space_limits["low"], min_angle),
                                high=np.append(action_space_limits["high"], max_angle),
                                shape=(action_dimensionality,),
                                dtype=np.float32,
                            ),
                        }
                    )
                else:
                    self._do_action = self._do_action_array
                    # The first 5 values are for the gripper, the remaining 5 for the loop
                    self.action_space = spaces.Box(
                        low=np.concatenate(
                            np.append(action_space_limits["low"], 0.0),
                            np.append(action_space_limits["low"], 1.0),
                        ),
                        high=np.concatenate(
                            np.append(action_space_limits["high"], max_rope_index),
                            np.append(action_space_limits["high"], max_angle),
                        ),
                        shape=(action_dimensionality * 2,),
                        dtype=np.float32,
                    )

            else:
                self._do_action = self._do_loop_action
                self.action_space = spaces.Box(
                    low=np.append(action_space_limits["low"], 0.0),
                    high=np.append(action_space_limits["high"], 1.0),
                    shape=(action_dimensionality,),
                    dtype=np.float32,
                )

        elif action_type == ActionType.CONTINUOUS:
            self._scale_action = self._scale_continuous_action
            if with_gripper:
                if individual_agents:
                    self._do_action = self._do_action_dict
                    self.action_space = spaces.Dict(
                        {
                            "ligating_loop": spaces.Box(low=-1.0, high=1.0, shape=(action_dimensionality,), dtype=np.float32),
                            "gripper": spaces.Box(low=-1.0, high=1.0, shape=(action_dimensionality,), dtype=np.float32),
                        }
                    )
                else:
                    self._do_action = self._do_action_array
                    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dimensionality * 2,), dtype=np.float32)
            else:
                self._do_action = self._do_loop_action
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dimensionality,), dtype=np.float32)
        else:
            self._scale_action = self._scale_discrete_action
            if with_gripper:
                if individual_agents:
                    self._do_action = self._do_action_dict
                    self.action_space = spaces.Dict(
                        {
                            "ligating_loop": spaces.Discrete(action_dimensionality * 2 + 1),
                            "gripper": spaces.Discrete(action_dimensionality * 2 + 1),
                        }
                    )
                else:
                    raise NotImplementedError("Discrete action space not implemented for with_gripper=True and individual_agents=False.")
            else:
                self._do_action = self._do_loop_action
                self.action_space = spaces.Discrete(action_dimensionality * 2 + 1)

            if isinstance(discrete_action_magnitude, np.ndarray):
                if not len(discrete_action_magnitude) == action_dimensionality * 2:
                    raise ValueError(f"If you want to use individual discrete action step sizes per action dimension, please pass an array of length {action_dimensionality * 2} as discrete_action_magnitude. Received {discrete_action_magnitude=} with lenght {len(discrete_action_magnitude)}.")

            # [step, 0, 0, ...], [-step, 0, 0, ...], [0, step, 0, ...], [0, -step, 0, ...]
            action_list = []
            for i in range(action_dimensionality * 2):
                action = [0.0] * (action_dimensionality * 2)
                step_size = discrete_action_magnitude if isinstance(discrete_action_magnitude, float) else discrete_action_magnitude[int(i / 2)]
                action[int(i / 2)] = (1 - 2 * (i % 2)) * step_size
                action_list.append(action)

            # Noop action
            action_list.append([0.0] * (action_dimensionality * 2))

            self._discrete_action_lookup_gripper = np.array(action_list)
            self._discrete_action_lookup_gripper *= self.time_step
            self._discrete_action_lookup_gripper.flags.writeable = False

            self._discrete_action_lookup_loop = np.array(action_list)
            self._discrete_action_lookup_loop *= self.time_step
            self._discrete_action_lookup_loop[-3:] *= 1 / (self.time_step * discrete_action_magnitude[-1]) if isinstance(discrete_action_magnitude, np.ndarray) else 1 / (self.time_step * discrete_action_magnitude)

            self._discrete_action_lookup_loop.flags.writeable = False

        ###################
        # Observation Space
        ###################

        if observation_type == ObservationType.STATE:
            # ptsd_state -> 4
            # active_index -> 1
            # loop_tracking_points -> num_tracking_points_loop*3
            # cavity_tracking_point_indices -> r*h*a*3
            # marking_tracking_point_indices -> r2*h2*a2*3
            # gripper_pose -> 7
            # gripper_ptsda -> 5
            observations_size = 4 + 1 + num_tracking_points_loop * 3 + sum(num_tracking_points_cavity.values()) * 3 + sum(num_tracking_points_marking.values()) * 3
            if self.with_gripper:
                observations_size += 7 + 5
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observations_size,), dtype=np.float32)

        # Image observations
        elif observation_type == ObservationType.RGB:
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (3,), dtype=np.uint8)

        # Color image and depth map
        elif observation_type == ObservationType.RGBD:
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (4,), dtype=np.uint8)

        else:
            raise ValueError(f"Please set observation_type to a value of ObservationType. Received {observation_type}.")

        self.observation_type = observation_type

        #########################
        # Episode specific values
        #########################
        # Infos per episode
        self.episode_info = defaultdict(float)
        self.loop_activation_counter = 0

        # Infos from the reward
        self.reward_info = {}
        self.reward_features = {}

        # All parameters of the reward function that are not passed default to 0.0
        self.reward_amount_dict = defaultdict(float) | reward_amount_dict

        # Callback functions called on reset
        self.on_reset_callbacks = on_reset_callbacks if on_reset_callbacks is not None else []

    def _init_sim(self) -> None:
        super()._init_sim()

        self.gripper: ArticulatedGripper = self.scene_creation_result["gripper"]
        self.loop: LigatingLoop = self.scene_creation_result["loop"]
        self.cavity: Cavity = self.scene_creation_result["cavity"]
        self.camera: ControllableCamera = self.scene_creation_result["camera"]
        self.contact_listeners: Dict = self.scene_creation_result["contact_listeners"]

        # Factor for normalizing the distances in the reward function.
        # Based on the workspace of the loop.
        self._distance_normalization_factor = 1.0 / np.linalg.norm(self.loop.cartesian_workspace["high"] - self.loop.cartesian_workspace["low"])

        # Get the collision model indices that should be in contact with the ligating loop when it closes
        self.colored_collision_model_indices = self.cavity.colored_collision_model_indices
        self.colored_fem_indices = self.cavity.colored_fem_indices

        # Tracking points on the loop
        if self.num_tracking_points_loop > 0:
            if self.loop.rope.num_points < self.num_tracking_points_loop:
                raise Exception(f"The number of loop tracking points ({self.num_tracking_points_loop}) is larger than the number of points on the loop ({self.loop.rope.num_points}).")
            self.loop_tracking_point_indices = np.linspace(0, self.loop.rope.num_points, num=self.num_tracking_points_loop, endpoint=False, dtype=np.int16)
        elif self.num_tracking_points_loop == -1:
            self.loop_tracking_point_indices = np.array(range(self.loop.rope.num_points), dtype=np.int16)
        elif self.num_tracking_points_loop == 0:
            self.loop_tracking_point_indices = None
        else:
            if self.observation_type == ObservationType.STATE:
                raise ValueError(f"num_tracking_points_loop must be > 0 or == -1 (to use them all). Received {self.num_tracking_points_loop}.")

        # Tracking points on the cavity
        self.cavity_tracking_point_indices = self.cavity.get_subsampled_indices(
            discretization_height=self.num_tracking_points_cavity["height"],
            discretization_angle=self.num_tracking_points_cavity["angle"],
            discretization_radius=self.num_tracking_points_cavity["radius"],
        )
        if len(self.cavity_tracking_point_indices) == 0 and self.observation_type == ObservationType.STATE:
            raise ValueError(f"Something went wrong with determining the tracking points on the cavity...\n{self.num_tracking_points_cavity=}")

        # Tracking points on the cavity's colored marking
        self.marking_tracking_point_indices = self.cavity.get_subsampled_indices_on_band(
            discretization_height=self.num_tracking_points_marking["height"],
            discretization_angle=self.num_tracking_points_marking["angle"],
            discretization_radius=self.num_tracking_points_marking["radius"],
        )
        if len(self.marking_tracking_point_indices) == 0 and self.observation_type == ObservationType.STATE:
            raise ValueError(f"Something went wrong with determining the tracking points on the marked band...\n{self.num_tracking_points_marking=}")

    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        """Step function of the environment that applies the action to the simulation and returns observation, reward, done signal, and info."""

        maybe_rgb_observation = super().step(action)

        observation = self._get_observation(maybe_rgb_observation=maybe_rgb_observation)
        reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()

        return observation, reward, done, False, info

    def _scale_continuous_action(self, action: np.ndarray, loop: bool = True) -> np.ndarray:
        """
        Policy is output is clipped to [-1, 1] e.g. [-0.3, 0.8, 1].
        We want to scale that to the maximum velocities defined
        in ``maximum_state_velocity`` in [angle, angle, angle, mm, angle] / step.
        and further to per second (because delta T is not 1 second).
        """
        if loop:
            return self.time_step * self.maximum_loop_state_velocity * action
        else:
            return self.time_step * self.maximum_gripper_state_velocity * action

    def _scale_discrete_action(self, action: int, loop: bool = True) -> np.ndarray:
        """Maps action indices to a motion delta."""
        if loop:
            return self._discrete_action_lookup_loop[action]
        else:
            return self._discrete_action_lookup_gripper[action]

    def _do_action(self, action) -> None:
        """Only defined to satisfy ABC."""
        pass

    def _do_action_dict(self, action: Dict[str, np.ndarray]) -> None:
        """Apply action to the simulation."""

        if self.action_type in [ActionType.CONTINUOUS, ActionType.DISCRETE]:
            self.gripper.set_articulated_state(self.gripper.get_articulated_state() + self._scale_action(action["gripper"], loop=False))
            self.loop_activation_counter += 1
            if self.loop_activation_counter == 1:
                pass
            else:
                if self.loop_activation_counter < self.frame_skip:
                    action["ligating_loop"][-1] = 0.0
                else:
                    self.loop_activation_counter = 0
            self.loop.do_action(self._scale_action(action["ligating_loop"]))
        else:  # Position control, directly set loop to desired position
            self.gripper.set_articulated_state(action["gripper"])
            self.loop.do_action(action["ligating_loop"])

    def _do_action_array(self, action: np.ndarray) -> None:
        """Apply action to the simulation."""
        if self.action_type in [ActionType.CONTINUOUS, ActionType.DISCRETE]:
            self.gripper.set_articulated_state(self.gripper.get_articulated_state() + self._scale_action(action[:5], loop=False))
            self.loop_activation_counter += 1
            if self.loop_activation_counter == 1:
                pass
            else:
                if self.loop_activation_counter < self.frame_skip:
                    action[-1] = 0.0
                else:
                    self.loop_activation_counter = 0

            self.loop.do_action(self._scale_action[action[5:]])
        else:  # Position control, directly set loop to desired position
            self.gripper.set_articulated_state(action[:5])
            self.loop.do_action(action[5:])

    def _do_loop_action(self, action: np.ndarray) -> None:
        """Apply action to the simulation."""

        if self.action_type in [ActionType.CONTINUOUS, ActionType.DISCRETE]:
            self.loop_activation_counter += 1
            if self.loop_activation_counter == 1:
                pass
            else:
                if self.loop_activation_counter < self.frame_skip:
                    action[-1] = 0.0
                else:
                    self.loop_activation_counter = 0

            self.loop.do_action(self._scale_action(action))
        else:  # Position control, directly set loop to desired position
            self.loop.do_action(action)

    def _get_reward_features(self, previous_reward_features: dict) -> dict:
        """Get the features that may be used to assemble the reward function

        Features:
            - distance_loop_to_marking_center (float): Distance between the loop center and the center of the marking.
            - delta_distance_loop_to_marking_center (float): Change in distance between the loop center and the center of the marking.
            - loop_center_in_cavity (bool): Whether the loop center is inside the cavity.
            - instrument_not_in_cavity (bool): Whether the instrument is not inside the cavity.
            - instrument_shaft_collisions (int): Number of collisions between the instrument shaft and the cavity.
            - loop_marking_overlap (float): Ratio of collions between loop and marking possible collision elements of the marking.
            - loop_closed_around_marking (float): Closing the loop [0, 1] scaled by contact to the colored marking, masked if the loop is not around cavity
            - loop_closed_in_thin_air (float): Closing the loop [0, 1] when the loop is not around the cavity
            - successful_task (bool): Success, if loop closed ``ratio > self.target_loop_closed_ratio`` and ``loop_marking_overlap > self.target_loop_overlap_ratio``, while loop around cavity.
        """

        reward_features = {}

        cavity_state = self.cavity.get_state()
        cavity_marking_center_of_mass = np.mean(cavity_state[self.cavity.colored_fem_indices], axis=0)
        loop_center_of_mass = self.loop.get_center_of_mass()

        # Distance between the loop's center of mass and the cavity's colored marking center of mass.
        reward_features["distance_loop_to_marking_center"] = np.linalg.norm(cavity_marking_center_of_mass - loop_center_of_mass)
        reward_features["delta_distance_loop_to_marking_center"] = reward_features["distance_loop_to_marking_center"] - previous_reward_features["distance_loop_to_marking_center"]

        # Is the center of mass of the loop inside the cavity's shell?
        if self.disable_in_cavity_checks:
            reward_features["loop_center_in_cavity"] = False
        else:
            reward_features["loop_center_in_cavity"] = self.cavity.is_in_cavity(loop_center_of_mass)

        # Is the instrument tip inside the cavity's shell?
        if self.disable_in_cavity_checks:
            reward_features["instrument_not_in_cavity"] = True
        else:
            reward_features["instrument_not_in_cavity"] = not self.cavity.is_in_cavity(self.loop.get_pose()[:3])

        # Unwanted collisions between the instrument shaft and the cavity.
        reward_features["instrument_shaft_collisions"] = self.contact_listeners["shaft"].getNumberOfContacts()

        # Desired contacts between loop and colored marking on the cavity.
        loop_contacts = self.contact_listeners["loop"].getContactElements()
        loop_contacts_with_marking = []
        maximum_contact_amount = len(self.cavity.colored_collision_model_indices)
        for contact in loop_contacts:
            triangle_index_on_cavity = contact[1] if contact[0] == 0 else contact[3]
            if triangle_index_on_cavity in self.cavity.colored_collision_model_indices:
                loop_contacts_with_marking.append(triangle_index_on_cavity)
        # Count how many (unique) triangles of the marking are in contact with the loop, divided by the maximum amount of triangles on the marking.
        reward_features["loop_marking_overlap"] = len(set(loop_contacts_with_marking)) / maximum_contact_amount

        # Closing the loop [0, 1] scaled by contact to the colored marking, masked if the loop is not around cavity
        if self.disable_in_cavity_checks:
            reward_features["loop_closed_around_marking"] = self.loop.get_ratio_loop_closed() * reward_features["loop_marking_overlap"]
        else:
            reward_features["loop_closed_around_marking"] = self.loop.get_ratio_loop_closed() * reward_features["loop_marking_overlap"] * reward_features["loop_center_in_cavity"] * reward_features["instrument_not_in_cavity"]

        # Closing the loop [0, 1] when the loop is not around the cavity
        reward_features["loop_closed_in_thin_air"] = self.loop.get_ratio_loop_closed() * (1 - reward_features["loop_center_in_cavity"])

        # Success, if loop closed ratio > self.target_loop_closed_ratio and loop_marking_overlap > self.target_loop_overlap_ratio, while loop around cavity
        if reward_features["loop_closed_around_marking"]:
            reward_features["successful_task"] = reward_features["loop_marking_overlap"] > self.target_loop_overlap_ratio and self.loop.get_ratio_loop_closed() > self.target_loop_closed_ratio
        else:
            reward_features["successful_task"] = False

        return reward_features

    def _get_reward(self) -> float:
        """Retrieve the reward features and scale them with the ``reward_amount_dict``."""
        reward = 0.0
        self.reward_info = {}

        reward_features = self._get_reward_features(self.reward_features)
        # we change the values of the dict -> do a copy (deepcopy not necessary, because the value itself is not manipulated)
        self.reward_features = reward_features.copy()

        for key, value in reward_features.items():
            if "distance" in key or "velocity" in key:
                if np.isnan(value):
                    value = 1.0 / self._distance_normalization_factor
                value = np.clip(value, -1.0 / self._distance_normalization_factor, 1.0 / self._distance_normalization_factor)
                value = self._distance_normalization_factor * value
            self.reward_info[f"reward_{key}"] = self.reward_amount_dict[key] * value
            reward += self.reward_info[f"reward_{key}"]

        self.reward_info["reward"] = reward

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
            cavity_state = self.cavity.get_state()
            state_dict["ptsd_state"] = self.loop.get_state()
            state_dict["active_index"] = np.asarray(self.loop.get_active_index())[None]  # 4 -> [4]
            state_dict["loop_tracking_positions"] = self.loop.get_loop_positions()[self.loop_tracking_point_indices].ravel()
            state_dict["cavity_tracking_positions"] = cavity_state[self.cavity_tracking_point_indices].ravel()
            state_dict["marking_tracking_positions"] = cavity_state[self.marking_tracking_point_indices].ravel()

            if self.with_gripper:
                state_dict["gripper_pose"] = self.gripper.get_pose()
                state_dict["gripper_ptsda_state"] = self.gripper.get_articulated_state()

            observation = np.concatenate(tuple(state_dict.values()))
            # Overwrite possible NaNs with the maximum distance in the workspace
            observation = np.where(np.isnan(observation), 1.0 / self._distance_normalization_factor, observation)

        return observation

    def _get_info(self) -> dict:
        """Assemble the info dictionary."""

        self.info = {}

        for key, value in self.reward_info.items():
            # shortens 'reward_delta_gripper_distance_to_torus_tracking_points'
            # to 'ret_del_gri_dis_to_tor_tra_poi'
            words = key.split("_")[1:]
            shortened_key = reduce(lambda x, y: x + "_" + y[:3], words, "ret")
            self.episode_info[shortened_key] += value

        return {**self.info, **self.reward_info, **self.episode_info, **self.reward_features}

    def reset(self, seed: Union[int, np.random.SeedSequence, None] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Union[np.ndarray, None], Dict]:
        """Reset the state of the environment and return the initial observation."""
        # Reset from parent class -> calls the simulation's reset function
        super().reset(seed)

        # Seed the instruments
        if self.unconsumed_seed:
            with_gripper = self.gripper is not None
            seeds = self.seed_sequence.spawn(2 if with_gripper else 1)
            self.loop.seed(seed=seeds[0])
            if with_gripper:
                self.gripper.seed(seed=seeds[1])
            self.unconsumed_seed = False

        # Reset loop
        self.loop.reset_state()
        self.loop_activation_counter = 0

        def reset_loop(state: np.ndarray, closedness: float):
            # denormalize loop closedness to range [min_active_index, max_active index]
            # but inverse, i.e. closedness=0 corresponds to max_active_index
            min_active_index = 5
            max_active_index = self.loop.rope.num_points - 1
            new_active_index_float = min_active_index + (1 - closedness) * (max_active_index - min_active_index)
            new_active_index_int = round(new_active_index_float)
            new_active_index_int_clipped = max(min_active_index, min(new_active_index_int, max_active_index))
            #self.loop.active_index = new_active_index_int_clipped
            for index in range(self.loop.active_index, new_active_index_int_clipped - 1, -1):
                self.loop.active_index = index
                self.sofa_simulation.animate(self._sofa_root_node, self._sofa_root_node.getDt())
            #assert self.loop.active_index == new_active_index_int_clipped
            self.loop.set_state(state)

        if options and "loop_state" in options:
            reset_loop(options["loop_state"][:-1], options["loop_state"][-1])
        elif self.randomize_loop_state:
            # Optionally add noise to the loop
            loop_closedness = self.rng.uniform(*self.loop_closure_reset_interval)
            ptsd_state_noise = self.rng.uniform(-self.loop_ptsd_reset_noise, self.loop_ptsd_reset_noise)
            reset_loop(self.loop.get_state() + ptsd_state_noise, loop_closedness)

        # Reset the gripper
        if self.gripper is not None:
            # Reset the gripper
            self.gripper.reset_gripper()

        # Colored band on cavity
        if self.randomize_marking_position:
            band_start = self.rng.uniform(20, self.cavity.height - 10)
        else:
            band_start = self.cavity.height - 15.0
        self.cavity.color_band(band_start, band_start + self.band_width)
        self.colored_collision_model_indices = self.cavity.colored_collision_model_indices
        self.colored_fem_indices = self.cavity.colored_fem_indices

        # Clear the episode info values
        for key in self.episode_info:
            self.episode_info[key] = 0.0

        # and the reward info dict
        self.reward_info = {}

        # and fill the first values used as previous_reward_features
        self.reward_features = {}
        cavity_state = self.cavity.get_state()
        cavity_marking_center_of_mass = np.mean(cavity_state[self.cavity.colored_fem_indices], axis=0)
        loop_center_of_mass = self.loop.get_center_of_mass()
        self.reward_features["distance_loop_to_marking_center"] = np.linalg.norm(cavity_marking_center_of_mass - loop_center_of_mass)

        # Execute any callbacks passed to the env.
        for callback in self.on_reset_callbacks:
            callback(self)

        # Animate several timesteps without actions until simulation settles
        for _ in range(self._settle_steps):
            self.sofa_simulation.animate(self._sofa_root_node, self._sofa_root_node.getDt())

        # Do one call to the jit compiled function to move the compunation time here, instead of the first step call.
        if not self.disable_in_cavity_checks:
            loop_center_of_mass = self.loop.get_center_of_mass()
            self.cavity.is_in_cavity(loop_center_of_mass)
            self.cavity.is_in_cavity(self.loop.get_pose()[:3])

        return self._get_observation(maybe_rgb_observation=self._maybe_update_rgb_buffer()), {}


if __name__ == "__main__":
    import pprint
    import time

    pp = pprint.PrettyPrinter()

    env = LigatingLoopEnv(
        observation_type=ObservationType.STATE,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.CONTINUOUS,
        image_shape=(800, 800),
        frame_skip=1,
        time_step=0.1,
        settle_steps=50,
    )

    env.reset()
    done = False

    fps_list = deque(maxlen=100)
    while not done:
        for _ in range(100):
            start = time.perf_counter()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            end = time.perf_counter()
            done = terminated or truncated
            fps = 1 / (end - start)
            fps_list.append(fps)
            # pp.pprint(info)
            print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")

        env.reset()
