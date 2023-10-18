import gymnasium.spaces as spaces
import numpy as np

from collections import defaultdict, deque
from enum import Enum, unique
from pathlib import Path
from functools import reduce

from typing import Callable, Union, Tuple, Optional, List, Any, Dict
from sofa_env.base import SofaEnv, RenderMode, RenderFramework
from sofa_env.scenes.deflect_spheres.sofa_objects.post import Post, State

from sofa_env.scenes.tissue_dissection.sofa_objects.cauter import PivotizedCauter
from sofa_env.sofa_templates.camera import PhysicalCamera
from sofa_env.utils.math_helper import distance_between_line_segments

HERE = Path(__file__).resolve().parent
SCENE_DESCRIPTION_FILE_PATH = HERE / "scene_description.py"


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


@unique
class Mode(Enum):
    WITH_REPLACEMENT = 0
    WITHOUT_REPLACEMENT = 1


class DeflectSpheresEnv(SofaEnv):
    """Deflect Spheres Environment

    The goal of this environment is to touch and move the highlighted spheres with one or more cauter instruments.
    The positions of the spheres are randomized at each reset.
    The workspace size may be adapted through the ``create_scene_kwargs``. See ``scene_description.py`` for more details.

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
        single_agent (bool): Whether to create the scene with one, or two cauter instruments.
        individual_agents (bool): Whether the instruments are controlled individually, or the action is one large array.
        num_objects (int): How many spheres to place in the scene.
        num_deflect_to_win (int): How many spheres are to be deflected correctly, before the episode is marked is done.
        min_deflection_distance (float): How many millimeters the spheres have to be deflected before marked as done.
        mode (Mode): Whether the done spheres stay marked as done (``Mode.WITHOUT_REPLACEMENT``) or can be chosen again (``Mode.WITH_REPLACEMENT``).
        allow_deflection_with_instrument_shaft (bool): If set to ``False`` the spheres have to be in contact with the cauter tip to be marked as done.
    """

    def __init__(
        self,
        scene_path: Union[str, Path] = SCENE_DESCRIPTION_FILE_PATH,
        image_shape: Tuple[int, int] = (124, 124),
        observation_type: ObservationType = ObservationType.RGB,
        action_type: ActionType = ActionType.CONTINUOUS,
        time_step: float = 0.1,
        frame_skip: int = 1,
        settle_steps: int = 10,
        render_mode: RenderMode = RenderMode.HEADLESS,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        reward_amount_dict={
            "action_violated_cartesian_workspace": -0.0,
            "action_violated_state_limits": -0.0,
            "tool_collision": -0.0,
            "distance_to_active_sphere": -0.0,
            "delta_distance_to_active_sphere": -0.0,
            "deflection_of_inactive_spheres": -0.0,
            "deflection_of_active_sphere": 0.0,
            "delta_deflection_of_active_sphere": 0.0,
            "done_with_active_sphere": 0.0,
            "successful_task": 0.0,
            "rcm_violation_xyz": -0.0,
            "rcm_violation_rotation": -0.0,
        },
        maximum_state_velocity: Union[np.ndarray, float] = np.array([20.0, 20.0, 20.0, 20.0]),
        discrete_action_magnitude: Union[np.ndarray, float] = np.array([15.0, 15.0, 15.0, 10.0]),
        create_scene_kwargs: Optional[dict] = None,
        on_reset_callbacks: Optional[List[Callable]] = None,
        single_agent: bool = True,
        individual_agents: bool = False,
        num_objects: int = 5,
        num_deflect_to_win: int = 5,
        min_deflection_distance: float = 3.0,
        mode: Mode = Mode.WITH_REPLACEMENT,
        allow_deflection_with_instrument_shaft: bool = False,
    ) -> None:
        # Pass image shape to the scene creation function
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape
        create_scene_kwargs["single_agent"] = single_agent
        create_scene_kwargs["num_objects"] = num_objects

        self.single_agent = single_agent
        self.individual_agents = individual_agents
        if allow_deflection_with_instrument_shaft and not single_agent:
            print(
                "[WARNING]: Allowing deflection with the instrument shaft is only valid for the single agent case. \
                    If you want to allow deflection with the instrument shaft for multiple agents, \
                    please add ContactListener between the shaft collision models and the post collision models. \
                    Setting allow_deflection_with_instrument_shaft to False."
            )
        self.allow_deflection_with_instrument_shaft = allow_deflection_with_instrument_shaft and single_agent

        self.num_objects = num_objects
        self.num_deflect_to_win = num_deflect_to_win
        self.min_deflection_distance = min_deflection_distance
        self.mode = mode

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

        self.maximum_state_velocity = maximum_state_velocity
        self.num_deflections = 0

        ##############
        # Action Space
        ##############
        if self.individual_agents and self.single_agent:
            raise ValueError("Cannot have both individual agents (Dict action space) and single agent.")

        if self.single_agent:
            action_dimensionality = 4
        else:
            if self.individual_agents:
                action_dimensionality = 4
            else:
                action_dimensionality = 8

        self.action_type = action_type

        # Discrete action space is a lookup table with intereger values mapping to a continuous action vector
        if action_type == ActionType.DISCRETE:
            self._scale_action = self._scale_discrete_action
            if isinstance(discrete_action_magnitude, np.ndarray):
                if not len(discrete_action_magnitude) == action_dimensionality:
                    raise ValueError(f"If you want to use individual discrete action step sizes per action dimension, please pass an array of length {action_dimensionality} as discrete_action_magnitude. Received {discrete_action_magnitude=} with lenght {len(discrete_action_magnitude)}.")

            # Single agent
            if self.single_agent:
                self._do_action = self._do_action_single
                self.action_space = spaces.Discrete(action_dimensionality * 2 + 1)
            # Multi-agent
            else:
                # Individual agents
                if self.individual_agents:
                    self._do_action = self._do_action_individual
                    self.action_space = spaces.Dict(
                        {
                            "left_cauter": spaces.Discrete(action_dimensionality * 2 + 1),
                            "right_cauter": spaces.Discrete(action_dimensionality * 2 + 1),
                        }
                    )
                # Joint agents
                else:
                    self._do_action = self._do_action_multi
                    self.action_space = spaces.Discrete(action_dimensionality * 2 + 1)

            # [step, 0, 0, ...], [-step, 0, 0, ...], [0, step, 0, ...], [0, -step, 0, ...]
            action_list = []
            for i in range(action_dimensionality * 2):
                action = [0.0] * (action_dimensionality)
                step_size = discrete_action_magnitude if isinstance(discrete_action_magnitude, float) else discrete_action_magnitude[int(i / 2)]
                action[int(i / 2)] = (1 - 2 * (i % 2)) * step_size
                action_list.append(action)

            # Noop action
            action_list.append([0.0] * (action_dimensionality))

            self._discrete_action_lookup = np.array(action_list)
            self._discrete_action_lookup *= self.time_step
            self._discrete_action_lookup.flags.writeable = False

        else:
            # Determine the action space limits and the scaling factor
            if action_type == ActionType.CONTINUOUS:
                action_space_limits = {
                    "low": -np.ones(4, dtype=np.float32),
                    "high": np.ones(4, dtype=np.float32),
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
                    "low": np.array([-90.0, -90.0, -90.0, 0.0]),
                    "high": np.array([90.0, 90.0, 90.0, 300.0]),
                }

            # Determine the do_action function
            if self.individual_agents:
                self._do_action = self._do_action_individual
                self.action_space = spaces.Dict(
                    {
                        "left_cauter": spaces.Box(low=action_space_limits["low"], high=action_space_limits["high"], shape=(action_dimensionality,), dtype=np.float32),
                        "right_cauter": spaces.Box(low=action_space_limits["low"], high=action_space_limits["high"], shape=(action_dimensionality,), dtype=np.float32),
                    }
                )
            else:
                if self.single_agent:
                    self._do_action = self._do_action_single
                else:
                    self._do_action = self._do_action_multi
                    action_space_limits["low"] = np.concatenate((action_space_limits["low"], action_space_limits["low"]))
                    action_space_limits["high"] = np.concatenate((action_space_limits["high"], action_space_limits["high"]))
                self.action_space = spaces.Box(low=action_space_limits["low"], high=action_space_limits["high"], shape=(action_dimensionality,), dtype=np.float32)

            self.action_space_limits = action_space_limits

        ###################
        # Observation Space
        ###################

        # State observations
        if observation_type == ObservationType.STATE:
            # all objects -> num_objects * 3
            # current target object -> 3
            # the active agent -> 0 if single agent, 1 if multi agent
            # ptsd_state -> 4 if single agent, 8 if multi agent
            # pose -> 7 if single agent, 14 if multi agent

            observations_size = self.num_objects * 3 + 3 + (1 - self.single_agent) + 4 * (1 + (not self.single_agent)) + 7 * (1 + (not self.single_agent))
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observations_size,), dtype=np.float32)

        # Image observations
        elif observation_type == ObservationType.RGB:
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (3,), dtype=np.uint8)

        # RGB + Depth observations
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

        # Infos from the reward
        self.reward_info = {}
        self.reward_features = {}

        # All parameters of the reward function that are not passed default to 0.0
        self.reward_amount_dict = defaultdict(float) | reward_amount_dict

        # Callback functions called on reset
        self.on_reset_callbacks = on_reset_callbacks if on_reset_callbacks is not None else []

    def _init_sim(self) -> None:
        super()._init_sim()

        self.left_cauter: PivotizedCauter = self.scene_creation_result["left_cauter"]
        self.right_cauter: PivotizedCauter = self.scene_creation_result["right_cauter"]
        self.posts: List[Post] = self.scene_creation_result["posts"]
        self.camera: PhysicalCamera = self.scene_creation_result["physical_camera"]
        self.sample_positions_func: Callable = self.scene_creation_result["sample_positions_func"]
        self.contact_listener: Dict[str, self.sofa_core.ContactListener] = self.scene_creation_result["contact_listener"]

        # Factor for normalizing the distances in the reward function.
        # Based on the workspace of the cauter.
        self._distance_normalization_factor = 1.0 / np.linalg.norm(self.right_cauter.cartesian_workspace["high"] - self.right_cauter.cartesian_workspace["low"])

    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        """Step function of the environment that applies the action to the simulation and returns observation, reward, terminated (done) and truncated signals, and info."""

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
        in ``maximum_state_velocity`` in [angle, angle, angle, mm] / step.
        and further to per second (because delta T is not 1 second).
        """
        return self.time_step * self.maximum_state_velocity * action

    def _scale_discrete_action(self, action: int) -> np.ndarray:
        """Maps action indices to a motion delta."""
        return self._discrete_action_lookup[action]

    def _do_action(self, action) -> None:
        """Only defined to satisfy ABC."""
        pass

    def _do_action_individual(self, action: Dict[str, np.ndarray]) -> None:
        """Apply action to the simulation."""
        if self.action_type == ActionType.POSITION:
            self.left_cauter.set_state(action["left_cauter"])
            self.right_cauter.set_state(action["right_cauter"])
        else:
            self.left_cauter.set_state(self.left_cauter.get_state() + self._scale_action(action["left_cauter"]))
            self.right_cauter.set_state(self.right_cauter.get_state() + self._scale_action(action["right_cauter"]))

    def _do_action_multi(self, action: np.ndarray) -> None:
        """Apply action to the simulation."""
        if self.action_type == ActionType.POSITION:
            self.left_cauter.set_state(action[:4])
            self.right_cauter.set_state(action[4:])
        else:
            self.left_cauter.set_state(self.left_cauter.get_state() + self._scale_action(action[:4]))
            self.right_cauter.set_state(self.right_cauter.get_state() + self._scale_action(action[4:]))

    def _do_action_single(self, action: np.ndarray) -> None:
        """Apply action to the simulation."""
        if self.action_type == ActionType.POSITION:
            self.right_cauter.set_state(action)
        else:
            self.right_cauter.set_state(self.right_cauter.get_state() + self._scale_action(action))

    def _get_reward_features(self, previous_reward_features: dict) -> dict:
        """Get the features that may be used to assemble the reward function

        Features:
            - action_violated_cartesian_workspace (int): Number of tools that violate their Cartesian workspace
            - action_violated_state_limits (int): Number of tools that violate their state limits
            - rcm_violation_xyz (float): Deviation from planned to actual positions of the instruments.
            - rcm_violation_rotation (float): Deviation from planned to actual orientations of the instruments.
            - tool_collision (bool): Whether the tools collide
            - distance_to_active_sphere (float): Distance from the cauter tip to the active sphere
            - delta_distance_to_active_sphere (float): Change in distance from the cauter tip to the active sphere
            - deflection_of_inactive_spheres (float): Sum of the deflections of the inactive spheres
            - deflection_of_active_sphere (float): Deflection of the active sphere
            - delta_deflection_of_active_sphere (float): Change in deflection of the active sphere
            - done_with_active_sphere (bool): Whether the active sphere is deflected enough to be considered done
        """

        reward_features = {}

        active_post = self.posts[self.active_post_index]
        position_active_sphere = active_post.get_sphere_position()

        right_cauter_position = self.right_cauter.get_cutting_center_position()
        right_rcm_position = self.right_cauter.remote_center_of_motion[:3]

        # Did the action violate state or workspace limits?
        reward_features["action_violated_cartesian_workspace"] = int(self.right_cauter.last_set_state_violated_workspace_limits)
        reward_features["action_violated_state_limits"] = int(self.right_cauter.last_set_state_violated_state_limits)

        right_rcm_difference = self.right_cauter.get_pose_difference(position_norm=True)
        if not self.single_agent:
            left_rcm_difference = self.left_cauter.get_pose_difference(position_norm=True)
        else:
            left_rcm_difference = np.zeros_like(right_rcm_difference)
        reward_features["rcm_violation_xyz"] = (left_rcm_difference[0] + right_rcm_difference[0]) * self._distance_normalization_factor
        reward_features["rcm_violation_rotation"] = left_rcm_difference[1] + right_rcm_difference[1]

        if not self.single_agent:
            left_cauter_position = self.left_cauter.get_cutting_center_position()
            left_rcm_position = self.left_cauter.remote_center_of_motion[:3]

            # Did the action violate state or workspace limits?
            reward_features["action_violated_cartesian_workspace"] += int(self.left_cauter.last_set_state_violated_workspace_limits)
            reward_features["action_violated_state_limits"] += int(self.left_cauter.last_set_state_violated_state_limits)

            # Do the two cauters collide?
            reward_features["tool_collision"] = distance_between_line_segments(right_cauter_position, right_rcm_position, left_cauter_position, left_rcm_position, clamp_segments=True)[-1] < 1.5
        else:
            left_cauter_position = None
            reward_features["tool_collision"] = False

        if active_post.state == State.ACTIVE_LEFT:
            active_position = left_cauter_position
            # NOTE: or is only valid, because allow_deflection_with_instrument_shaft is always False for the multi-agent case
            valid_deflection = self.contact_listener["left_cauter"][self.active_post_index].getNumberOfContacts() > 0 or self.allow_deflection_with_instrument_shaft
        else:
            active_position = right_cauter_position
            # NOTE: or is only valid, because allow_deflection_with_instrument_shaft is always False for the multi-agent case
            valid_deflection = self.contact_listener["right_cauter"][self.active_post_index].getNumberOfContacts() > 0 or self.allow_deflection_with_instrument_shaft

        reward_features["distance_to_active_sphere"] = np.linalg.norm(active_position - position_active_sphere)
        reward_features["delta_distance_to_active_sphere"] = reward_features["distance_to_active_sphere"] - previous_reward_features["distance_to_active_sphere"]

        reward_features["deflection_of_inactive_spheres"] = sum([post.get_deflection() / self.min_deflection_distance for index, post in enumerate(self.posts) if index not in [self.active_post_index, self.previously_active_index]])

        reward_features["deflection_of_active_sphere"] = valid_deflection * active_post.get_deflection() / self.min_deflection_distance
        reward_features["delta_deflection_of_active_sphere"] = reward_features["deflection_of_active_sphere"] - previous_reward_features["deflection_of_active_sphere"]

        reward_features["done_with_active_sphere"] = valid_deflection * (active_post.get_deflection() >= self.min_deflection_distance)

        return reward_features

    def _get_reward(self) -> float:
        """Retrieve the reward features and scale them with the ``reward_amount_dict``."""
        reward = 0.0
        self.reward_info = {}

        reward_features = self._get_reward_features(self.reward_features)

        if reward_features["done_with_active_sphere"]:
            # Increase couter for current number of performed deflections
            self.num_deflections += 1

            # Set state to DONE or IDLE, depending on whether the env's mode is with or without replacement
            if self.mode == Mode.WITHOUT_REPLACEMENT:
                self.valid_indices.remove(self.active_post_index)
                self.posts[self.active_post_index].set_state(State.DONE)
            else:
                self.posts[self.active_post_index].set_state(State.IDLE)

        if self.num_deflections >= self.num_deflect_to_win:
            reward_features["successful_task"] = True
        else:
            reward_features["successful_task"] = False

        if reward_features["done_with_active_sphere"] and not reward_features["successful_task"]:
            # Chose a new active post
            self.previously_active_index = self.active_post_index
            self.active_post_index = self.rng.choice(self.valid_indices)
            if self.single_agent:
                self.posts[self.active_post_index].set_state(State.ACTIVE_RIGHT)
            else:
                # randomly choose which agent is active
                if self.rng.integers(0, 2) == 0:
                    self.posts[self.active_post_index].set_state(State.ACTIVE_LEFT)
                else:
                    self.posts[self.active_post_index].set_state(State.ACTIVE_RIGHT)

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

        # If there was a successful deflection, recalculate some values for previous_reward_features and the new active post
        if reward_features["done_with_active_sphere"]:
            active_post = self.posts[self.active_post_index]
            position_active_sphere = active_post.get_sphere_position()
            right_cauter_position = self.right_cauter.get_cutting_center_position()
            if not self.single_agent:
                left_cauter_position = self.right_cauter.get_cutting_center_position()
            else:
                left_cauter_position = None

            if active_post.state == State.ACTIVE_LEFT:
                active_position = left_cauter_position
            else:
                active_position = right_cauter_position

            self.reward_features["distance_to_active_sphere"] = np.linalg.norm(active_position - position_active_sphere)
            self.reward_features["deflection_of_active_sphere"] = active_post.get_deflection() / self.min_deflection_distance

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
            state_dict["sphere_positions"] = np.asarray([post.get_sphere_position() for post in self.posts]).ravel()
            state_dict["active_position"] = self.posts[self.active_post_index].get_sphere_position()
            state_dict["right_cauter_ptsd"] = self.right_cauter.get_state()
            state_dict["right_cauter_pose"] = self.right_cauter.get_pose()
            if not self.single_agent:
                state_dict["active_agent"] = np.asarray(1 if self.posts[self.active_post_index].state == State.ACTIVE_LEFT else 0)[None]  # 1 -> [1]
                state_dict["left_cauter_ptsd"] = self.left_cauter.get_state()
                state_dict["left_cauter_pose"] = self.left_cauter.get_pose()

            observation = np.concatenate(tuple(state_dict.values()), dtype=self.observation_space.dtype)

        return observation

    def _get_info(self) -> dict:
        """Assemble the info dictionary."""

        self.info = {}
        self.info["num_deflections"] = self.num_deflections

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
            seeds = self.seed_sequence.spawn(1 if self.single_agent else 2)
            self.right_cauter.seed(seed=seeds[0])
            if not self.single_agent:
                self.left_cauter.seed(seed=seeds[1])
            self.unconsumed_seed = False

        # Change the height and position of the spheres
        positions = self.sample_positions_func(self.rng)
        for index, position in enumerate(positions):
            self.posts[index].set_position(position[:2], height=position[2])

        # Chose an active post and reset the cauter(s)
        self.valid_indices = list(range(self.num_objects))
        self.active_post_index = self.rng.choice(self.valid_indices)
        self.previously_active_index = None
        self.num_deflections = 0

        for post in self.posts:
            post.set_state(State.IDLE)

        self.right_cauter.reset_cauter()
        if self.single_agent:
            self.posts[self.active_post_index].set_state(State.ACTIVE_RIGHT)
        else:
            # randomly choose which agent is active
            self.left_cauter.reset_cauter()
            if self.rng.integers(0, 2) == 0:
                self.posts[self.active_post_index].set_state(State.ACTIVE_LEFT)
            else:
                self.posts[self.active_post_index].set_state(State.ACTIVE_RIGHT)

        # Clear the episode info values
        for key in self.episode_info:
            self.episode_info[key] = 0.0

        # and the reward info dict
        self.reward_info = {}

        # and fill the first values used as previous_reward_features
        self.reward_features = {}

        active_post = self.posts[self.active_post_index]
        position_active_sphere = active_post.get_sphere_position()
        right_cauter_position = self.right_cauter.get_cutting_center_position()
        if not self.single_agent:
            left_cauter_position = self.right_cauter.get_cutting_center_position()
        else:
            left_cauter_position = None

        if active_post.state == State.ACTIVE_LEFT:
            active_position = left_cauter_position
            # NOTE: or is only valid, because allow_deflection_with_instrument_shaft is always False for the multi-agent case
            valid_deflection = self.contact_listener["left_cauter"][self.active_post_index].getNumberOfContacts() > 0 or self.allow_deflection_with_instrument_shaft
        else:
            active_position = right_cauter_position
            # NOTE: or is only valid, because allow_deflection_with_instrument_shaft is always False for the multi-agent case
            valid_deflection = self.contact_listener["right_cauter"][self.active_post_index].getNumberOfContacts() > 0 or self.allow_deflection_with_instrument_shaft

        self.reward_features["distance_to_active_sphere"] = np.linalg.norm(active_position - position_active_sphere)
        self.reward_features["deflection_of_active_sphere"] = valid_deflection * active_post.get_deflection() / self.min_deflection_distance

        # Execute any callbacks passed to the env.
        for callback in self.on_reset_callbacks:
            callback(self)

        # Animate several timesteps without actions until simulation settles
        for _ in range(self._settle_steps):
            self.sofa_simulation.animate(self._sofa_root_node, self.time_step)
            self._maybe_update_rgb_buffer()

        return self._get_observation(maybe_rgb_observation=self._maybe_update_rgb_buffer()), {}


if __name__ == "__main__":
    import pprint
    import time

    pp = pprint.PrettyPrinter()

    env = DeflectSpheresEnv(
        observation_type=ObservationType.STATE,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.VELOCITY,
        image_shape=(124, 124),
        frame_skip=1,
        time_step=0.1,
        settle_steps=10,
        single_agent=False,
        individual_agents=True,
        # discrete_action_magnitude=np.array([15.0, 15.0, 15.0, 10.0, 15.0, 15.0, 15.0, 10.0]),
    )

    env.reset(seed=42)
    done = False

    fps_list = deque(maxlen=100)
    counter = 0
    while not done:
        for _ in range(100):
            start = time.perf_counter()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if counter % 300 == 0:
                env.reset(seed=42)
                counter = 0
            counter += 1
            end = time.perf_counter()
            fps = 1 / (end - start)
            fps_list.append(fps)
            # print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")

        env.reset(seed=42)
