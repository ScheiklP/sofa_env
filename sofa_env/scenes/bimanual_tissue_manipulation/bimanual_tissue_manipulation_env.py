import gymnasium.spaces as spaces
import numpy as np

from collections import defaultdict, deque
from enum import Enum, unique
from functools import reduce
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Callable, List, Any

from sofa_env.scenes.bimanual_tissue_manipulation.sofa_objects.gripper import PivotizedGripper
from sofa_env.scenes.bimanual_tissue_manipulation.sofa_objects.tissue import Color, Tissue

from sofa_env.base import SofaEnv, RenderMode, RenderFramework
from sofa_env.sofa_templates.camera import Camera

from sofa_env.utils.camera import world_to_pixel_coordinates

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


class BimanualTissueManipulationEnv(SofaEnv):
    """Bimanual Tissue Manipulation Environment.

    Args:
        time_step (float): size of simulation time step in seconds (default: 0.01).
        frame_skip (int): number of simulation time steps taken (call ``_do_action`` and advance simulation) each time step is called (default: 1).
        scene_path (Union[str, Path]): Path to the scene description script that contains this environment's ``createScene`` function.
        render_mode (RenderMode): Create a window (``RenderMode.HUMAN``), run headless (``RenderMode.HEADLESS``), or do not create a render buffer at all (``RenderMode.NONE``).
        render_framework (RenderFramework): choose between pyglet and pygame for rendering
        create_scene_kwargs (Optional[dict]): A dictionary to pass additional keyword arguments to the ``createScene`` function.
        on_reset_callbacks (Optional[List[Callable]]): Functions that are called as the last step of the ``env.reset()`` function.
        maximum_state_velocity (Union[np.ndarray, float]): Velocity in deg/s for pts and mm/s for d in state space which are applied with a normalized action of value 1.
        discrete_action_magnitude (Union[np.ndarray, float]): Discrete change in state space in deg/s for pts and mm/s for d.
        image_shape (Tuple[int, int]): Height and Width of the rendered images.
        observation_type (ObservationType): Whether to return RGB images or an array of states as the observation.
        action_type (ActionType): Discrete or continuous actions to define the action space of the environment.
        settle_steps (int): How many steps to simulate without returning an observation after resetting the environment.
        reward_amount_dict (dict): Dictionary to weigh the components of the reward function.
        individual_agents (bool): Whether the instruments are controlled individually, or the action is one large array.
        with_collision (bool): Whether to check for collisions.
        with_tissue_forces (bool): Whether to calculate the tissue forces for the reward.
        target_radius (Optional[float]): Size of the target spheres.

    TODO:
        - for OGL rendering of the markers, we can set an arbitrary color. Make it possible to set many markers and targets.
        markers and targets should have the same colors.
    """

    def __init__(
        self,
        time_step: float = 0.01,
        frame_skip: int = 1,
        scene_path: Union[str, Path] = SCENE_DESCRIPTION_FILE_PATH,
        render_mode: RenderMode = RenderMode.HUMAN,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        create_scene_kwargs: Optional[dict] = None,
        on_reset_callbacks: Optional[List[Callable]] = None,
        maximum_state_velocity: Union[np.ndarray, float] = 25.0,
        discrete_action_magnitude: Union[np.ndarray, float] = 40.0,
        image_shape: Tuple[int, int] = (124, 124),
        observation_type: ObservationType = ObservationType.RGB,
        action_type: ActionType = ActionType.CONTINUOUS,
        settle_steps: int = 50,
        reward_amount_dict={
            "sum_distance_markers_to_target": -0.0,
            "sum_delta_distance_markers_to_target": -0.0,
            "sum_markers_at_target": 0.0,
            "left_gripper_workspace_violation": -0.0,
            "right_gripper_workspace_violation": -0.0,
            "left_gripper_state_limit_violation": -0.0,
            "right_gripper_state_limit_violation": -0.0,
            "left_gripper_collsion": -0.0,
            "right_gripper_collsion": -0.0,
            "force_on_tissue": -0.0,
            "successful_task": 0.0,
        },
        individual_agents: bool = False,
        with_collision: bool = True,
        with_tissue_forces: bool = True,
        marker_radius: float = 3.0,
        target_radius: Optional[float] = None,
        show_targets: bool = True,
        render_markers_with_ogl: bool = False,
        num_targets: int = 2,
        random_marker_range: List[Dict[str, Tuple[float, float]]] = [
            {"low": (0.1, 0.3), "high": (0.3, 0.75)},
            {"low": (0.7, 0.3), "high": (0.9, 0.75)},
        ],
        randomize_marker: bool = False,
        random_target_range: List[Dict[str, Tuple[float, float]]] = [
            {"low": (0.0, 0.6), "high": (0.2, 1.2)},
            {"low": (0.8, 0.6), "high": (1.0, 1.2)},
        ],
        randomize_target: bool = True,
    ) -> None:
        # Pass arguments to create_scene function
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape
        create_scene_kwargs["debug_rendering"] = False
        create_scene_kwargs["check_collision"] = with_collision
        self.image_shape = np.array(image_shape)

        # Rendering of marker and target
        if target_radius is None:
            target_radius = marker_radius
        self.marker_radius = marker_radius
        self.target_radius = target_radius
        self.show_targets = show_targets
        self.render_markers_with_ogl = render_markers_with_ogl

        # Positions of markers and targets
        if num_targets != 2:
            raise NotImplementedError("Only two targets are supported at the moment.")
        self.num_targets = num_targets
        self.target_positions = None

        self.randomize_target = randomize_target
        self.random_target_range = random_target_range
        if randomize_target:
            if not len(random_target_range) == num_targets:
                raise ValueError(f"Please pass a range for each target. Received {random_target_range=} with lenght {len(random_target_range)}.")

        self.randomize_marker = randomize_marker
        self.random_marker_range = random_marker_range
        if randomize_marker:
            if not len(random_marker_range) == num_targets:
                raise ValueError(f"Please pass a range for each marker. Received {random_marker_range=} with lenght {len(random_marker_range)}.")

        self.with_collision = with_collision
        self.with_tissue_forces = with_tissue_forces

        super().__init__(
            scene_path,
            time_step=time_step,
            frame_skip=frame_skip,
            render_mode=render_mode,
            render_framework=render_framework,
            create_scene_kwargs=create_scene_kwargs,
        )

        self._settle_steps = settle_steps

        self.individual_agents = individual_agents

        ##############
        # Action Space
        ##############
        action_dimensionality = 2 if self.individual_agents else 4
        self.action_type = action_type

        if action_type == ActionType.CONTINUOUS:
            if isinstance(maximum_state_velocity, np.ndarray):
                if not len(maximum_state_velocity) == 2:
                    raise ValueError(f"If you want to use per DoF maximum state velocities, please pass an array of length 2 as maximum_state_velocity. Received {maximum_state_velocity=} with lenght {len(maximum_state_velocity)}.")
            elif not isinstance(maximum_state_velocity, float):
                raise ValueError(f"Please pass either a float or an array as maximum_state_velocity. Received {maximum_state_velocity=}.")

            self.maximum_velocity = maximum_state_velocity
            self._scale_action = self._scale_continuous_action

            if self.individual_agents:
                self._do_action = self._do_action_individual_continuous
                self.action_space = spaces.Dict(
                    {
                        "left_gripper": spaces.Box(low=-1.0, high=1.0, shape=(action_dimensionality,), dtype=np.float32),
                        "right_gripper": spaces.Box(low=-1.0, high=1.0, shape=(action_dimensionality,), dtype=np.float32),
                    }
                )
            else:
                self._do_action = self._do_action_multi_continuous
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dimensionality,), dtype=np.float32)

        if action_type == ActionType.DISCRETE:
            if isinstance(discrete_action_magnitude, np.ndarray):
                if not len(discrete_action_magnitude) == 2:
                    raise ValueError(f"If you want to use per DoF discrete action magnitudes, please pass an array of length 2 as discrete_action_magnitude. Received {discrete_action_magnitude=} with lenght {len(discrete_action_magnitude)}.")

            self._scale_action = self._scale_discrete_action

            if self.individual_agents:
                self._do_action = self._do_action_individual_discrete
                self.action_space = spaces.Dict(
                    {
                        "left_gripper": spaces.Discrete(action_dimensionality * 2 + 1),
                        "right_gripper": spaces.Discrete(action_dimensionality * 2 + 1),
                    }
                )
            else:
                self._do_action = self._do_action_multi_discrete
                self.action_space = spaces.Discrete(action_dimensionality * 2 + 1)

            self._discrete_action_lookup = self.create_discrete_action_lookup(action_dimensionality, discrete_action_magnitude)
            self._discrete_action_lookup.flags.writeable = False

        else:
            if isinstance(maximum_state_velocity, np.ndarray):
                if not len(maximum_state_velocity) == 2:
                    raise ValueError(f"If you want to use per DoF maximum state velocities, please pass an array of length 2 as maximum_state_velocity. Received {maximum_state_velocity=} with lenght {len(maximum_state_velocity)}.")
            elif not isinstance(maximum_state_velocity, float):
                raise ValueError(f"Please pass either a float or an array as maximum_state_velocity. Received {maximum_state_velocity=}.")

            if action_type == ActionType.CONTINUOUS:
                action_space_limits = {
                    "low": -np.ones(2, dtype=np.float32),
                    "high": np.ones(2, dtype=np.float32),
                }

                # Scale 1.0 to the maximum velocity
                self.maximum_state_velocity = maximum_state_velocity
                self._scale_action = self._scale_continuous_action
            elif action_type == ActionType.VELOCITY:
                action_space_limits = {
                    "low": -maximum_state_velocity,
                    "high": maximum_state_velocity,
                }
                # Do not scale the velocity, as it is already scaled
                self.maximum_state_velocity = 1.0
                self._scale_action = self._scale_continuous_action

            elif action_type == ActionType.POSITION:
                # Same as the state limits of the instruments
                action_space_limits = {
                    "low": np.array([-45.0, -156.0]),
                    "high": np.array([45.0, 0.0]),
                }

            if self.individual_agents:
                self._do_action = self._do_action_individual_continuous
                self.action_space = spaces.Dict(
                    {
                        "left_gripper": spaces.Box(low=action_space_limits["low"], high=action_space_limits["high"], shape=(action_dimensionality,), dtype=np.float32),
                        "right_gripper": spaces.Box(low=action_space_limits["low"], high=action_space_limits["high"], shape=(action_dimensionality,), dtype=np.float32),
                    }
                )
            else:
                self._do_action = self._do_action_multi_continuous
                if isinstance(action_space_limits["low"], np.ndarray):
                    action_space_limits["low"] = np.concatenate((action_space_limits["low"], action_space_limits["low"]))
                    action_space_limits["high"] = np.concatenate((action_space_limits["high"], action_space_limits["high"]))
                self.action_space = spaces.Box(low=action_space_limits["low"], high=action_space_limits["high"], shape=(action_dimensionality,), dtype=np.float32)

            self.action_space_limits = action_space_limits

        ###################
        # Observation Space
        ###################
        # State observations
        if observation_type == ObservationType.STATE:
            # pd_state left_gripper -> 2
            # pd_state right_gripper -> 2
            # targets -> num_targets * 2
            # markers -> num_targets * 2
            observations_size = 2 + 2 + num_targets * 2 + num_targets * 2
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observations_size,), dtype=np.float32)
            self._get_observation = self._get_state_observation

        # Image observations
        elif observation_type == ObservationType.RGB:
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (3,), dtype=np.uint8)
            self._get_observation = self._get_rgb_observation

        # RGB + Depth observations
        elif observation_type == ObservationType.RGBD:
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (4,), dtype=np.uint8)
            self._get_observation = self._get_rgbd_observation

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

        # Split the reward parts for the individual agents
        self.agent_specific_rewards = defaultdict(float)

        # Callback functions called on reset
        self.on_reset_callbacks = on_reset_callbacks if on_reset_callbacks is not None else []

    def _init_sim(self) -> None:
        super()._init_sim()

        self.left_gripper: PivotizedGripper = self.scene_creation_result["left_gripper"]
        self.right_gripper: PivotizedGripper = self.scene_creation_result["right_gripper"]
        self.camera: Camera = self.scene_creation_result["camera"]
        self.tissue: Tissue = self.scene_creation_result["tissue"]
        self.left_contact_listener = self.scene_creation_result["left_contact_listener"]
        self.right_contact_listener = self.scene_creation_result["right_contact_listener"]

        if self.action_type == ActionType.POSITION:
            for key in ("high", "low"):
                gripper_limits = np.concatenate([self.left_gripper.state_limits[key][[0, 3]], self.right_gripper.state_limits[key][[0, 3]]])
                env_limits = self.action_space_limits[key]
                if not np.allclose(gripper_limits, env_limits):
                    raise ValueError(f"Please set the action space limits to the same values as the gripper state limits. Received {self.action_space_limits=}, but the gripper state limits are {gripper_limits=}.")

        # Factor for normalizing the distances in the reward function.
        # Based on the size of the tissue.
        self._distance_normalization_factor = 1.0 / np.linalg.norm(self.tissue.grid_size)

        self.force_normalization_factor = 1.0 / (5000.0 * self.tissue.grid_resolution)  # Lin. reg ≈ 2000 * grid_resolution - 7000 (Coefficient of determination: 0.9947)

    def _update_sofa_visuals(self) -> None:
        super()._update_sofa_visuals()

        # TODO make more generic
        colors = [(0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        # discretization of the circle
        side_num = 20

        # Render markers on the tissue
        if self.render_markers_with_ogl:
            marker_positions = self.tissue.get_marker_positions()

            if marker_positions is not None:
                # We disable the depth test to make sure the targets are always visible.
                self.opengl_gl.glPushAttrib(self.opengl_gl.GL_DEPTH_BUFFER_BIT)  # save current depth buffer state
                self.opengl_gl.glDisable(self.opengl_gl.GL_DEPTH_TEST)  # disable depth test

                for index, position in enumerate(marker_positions):
                    self.opengl_gl.glBegin(self.opengl_gl.GL_POLYGON)
                    self.opengl_gl.glColor3f(*colors[index])
                    for vertex in range(0, side_num):
                        angle = float(vertex) * 2.0 * np.pi / side_num
                        relative_position = np.array([np.cos(angle) * self.marker_radius, np.sin(angle) * self.marker_radius, 0.0])
                        absolute_position = position + relative_position
                        self.opengl_gl.glVertex3f(*absolute_position)
                    self.opengl_gl.glEnd()

                self.opengl_gl.glColor3f(1.0, 1.0, 1.0)  # reset color to white
                self.opengl_gl.glPopAttrib()  # restore previous depth buffer state

        # Render targets in the image
        if self.show_targets:
            if self.target_positions is None:
                return

            # We disable the depth test to make sure the targets are always visible.
            self.opengl_gl.glPushAttrib(self.opengl_gl.GL_DEPTH_BUFFER_BIT)  # save current depth buffer state
            self.opengl_gl.glDisable(self.opengl_gl.GL_DEPTH_TEST)  # disable depth test
            self.opengl_gl.glDisable(self.opengl_gl.GL_LIGHTING)  # disable lighting

            for index, position in enumerate(self.target_positions):
                self.opengl_gl.glBegin(self.opengl_gl.GL_POLYGON)
                self.opengl_gl.glColor3f(*colors[index])
                for vertex in range(0, side_num):
                    angle = float(vertex) * 2.0 * np.pi / side_num
                    relative_position = np.array([np.cos(angle) * self.target_radius, np.sin(angle) * self.target_radius, 0.0])
                    absolute_position = position + relative_position
                    self.opengl_gl.glVertex3f(*absolute_position)
                self.opengl_gl.glEnd()

            self.opengl_gl.glColor3f(1.0, 1.0, 1.0)  # reset color to white
            self.opengl_gl.glPopAttrib()  # restore previous depth buffer state

    def create_discrete_action_lookup(self, dimensionality, discrete_action_magnitude):
        """Returns a list of discrete actions.

        Looks like: [step, 0, 0, ...], [-step, 0, 0, ...], [0, step, 0, ...], [0, -step, 0, ...].

        Args:
            dimensionality (int): Action dimensionality.
            discrete_action_magnitude (np.array): Array with the magnitude of each action.
        """
        action_list = []
        for i in range(dimensionality * 2):
            action = [0.0] * dimensionality
            step_size = discrete_action_magnitude if isinstance(discrete_action_magnitude, float) else discrete_action_magnitude[int(i / 2)]
            action[int(i / 2)] = (1 - 2 * (i % 2)) * step_size
            action_list.append(action)

        # Noop action
        action_list.append([0.0] * dimensionality)

        action_lookup = np.array(action_list)
        action_lookup *= self.time_step

        return action_lookup

    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        """Step function of the environment that applies the action to the simulation and returns observation, reward, done signal, and info."""
        maybe_rgb_observation = super().step(action)

        observation = self._get_observation(maybe_rgb_observation)
        reward = self._get_reward()
        terminated = self._get_done()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _scale_continuous_action(self, action: np.ndarray) -> np.ndarray:
        """
        Policy output is clipped to [-1, 1] e.g. [-0.3, 0.8, 1].
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

    def _do_action_individual_continuous(self, action: Dict[str, np.ndarray]) -> None:
        """Apply action to the simulation."""

        if self.action_type == ActionType.POSITION:
            self.left_gripper.do_action(action["left_gripper"], absolute=True)  # type: ignore
            self.right_gripper.do_action(action["right_gripper"], absolute=True)  # type: ignore
        else:
            self.left_gripper.do_action(self._scale_action(action["left_gripper"]))  # type: ignore
            self.right_gripper.do_action(self._scale_action(action["right_gripper"]))  # type: ignore

    def _do_action_individual_discrete(self, action: Dict[str, int]) -> None:
        """Apply action to the simulation."""
        self.left_gripper.do_action(self._scale_action(action["left_gripper"]))  # type: ignore
        self.right_gripper.do_action(self._scale_action(action["right_gripper"]))  # type: ignore

    def _do_action_multi_discrete(self, action: int) -> None:
        """Apply action to the simulation."""
        if action < 4:
            self.left_gripper.do_action(self._scale_action(action)[:2])  # type: ignore
        else:
            self.right_gripper.do_action(self._scale_action(action - 2)[2:])  # type: ignore

    def _do_action_multi_continuous(self, action: np.ndarray) -> None:
        """Apply action to the simulation."""
        if self.action_type == ActionType.POSITION:
            self.left_gripper.do_action(action[:2], absolute=True)  # type: ignore
            self.right_gripper.do_action(action[2:], absolute=True)  # type: ignore
        else:
            self.left_gripper.do_action(self._scale_action(action)[:2])  # type: ignore
            self.right_gripper.do_action(self._scale_action(action)[2:])  # type: ignore

    def _get_reward_features(self, previous_reward_features: dict) -> dict:
        """Get the features that may be used to assemble the reward function

        Features:
            - distance_markers_to_target: Distances between the markers and the targets.
            - delta_distance_markers_to_target: Change of distance between the markers and the targets.
            - marker_at_target: Whether the marker point is inside the target sphere.
            - left_gripper_workspace_violation: Whether the left gripper tried to violate its workspace.
            - right_gripper_workspace_violation: Whether the right gripper tried to violate its workspace.
            - left_gripper_state_limit_violation: Whether the left gripper tried to violate its state limits.
            - right_gripper_state_limit_violation: Whether the right gripper tried to violate its state limits.
            - left_gripper_collision: Whether the left gripper shaft is in collsion with the tissue.
            - right_gripper_collision: Whether the right gripper shaft is in collsion with the tissue.
            - force_on_tissue: Relativ force on the tissue.
            - successful_task: Successful, if all markers are inside their targets.
        """

        reward_features = {}

        target_positions = self.target_positions
        marker_positions = self.tissue.get_marker_positions()
        assert marker_positions is not None

        reward_features["distances_markers_to_target"] = np.linalg.norm(marker_positions - target_positions, axis=1)
        reward_features["delta_distances_markers_to_target"] = reward_features["distances_markers_to_target"] - previous_reward_features["distances_markers_to_target"]

        reward_features["sum_distance_markers_to_target"] = np.sum(reward_features["distances_markers_to_target"])
        reward_features["sum_delta_distance_markers_to_target"] = np.sum(reward_features["delta_distances_markers_to_target"])

        reward_features["markers_at_target"] = reward_features["distances_markers_to_target"] < self.target_radius
        reward_features["sum_markers_at_target"] = np.sum(reward_features["markers_at_target"])

        reward_features["left_gripper_workspace_violation"] = int(self.left_gripper.last_set_state_violated_workspace_limits)
        reward_features["right_gripper_workspace_violation"] = int(self.right_gripper.last_set_state_violated_workspace_limits)

        reward_features["left_gripper_state_limit_violation"] = int(self.left_gripper.last_set_state_violated_state_limits)
        reward_features["right_gripper_state_limit_violation"] = int(self.right_gripper.last_set_state_violated_state_limits)

        reward_features["left_gripper_collision"] = self.left_contact_listener.getNumberOfContacts() if self.with_collision else False
        reward_features["right_gripper_collision"] = self.right_contact_listener.getNumberOfContacts() if self.with_collision else False

        reward_features["force_on_tissue"] = self.tissue.get_internal_force_magnitude() * self.force_normalization_factor if self.with_tissue_forces else 0.0
        reward_features["successful_task"] = np.all(reward_features["markers_at_target"])

        return reward_features

    def _get_reward(self) -> float:
        """Retrieve the reward features and scale them with the ``reward_amount_dict``."""
        reward = 0.0
        self.reward_info = {}

        reward_features = self._get_reward_features(self.reward_features)

        self.reward_features = reward_features.copy()

        array_keys = ["distances_markers_to_target", "delta_distances_markers_to_target", "markers_at_target"]

        for key, value in reward_features.items():
            if key in array_keys:
                continue
            if "distance" in key or "velocity" in key:
                if np.any(np.isnan(value)):
                    value[:] = 1.0 / self._distance_normalization_factor
                value = np.clip(value, -1.0 / self._distance_normalization_factor, 1.0 / self._distance_normalization_factor)
                value = self._distance_normalization_factor * value

            self.reward_info[f"reward_{key}"] = self.reward_amount_dict[key] * value

            # Add to reward
            reward += self.reward_info[f"reward_{key}"]

        self.reward_info["reward"] = reward

        return float(reward)

    def _get_done(self) -> bool:
        """Look up if the episode is finished."""
        return self.reward_features["successful_task"]

    def _get_info(self) -> dict:
        """Assemble the info dictionary."""

        for key, value in self.reward_info.items():
            # shortens 'reward_delta_gripper_distance_to_torus_tracking_points'
            # to 'ret_del_gri_dis_to_tor_tra_poi'
            words = key.split("_")[1:]
            shortened_key = reduce(lambda x, y: x + "_" + y[:3], words, "ret")
            self.episode_info[shortened_key] += value

        return {**self.reward_info, **self.episode_info, **self.reward_features, **self.agent_specific_rewards}

    def _get_observation(self, maybe_rgb_observation: Union[np.ndarray, None]) -> Union[np.ndarray, dict]:
        """Return the observation based on the ObservationType.

        Placeholder for the selected observation.
        """
        pass

    def _get_rgb_observation(self, maybe_rgb_observation: Union[np.ndarray, None]) -> Union[np.ndarray, dict]:
        """Returns the rgb observation."""
        return maybe_rgb_observation

    def _get_rgbd_observation(self, maybe_rgb_observation: Union[np.ndarray, None]) -> Union[np.ndarray, dict]:
        """Returns the rgbd observation."""
        observation = self.observation_space.sample()
        observation[:, :, :3] = maybe_rgb_observation
        observation[:, :, 3:] = self.get_depth()

        return observation

    def _get_state_observation(self, maybe_rgb_observation: Union[np.ndarray, None]) -> Union[np.ndarray, dict]:
        """Returns the state observation."""
        state_dict = {}
        state_dict["left_gripper_state"] = self.left_gripper.get_state()[[0, 3]]
        state_dict["right_gripper_state"] = self.right_gripper.get_state()[[0, 3]]
        state_dict["target_positions"] = self.target_positions[:, :2].ravel()
        state_dict["marker_positions"] = self.tissue.get_marker_positions()[:, :2].ravel()

        return np.concatenate(tuple(state_dict.values()), dtype=self.observation_space.dtype)

    def get_gripper_state(self) -> np.ndarray:
        """Returns the gripper state."""
        return np.concatenate((self.left_gripper.get_state()[[0, 3]], self.right_gripper.get_state()[[0, 3]]))

    def get_target_positions_in_image(self) -> np.ndarray:
        """Returns the target positions in image coordinates.

        Returns:
            np.ndarray: The target positions in image coordinates with shape (n_targets, 2) between [0, 1].
        """
        pixel_coordinates = np.zeros((len(self.target_positions), 2), dtype=np.float32)
        for i, position in enumerate(self.target_positions):
            pixel_coordinates[i] = world_to_pixel_coordinates(position, self.camera.sofa_object)

        return pixel_coordinates / self.image_shape

    def get_marker_positions_in_image(self) -> np.ndarray:
        """Returns the marker positions in image coordinates.

        Returns:
            np.ndarray: The marker positions in image coordinates with shape (n_markers, 2) between [0, 1].
        """
        marker_positions = self.tissue.get_marker_positions()
        pixel_coordinates = np.zeros((len(marker_positions), 2), dtype=np.float32)
        for i, position in enumerate(marker_positions):
            pixel_coordinates[i] = world_to_pixel_coordinates(position, self.camera.sofa_object)

        return pixel_coordinates / self.image_shape

    def reset(self, seed: Union[int, np.random.SeedSequence, None] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Union[np.ndarray, None], Dict]:
        super().reset(seed=seed, options=options)

        # Seed the instruments
        if self.unconsumed_seed:
            seeds = self.seed_sequence.spawn(5)
            self.right_gripper.seed(seed=seeds[0])
            self.left_gripper.seed(seed=seeds[1])
            self.tissue.seed(seed=seeds[4])
            self.unconsumed_seed = False

        # Reset grippers
        self.left_gripper.reset_state()
        self.right_gripper.reset_state()

        # Reset the tissue
        marker_positions = []
        if self.randomize_marker:
            for marker_range in self.random_marker_range:
                marker_positions.append(self.rng.uniform(marker_range["low"], marker_range["high"]))
        else:
            # TODO generalize to N markers
            marker_positions = [(0.25, 0.5), (0.75, 0.5)]

        if self.render_markers_with_ogl:
            marker_colors = None
        else:
            # TODO generalize to N markers
            marker_colors = [Color.GREEN, Color.BLUE]

        self.tissue.set_marker_positions(marker_positions, marker_colors, marker_radii=self.marker_radius)

        # Image corners:
        # [-80, 130] # top left
        # [80, 130] # top right
        # [-80, -30] # bottom left
        # [80, -30] # bottom right

        # Reset targets
        if options is not None and "target_positions" in options:
            input_target_positions = np.array(options["target_positions"])
            input_target_positions = input_target_positions.reshape(-1, 2)
            target_positions = np.zeros((len(input_target_positions), 3))
            target_positions[:, :2] = input_target_positions
            self.target_positions = target_positions
        else:
            relative_target_positions = []
            if self.randomize_target:
                for target_range in self.random_target_range:
                    relative_target_positions.append(self.rng.uniform(target_range["low"], target_range["high"]))
            else:
                # TODO generalize to N targets
                relative_target_positions = [(-0.25, 1.5), (1.25, 1.5)]

            target_positions = np.zeros((len(relative_target_positions), 3))
            for i, position in enumerate(relative_target_positions):
                target_positions[i, :2] = self.tissue.relative_to_absolute_position(position)

            self.target_positions = target_positions

        # Clear the info values
        for key in self.episode_info:
            self.episode_info[key] = 0.0
        self.reward_info = {}

        # Calculate rewards needed for previous_reward_features
        self.reward_features = {}
        target_positions = self.target_positions
        marker_positions = self.tissue.get_marker_positions()
        assert marker_positions is not None

        self.reward_features["distances_markers_to_target"] = np.linalg.norm(marker_positions - target_positions, axis=1)

        # Execute any callbacks passed to the env.
        for callback in self.on_reset_callbacks:
            callback(self)

        # Animate several timesteps without actions until simulation settles
        for _ in range(self._settle_steps):
            self.sofa_simulation.animate(self._sofa_root_node, self._sofa_root_node.getDt())

        return self._get_observation(maybe_rgb_observation=self._maybe_update_rgb_buffer()), {}


if __name__ == "__main__":
    import time

    env = BimanualTissueManipulationEnv(
        render_mode=RenderMode.HUMAN,
        observation_type=ObservationType.STATE,
        image_shape=(512, 512),
        action_type=ActionType.VELOCITY,
        individual_agents=False,
        maximum_state_velocity=np.array([50.0, 10.0]),
        discrete_action_magnitude=40.0,
        with_collision=True,
    )

    env.reset()
    done = False
    counter = 0
    fps_list = deque(maxlen=100)

    while True:
        for _ in range(200):
            start = time.perf_counter()
            obs, done, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
            if counter == 200:
                env.reset()
                counter = 0
            counter += 1
            end = time.perf_counter()
            fps = 1 / (end - start)
            fps_list.append(fps)
            print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")

        env.reset()
