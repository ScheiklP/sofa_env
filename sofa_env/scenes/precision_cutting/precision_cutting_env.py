from typing import Dict, Union, Tuple, Optional, Any, List, Callable
from collections import defaultdict, deque
from enum import Enum, unique
from pathlib import Path
from functools import reduce

import numpy as np
import gymnasium.spaces as spaces

from sofa_env.base import SofaEnv, RenderMode, RenderFramework
from sofa_env.scenes.precision_cutting.sofa_objects.gripper import ArticulatedGripper
from sofa_env.sofa_templates.camera import Camera
from sofa_env.utils.math_helper import euler_angles_to_quaternion, quaternion_to_euler_angles

from sofa_env.scenes.precision_cutting.helper import farthest_point_sampling
from sofa_env.scenes.precision_cutting.sofa_objects.cloth import Cloth
from sofa_env.scenes.precision_cutting.sofa_objects.scissors import ArticulatedScissors


HERE = Path(__file__).resolve().parent
SCENE_DESCRIPTION_FILE_PATH = HERE / "scene_description.py"


@unique
class ObservationType(Enum):
    RGB = 0
    STATE = 1
    DEPTH = 2
    RGBD = 3


@unique
class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


class PrecisionCuttingEnv(SofaEnv):
    """Precision Cutting Environment

    The goal of this environment is to cut a cloth along a colored path with a pair of scissors.

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
        maximum_cartesian_velocity  (Union[np.ndarray, float]): Velocity in mm/s for d in Cartesian space which are applied with a normalized action of value 1.
        discrete_state_action_magnitude (Union[np.ndarray, float]): Discrete change in state space in deg/s for pts and mm/s for d.
        discrete_cartesian_action_magnitude (Union[np.ndarray, float]): Discrete change in Cartesian space in mm/s.
        create_scene_kwargs (Optional[dict]): A dictionary to pass additional keyword arguments to the ``createScene`` function.
        on_reset_callbacks (Optional[List[Callable]]): Functions that are called as the last step of the ``env.reset()`` function.
        camera_reset_noise (Optional[np.ndarray]): Optional noise to uniformly sample from that is added to the initial camera pose in xyz cartesian position and cartesian point to look at.
        num_tracking_points_on_cutting_path (int): Number of points on the cutting path to include in the state observation.
        num_tracking_points_off_cutting_path (int): Number of points off the cutting path to include in the state observation.
        randomize_scissors (bool): Whether to randomize the scissors' initial pose.
        ratio_to_cut (float): Ratio of the cutting path that needs to be cut to consider the task successful.
        cartesian_control (bool): Whether to use cartesian control for the scissors.
        cloth_cutting_path_func_generator (Callable): Function that returns a function that generates the cutting path for the cloth.
        with_gripper (bool): Whether to add a gripper to the scene.
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
        reward_amount_dict: Dict[str, float] = {
            "unstable_deformation": -0.0,
            "distance_scissors_cutting_path": -0.0,
            "delta_distance_scissors_cutting_path": -0.0,
            "cuts_on_path": 0.0,
            "cuts_off_path": -0.0,
            "cut_ratio": 0.0,
            "delta_cut_ratio": 0.0,
            "workspace_violation": -0.0,
            "state_limits_violation": -0.0,
            "rcm_violation_xyz": -0.0,
            "rcm_violation_rpy": -0.0,
            "jaw_angle_violation": -0.0,
            "successful_task": 0.0,
        },
        maximum_state_velocity: Union[np.ndarray, float] = np.array([20.0, 20.0, 20.0, 20.0, 20.0]),
        maximum_cartesian_velocity: Union[np.ndarray, float] = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]),
        discrete_state_action_magnitude: Union[np.ndarray, float] = np.array([15.0, 15.0, 15.0, 10.0, 5.0]),
        discrete_cartesian_action_magnitude: Union[np.ndarray, float] = np.array([15.0, 15.0, 15.0, 10.0, 10.0, 10.0, 5.0]),
        create_scene_kwargs: Optional[dict] = None,
        on_reset_callbacks: Optional[List[Callable]] = None,
        camera_reset_noise: Optional[np.ndarray] = None,
        num_tracking_points_on_cutting_path: int = 10,
        num_tracking_points_off_cutting_path: int = 10,
        randomize_scissors: bool = True,
        ratio_to_cut: float = 0.8,
        cartesian_control: bool = False,
        cloth_cutting_path_func_generator: Optional[Callable] = None,
        with_gripper: bool = False,
    ) -> None:
        # Pass image shape to the scene creation function.
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape
        create_scene_kwargs["with_gripper"] = with_gripper

        self.with_gripper = with_gripper
        if self.with_gripper:
            raise NotImplementedError("Controlling the gripper is not yet implemented in the env. Please implement adapting the action space, as well as scaling and performing the action.")

        # Whether the scissors initial state should be randomized at reset.
        self.randomize_scissors = randomize_scissors
        if not randomize_scissors:
            create_scene_kwargs["scissors_reset_noise"] = None

        # Whether to control the scissors in cartesian space or joint space
        self.cartesian_control = cartesian_control

        # How much of the triangles of the path should be removed, to consider the episode as done.
        self.ratio_to_cut = ratio_to_cut

        super().__init__(
            scene_path=scene_path,
            time_step=time_step,
            frame_skip=frame_skip,
            render_mode=render_mode,
            render_framework=render_framework,
            create_scene_kwargs=create_scene_kwargs,
        )

        # Replace the create_scene_kwargs["cloth_cutting_path_func"] from a function that returns a cutting path in init_sim.
        self.cloth_cutting_path_func_generator = cloth_cutting_path_func_generator

        # How many simulation steps to wait before starting the episode
        self._settle_steps = settle_steps
        self.maximum_velocity = maximum_cartesian_velocity if cartesian_control else maximum_state_velocity

        # Randomize camera pose on reset
        if camera_reset_noise is not None:
            if not isinstance(camera_reset_noise, np.ndarray) and not camera_reset_noise.shape == (6,):
                raise ValueError(
                    "Please pass the camera_reset_noise as a numpy array with 6 values for maximum deviation \
                        from the original camera pose in xyz cartesian position and cartesian point to look at."
                )
        self.camera_reset_noise = camera_reset_noise

        ##############
        # Action Space
        ##############
        action_dims = 7 if cartesian_control else ArticulatedScissors.get_action_dims()
        self._do_action = self._do_cartesian_action if cartesian_control else self._do_state_action
        self.action_type = action_type
        if action_type == ActionType.CONTINUOUS:
            self._scale_action = self._scale_continuous_action
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dims,), dtype=np.float32)
        else:
            discrete_action_magnitude = discrete_cartesian_action_magnitude if cartesian_control else discrete_state_action_magnitude
            self._scale_action = self._scale_discrete_action
            self.action_space = spaces.Discrete(2 * action_dims + 1)
            # [step, 0, 0, ...], [-step, 0, 0, ...], [0, step, 0, ...], [0, -step, 0, ...]
            if isinstance(discrete_action_magnitude, np.ndarray):
                steps, num_steps = discrete_action_magnitude, len(discrete_action_magnitude)
                if num_steps == action_dims:
                    action_list = np.stack([np.diag(steps), -np.diag(steps)], axis=1)
                elif num_steps == 2 * action_dims:
                    action_list = np.stack([np.diag(steps[:action_dims]), np.diag(steps[action_dims:])], axis=1)
                else:
                    raise ValueError(f"If you want to use individual discrete action step sizes per action dimension, please pass an array of length {action_dims} or {action_dims * 2} as discrete_action_magnitude. Received {steps=} with length {num_steps}.")
            else:
                steps = np.full(action_dims, discrete_action_magnitude)
                action_list = np.stack([np.diag(steps), -np.diag(steps)], axis=1)

            action_list = action_list.reshape(2 * action_dims, action_dims)
            # No-op action
            action_list = np.vstack([action_list, np.zeros(action_dims)])

            self._discrete_action_lookup = self.time_step * action_list
            self._discrete_action_lookup.flags.writeable = False

        ###################
        # Observation Space
        ###################
        # State observations
        self.num_tracking_points_on_cutting_path = num_tracking_points_on_cutting_path
        self.num_tracking_points_off_cutting_path = num_tracking_points_off_cutting_path
        if observation_type == ObservationType.STATE:
            observations_size = 0
            # scissors ptsd_state -> 4
            observations_size += 4
            # scissors jaw_angle -> 1
            observations_size += 1
            # scissors pose -> 7
            observations_size += 7
            # tracking points off cutting path -> 3 * num_tracking_points_off_cutting_path
            observations_size += 3 * num_tracking_points_off_cutting_path
            # tracking points on cutting path -> 3 * num_tracking_points_on_cutting_path
            observations_size += 3 * num_tracking_points_on_cutting_path
            # closest path point -> 3
            observations_size += 3
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observations_size,), dtype=np.float32)
        # Image observations
        elif observation_type == ObservationType.RGB:
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (3,), dtype=np.uint8)
        # RGB + Depth observations
        elif observation_type == ObservationType.RGBD:
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (4,), dtype=np.uint8)
        # Depthmap
        elif observation_type == ObservationType.DEPTH:
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (1,), dtype=np.uint8)
        else:
            raise Exception(f"Please set observation_type to a value of ObservationType. Received {observation_type}.")
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
        if self.cloth_cutting_path_func_generator is not None:
            self.create_scene_kwargs["cloth_cutting_path_func"] = self.cloth_cutting_path_func_generator(self.rng)

        super()._init_sim()

        self.cloth: Cloth = self.scene_creation_result["cloth"]
        self.scissors: ArticulatedScissors = self.scene_creation_result["scissors"]
        self.gripper: Union[ArticulatedGripper, None] = self.scene_creation_result["gripper"]
        self.camera: Camera = self.scene_creation_result["camera"]
        self.closest_path_point_mechanical_object = self.scene_creation_result["closest_path_point_mechanical_object"]

        # Factor for normalizing the distances in the reward function.
        # Based on the workspace of the scissors.
        self._distance_normalization_factor = 1.0 / np.linalg.norm(self.scissors.cartesian_workspace["high"] - self.scissors.cartesian_workspace["low"])

        # After initialization of the MechanicalObjects from the topologies, write the positions to the cloth's buffer.
        self.cloth.update_topology_buffer()

    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        """Step function of the environment that applies the action to the simulation and returns observation, reward, done signal, and info."""

        maybe_rgb_observation = super().step(action)

        # Calculate reward first, to update the tracking indices.
        reward = self._get_reward()

        observation = self._get_observation(maybe_rgb_observation=maybe_rgb_observation)
        terminated = self._get_done()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _scale_continuous_action(self, action: np.ndarray) -> np.ndarray:
        """
        Policy's output is clipped to [-1, 1] e.g. [-0.3, 0.8, 1].
        We want to scale that to the maximum velocities defined
        in ``maximume_velocity`` in [si unit] / step.
        and further to per second (because delta T is not 1 second).
        """
        return self.time_step * self.maximum_velocity * action

    def _scale_discrete_action(self, action: int) -> np.ndarray:
        """Maps action indices to a motion delta."""
        return self._discrete_action_lookup[action]

    def _do_action(self, action) -> None:
        raise RuntimeError("This method should redirect to either _do_continuous_action or _do_discrete_action.")

    def _scale_action(self, action: Union[np.ndarray, int]) -> np.ndarray:
        """Placeholder for action scaling."""
        raise RuntimeError("This method should redirect to either _scale_continuous_action or _scale_discrete_action.")

    def _do_state_action(self, action: Union[np.ndarray, int]) -> None:
        """Apply action to the simulation by adding deltas to the PTSDA state of the scissors."""
        self.scissors.set_articulated_state(self.scissors.get_articulated_state() + self._scale_action(action))

    def _do_cartesian_action(self, action: Union[np.ndarray, int]) -> None:
        """Apply action to the simulation by adding deltas to XYZ position, XYZ euler angles, and jaw opening."""
        scaled_action = self._scale_action(action)
        pose = self.scissors.get_pose().copy()
        pose[:3] += scaled_action[:3]
        euler_angles = quaternion_to_euler_angles(pose[3:])
        euler_angles += scaled_action[3:-1]
        pose[3:] = euler_angles_to_quaternion(euler_angles)
        self.scissors.set_pose(pose)
        self.scissors.set_angle(self.scissors.get_angle() + scaled_action[-1])

    def _get_reward_features(self, previous_reward_features: dict) -> dict:
        """Get the features that may be used to assemble the reward function

        Features:
            - unstable_deformation (bool): Whether the cloth is unstable (unstable simulation).
            - distance_scissors_cutting_path (float): Minimal distance between the scissors and the cutting path.
            - delta_distance_scissors_cutting_path (float): Change in distance between the scissors and the cutting path.
            - num_tris_on_path (int): Number of remaining triangles on the cutting path.
            - cut_on_path (int): Number of triangles on the path that were cut since the last step.
            - num_tris_off_path (int): Number of remaining triangles off the cutting path.
            - cut_off_path (int): Number of triangles off the path that were cut since the last step.
            - cut_ratio (float): Ratio of cut centerline path.
            - delta_cut_ratio (float): Change in ratio of cut centerline path.
            - workspace_violation (float): 1.0 if the scissors' action would have violated the workspace, 0.0 otherwise.
            - state_limits_violation (float): 1.0 if the scissors' action would have violated the state limits, 0.0 otherwise.
            - jaw_angle_violation (float): 1.0 if the scissors' action would have violated the jaw angle limits, 0.0 otherwise.
            - rcm_violation_xyz (float): Cartesian difference between desired and actual scissors position.
            - rcm_violation_rpy (float): Rotation angle difference between desired and actual scissors orientation in degrees.
            - successful_task (bool): Whether the task was successful in cutting along the path.
        """

        reward_features = {}

        scissors_position = self.scissors.get_cutting_center_position()
        cloth_points = self.cloth.get_points()
        points_on_cutting_path = self.cloth.get_points_on_cutting_path_centerline()
        points_off_cutting_path = self.cloth.get_points_off_cutting_path()
        cutting_stats = self.cloth.get_cutting_stats()

        # By cutting the cloth, some of its points could drift away very far.
        # The increased bounding box may reduce simulation accuracy or worse, crash the simulation.
        # Check, if any points of the cloth are far away from the actual scene, and set their positions to zero.
        faraway_indices_cloth = np.where(np.linalg.norm(cloth_points, axis=1) > 10.0 / self._distance_normalization_factor)[0]

        if len(faraway_indices_cloth) > 0:
            print(f"WARNING: {len(faraway_indices_cloth)} cloth points are far away from the scene. Setting their positions to [0, 0, 0].")
            with self.cloth.mechanical_object.position.writeable() as writable_array:
                writable_array[faraway_indices_cloth] = [0.0, 0.0, 0.0]

        # Is the cloth unstable? 500 is an empirical value.
        reward_features["unstable_deformation"] = np.amax(self.cloth.mechanical_object.velocity.array()) > 500.0

        # Cut on path
        reward_features["num_tris_on_path"] = cutting_stats["uncut_triangles_on_path"]
        reward_features["cut_on_path"] = previous_reward_features["num_tris_on_path"] - reward_features["num_tris_on_path"]
        # Update the tracking indices, and calculate how much of the path is currently cut.
        if reward_features["cut_on_path"] > 0:
            self.tracking_point_indices_on_cutting_path = farthest_point_sampling(
                points=points_on_cutting_path,
                num_samples=min(self.num_tracking_points_on_cutting_path, len(points_on_cutting_path)),
                rng=self.rng,
            )

            # Retrieve the triangles that are part of the path.
            on_path_triangles = self.cloth.cloth_subset_topology_on_path.triangles.array()
            # Retrieve the indices of the path's centerline.
            centerline_indices = self.cloth.on_path_subset_centerline.SubsetMapping.indices.array()
            # For each point on the centerline of the path, count how often the point occurs in triangles of the path.
            triangles_per_centerline_index = np.sum(centerline_indices[:, None] == on_path_triangles.ravel(), axis=1)
            # Indices that are part of 6 triangles are not yet cut.
            uncut_centerline_points = np.count_nonzero(triangles_per_centerline_index == 6)
            cut_ratio = 1 - uncut_centerline_points / self.cloth.initial_uncut_centerline_points
            reward_features["cut_ratio"] = cut_ratio
            reward_features["delta_cut_ratio"] = cut_ratio - previous_reward_features["cut_ratio"]

        else:
            reward_features["cut_ratio"] = previous_reward_features["cut_ratio"]
            reward_features["delta_cut_ratio"] = 0.0

        # Cut off path
        reward_features["num_tris_off_path"] = cutting_stats["uncut_triangles_off_path"]
        reward_features["cut_off_path"] = previous_reward_features["num_tris_off_path"] - reward_features["num_tris_off_path"]
        # Update the tracking indices
        if reward_features["cut_off_path"] > 0:
            self.tracking_point_indices_off_cutting_path = farthest_point_sampling(
                points=points_off_cutting_path,
                num_samples=min(self.num_tracking_points_off_cutting_path, len(points_off_cutting_path)),
            )

        # State and workspace limits
        reward_features["workspace_violation"] = float(self.scissors.last_set_state_violated_workspace_limits)
        reward_features["state_limits_violation"] = float(self.scissors.last_set_state_violated_state_limits)
        reward_features["jaw_angle_violation"] = float(self.scissors.last_set_angle_violated_jaw_limits)
        rcm_difference = self.scissors.get_pose_difference(position_norm=True)
        reward_features["rcm_violation_xyz"] = rcm_difference[0] * self._distance_normalization_factor
        reward_features["rcm_violation_rotation"] = rcm_difference[1]

        # Find the closest point on cutting path's centerline to the scissors' cutting center.
        # TODO: Filter out the points that already have cut triangles.
        closest_path_point = points_on_cutting_path[np.argmin(np.linalg.norm(points_on_cutting_path - scissors_position, axis=1), keepdims=True)[0]]
        reward_features["distance_scissors_cutting_path"] = np.linalg.norm(scissors_position - closest_path_point) * self._distance_normalization_factor
        reward_features["delta_distance_scissors_cutting_path"] = reward_features["distance_scissors_cutting_path"] - previous_reward_features["distance_scissors_cutting_path"]

        # Visualization of the closest point
        with self.closest_path_point_mechanical_object.position.writeable() as position:
            position[0, :3] = closest_path_point

        reward_features["successful_task"] = reward_features["cut_ratio"] >= self.ratio_to_cut

        return reward_features

    def _get_reward(self) -> float:
        """Retrieve the reward features and scale them with the ``reward_amount_dict``."""
        reward = 0.0
        self.reward_info = {}

        reward_features = self._get_reward_features(self.reward_features)
        # we change the values of the dict -> do a copy (deepcopy not necessary, because the value itself is not manipulated)
        self.reward_features = reward_features.copy()

        self.episode_info["total_cut_on_path"] += reward_features["cut_on_path"]
        self.episode_info["total_cut_off_path"] += reward_features["cut_off_path"]
        self.episode_info["total_unstable_deformation"] += reward_features["unstable_deformation"]
        self.episode_info["ratio_cut_on_path"] = self.episode_info["total_cut_on_path"] / self.initial_tris_on_path
        self.episode_info["ratio_cut_off_path"] = self.episode_info["total_cut_off_path"] / self.initial_tris_off_path

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
        elif self.observation_type == ObservationType.DEPTH:
            observation = self.observation_space.sample()
            observation[:] = self.get_depth()
        else:
            observation = np.zeros(self.observation_space.shape)
            state_dict = {}
            state_dict["ptsd_state"] = self.scissors.get_state()
            state_dict["jaw_angle"] = self.scissors.get_angle()
            state_dict["scissors_pose"] = self.scissors.get_pose()
            points_off_cutting_path = self.cloth.get_points_off_cutting_path()
            points_on_cutting_path = self.cloth.get_points_on_cutting_path_centerline()
            if len(points_off_cutting_path) > 0:
                # NOTE: Technically, there should not be any updates to the topology at this point, because there was no simulation step.
                # However, it happened once, that these indices were invalid -> therefore, we check again.
                indices = self.tracking_point_indices_off_cutting_path[self.tracking_point_indices_off_cutting_path < len(points_off_cutting_path) - 1]
                state_dict["tracking_points_off_cutting_path_positions"] = points_off_cutting_path[indices].ravel()
            if len(points_on_cutting_path) > 0:
                # NOTE: Technically, there should not be any updates to the topology at this point, because there was no simulation step.
                # However, it happened once, that these indices were invalid -> therefore, we check again.
                indices = self.tracking_point_indices_on_cutting_path[self.tracking_point_indices_on_cutting_path < len(points_on_cutting_path) - 1]
                state_dict["tracking_points_on_cutting_path_positions"] = points_on_cutting_path[indices].ravel()
            state_dict["closest_path_point"] = self.closest_path_point_mechanical_object.position.array()[0, :3]

            observation_stack = np.concatenate(tuple(state_dict.values()))
            # We could have fewer points left than specified in the number of tracking points -> fill the rest with zeros
            observation[: len(observation_stack)] = observation_stack
            observation = np.where(np.isnan(observation), 1.0 / self._distance_normalization_factor, observation)

        return observation

    def _get_info(self) -> dict:
        """Assemble the info dictionary."""
        self.info = {}

        for key, value in self.reward_info.items():
            words = key.split("_")[1:]
            shortened_key = reduce(lambda x, y: x + "_" + y[:3], words, "ret")
            self.episode_info[shortened_key] += value

        return {**self.info, **self.reward_info, **self.episode_info, **self.reward_features}

    def reset(self, seed: Union[int, np.random.SeedSequence, None] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Union[np.ndarray, None], Dict]:
        super().reset(seed)

        # Seed the instrument
        if self.unconsumed_seed:
            seeds = self.seed_sequence.spawn(1)
            self.scissors.seed(seed=seeds[0])
            # Identify the indices of cloth points that should be used for describing the state of the scene.
            # Cloth points laying off the cutting path.
            points_off_cutting_path = self.cloth.get_points_off_cutting_path()
            if len(points_off_cutting_path) < self.num_tracking_points_off_cutting_path:
                raise ValueError(f"Cannot track {self.num_tracking_points_off_cutting_path} points off the cutting path. Only {len(points_off_cutting_path)} points are available.")
            self.tracking_point_indices_off_cutting_path = farthest_point_sampling(
                points=points_off_cutting_path,
                num_samples=self.num_tracking_points_off_cutting_path,
                rng=self.rng,
            )
            # Cloth points laying on the centerline of the cutting path.
            points_on_cutting_path = self.cloth.get_points_on_cutting_path_centerline()
            if len(points_on_cutting_path) < self.num_tracking_points_on_cutting_path:
                raise ValueError(f"Cannot track {self.num_tracking_points_on_cutting_path} points on the cutting path. Only {len(points_on_cutting_path)} points are available.")
            self.tracking_point_indices_on_cutting_path = farthest_point_sampling(
                points=points_on_cutting_path,
                num_samples=self.num_tracking_points_on_cutting_path,
                rng=self.rng,
            )
            self.unconsumed_seed = False

        # Reset the scissors
        self.scissors.reset_state()

        # Reset camera
        if self.camera_reset_noise is not None:
            delta_position = self.rng.uniform(-self.camera_reset_noise[:3], self.camera_reset_noise[:3])
            camera_position = self.camera.initial_pose[:3] + delta_position
            self.camera.set_position(camera_position)
            # Reset orientation to avoid drift from set_look_at
            self.camera.set_orientation(self.camera.initial_pose[3:])

            delta_look_at = self.rng.uniform(-self.camera_reset_noise[3:], self.camera_reset_noise[3:])
            camera_look_at = self.camera.initial_look_at + delta_look_at
            self.camera.set_look_at(camera_look_at)

        # Clear the episode info values
        for key in self.episode_info:
            self.episode_info[key] = 0.0

        # and the reward info dict
        self.reward_info = {}

        # and fill the first values used as previous_reward_features
        self.reward_features = {}
        cutting_stats = self.cloth.get_cutting_stats()
        self.reward_features["num_tris_on_path"] = cutting_stats["uncut_triangles_on_path"]
        self.reward_features["num_tris_off_path"] = cutting_stats["uncut_triangles_off_path"]
        self.initial_tris_on_path = self.reward_features["num_tris_on_path"]
        self.initial_tris_off_path = self.reward_features["num_tris_off_path"]

        # Execute any callbacks passed to the env.
        for callback in self.on_reset_callbacks:
            callback(self)

        # Animate several timesteps without actions until simulation settles
        for _ in range(self._settle_steps):
            self.sofa_simulation.animate(self._sofa_root_node, self._sofa_root_node.getDt())

        points_on_cutting_path = self.cloth.get_points_on_cutting_path_centerline()
        cutting_stats = self.cloth.get_cutting_stats()
        nonempty_cutting_path = len(points_on_cutting_path) > 0 and cutting_stats["uncut_triangles_on_path"] > 0

        if nonempty_cutting_path:
            scissors_position = self.scissors.get_cutting_center_position()
            # Find point on cutting path with minimum distance to scissors jaws
            closest_path_point = points_on_cutting_path[np.argmin(np.linalg.norm(points_on_cutting_path - scissors_position, axis=1), keepdims=True)[0]]
            self.reward_features["distance_scissors_cutting_path"] = np.linalg.norm(scissors_position - closest_path_point) * self._distance_normalization_factor
        else:
            self.reward_features["distance_scissors_cutting_path"] = 0.0

        self.reward_features["cut_ratio"] = 0.0

        return self._get_observation(maybe_rgb_observation=self._maybe_update_rgb_buffer()), {}


if __name__ == "__main__":
    from functools import partial
    import pprint
    import time

    import sofa_env.scenes.precision_cutting.sofa_objects.cloth_cut as cloth_cut

    pp = pprint.PrettyPrinter()

    # cutting_path = partial(cloth_cut.sine_cut, amplitude=30, frequency=1.5 / 75, position=0.66)
    # cutting_path = partial(cloth_cut.linear_cut, slope=0.5, position=0.4, depth=1.0)
    def line_cutting_path_generator(rng: np.random.Generator) -> Callable:
        position = rng.uniform(low=0.3, high=0.7)
        depth = rng.uniform(low=0.5, high=0.7)
        slope = rng.uniform(low=-0.5, high=0.5)
        cutting_path = partial(cloth_cut.linear_cut, slope=slope, position=position, depth=depth)
        return cutting_path

    def sine_cutting_path_generator(rng: np.random.Generator) -> Callable:
        position = rng.uniform(low=0.3, high=0.7)
        depth = rng.uniform(low=0.3, high=0.7)
        frequency = rng.uniform(low=0.5, high=1.5) / 75
        amplitude = rng.uniform(low=10.0, high=20.0)
        cutting_path = partial(cloth_cut.sine_cut, frequency=frequency, amplitude=amplitude, position=position, depth=depth)
        return cutting_path

    env = PrecisionCuttingEnv(
        observation_type=ObservationType.STATE,
        action_type=ActionType.CONTINUOUS,
        image_shape=(800, 800),
        render_mode=RenderMode.HUMAN,
        frame_skip=4,
        time_step=0.025,
        settle_steps=10,
        # discrete_action_magnitude=np.array([1, 2, 3, 4, 5, -5, -4, -3, -2, -1]),
        create_scene_kwargs={
            "debug_rendering": False,
            "show_closest_point_on_path": True,
            # "cloth_cutting_path_func": cutting_path,
        },
        cartesian_control=False,
        cloth_cutting_path_func_generator=line_cutting_path_generator,
        ratio_to_cut=1.0,
    )
    env.reset()
    done = False

    fps_list = deque(maxlen=100)
    counter = 0
    episode_length = 50
    while not done:
        for _ in range(100):
            start = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
            if done or counter == episode_length:
                env.reset()
                counter = 0
            counter += 1
            end = time.perf_counter()
            fps = 1 / (end - start)
            fps_list.append(fps)

            print(
                f"({counter}/{episode_length})    FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}",
                end="\r",
            )

            # pp.pprint(info)

        env.reset()
