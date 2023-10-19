import gymnasium.spaces as spaces
import numpy as np

from collections import defaultdict, deque
from enum import Enum, unique
from pathlib import Path
from functools import reduce

from typing import Callable, Union, Tuple, Optional, List, Any, Dict
from sofa_env.base import SofaEnv, RenderMode, RenderFramework
from sofa_env.scenes.ligating_loop.sofa_objects.cavity import Cavity

from sofa_env.scenes.rope_threading.sofa_objects.gripper import ArticulatedGripper
from sofa_env.sofa_templates.camera import Camera
from sofa_env.sofa_templates.rope import Rope

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


class ThreadInHoleEnv(SofaEnv):
    """Thread in Hole Environment

    The goal of this environment to navigate a thread, grasped by a laparoscopic gripper, into a hole.
    The mechanical properties of thread and hole can be heavily modified through ``create_scene_kwargs``.


    Args:
        scene_path (Union[str, Path]): Path to the scene description script that contains this environment's ``createScene`` function.
        image_shape (Tuple[int, int]): Height and Width of the rendered images.
        observation_type (ObservationType): Whether to return RGB images or an array of states as the observation.
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
        action_type (ActionType): Discrete or continuous actions to define the action space of the environment.
        num_thread_tracking_points (int): Number of points on the thread to include in the state observation vector.
        camera_reset_noise (Optional[np.ndarray]): Optional noise to uniformly sample from that is added to the initial camera pose in xyz cartesian position and cartesian point to look at.
        hole_position_reset_noise (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Optional noise to uniformly sample from that is added to the initial position of the hole.
        hole_rotation_reset_noise (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Optional noise to uniformly sample from that is added to the XYZ Euler angle rotation of the hole.
        simple_success_check (bool): Checking whether the thread is inside of the hole is done by an expensive calculation by default, that is valid under large deformation of the hole.
        If this flag is ``True``, this check will be replaced by simply checking if XYZ coordinates of the thread are within the original position of the hole. Not valid under deformation of the hole.
        insertion_ratio_threshold (float): Ratio of the thread that should be inserted into the hole. The environment will adapt this ratio, if the mechanical config would not allow the desired ratio.
        For example if a 10 meter long thread should be inserted into a 1 meter deep hole with a ratio of more than 0.1. The task is only considered successful, if the last point on the thread,
        and as many consecutive points on the thread as specified by the ratio are within the hole.
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
            "thread_tip_distance_to_hole": -0.0,
            "delta_thread_tip_distance_to_hole": -0.0,
            "thread_center_distance_to_hole": -0.0,
            "delta_thread_center_distance_to_hole": -0.0,
            "thread_points_distance_to_hole": -0.0,
            "delta_thread_points_distance_to_hole": -0.0,
            "unstable_deformation": -0.0,
            "thread_velocity": -0.0,
            "gripper_velocity": -0.0,
            "successful_task": 0.0,
            "action_violated_cartesian_workspace": -0.0,
            "action_violated_state_limits": -0.0,
            "ratio_rope_in_hole": 0.0,
            "delta_ratio_rope_in_hole": 0.0,
            "gripper_collisions": -0.0,
        },
        maximum_state_velocity: Union[np.ndarray, float] = np.array([20.0, 20.0, 20.0, 20.0]),
        discrete_action_magnitude: Union[np.ndarray, float] = np.array([15.0, 15.0, 15.0, 10.0]),
        create_scene_kwargs: Optional[dict] = None,
        on_reset_callbacks: Optional[List[Callable]] = None,
        action_type: ActionType = ActionType.CONTINUOUS,
        num_thread_tracking_points: int = -1,
        camera_reset_noise: Optional[np.ndarray] = None,
        hole_position_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        hole_rotation_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        simple_success_check: bool = False,
        insertion_ratio_threshold: float = 0.5,
    ) -> None:
        # Pass image shape to the scene creation function
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape
        create_scene_kwargs["create_shell"] = not simple_success_check
        create_scene_kwargs["hole_position_reset_noise"] = hole_position_reset_noise
        create_scene_kwargs["hole_rotation_reset_noise"] = hole_rotation_reset_noise
        self.call_hole_reset = hole_rotation_reset_noise is not None or hole_rotation_reset_noise is not None

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

        # Check if the thread is in the hole with a simple approximation, or based on the actual deformed hull of the hole
        self.simple_success_check = simple_success_check

        # How much of the thread must be in the hole to be considered successful
        self.insertion_ratio_threshold = insertion_ratio_threshold

        # Randomize camera pose on reset
        self.camera_reset_noise = camera_reset_noise
        if camera_reset_noise is not None:
            if not isinstance(camera_reset_noise, np.ndarray) and not camera_reset_noise.shape == (6,):
                raise ValueError(
                    "Please pass the camera_reset_noise as a numpy array with 6 values for maximum deviation \
                        from the original camera pose in xyz cartesian position and cartesian point to look at."
                )

        ##############
        # Action Space
        ##############
        action_dimensionality = 4
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
                    raise ValueError(
                        "If you want to use individual discrete action step sizes per action dimension, \
                            please pass an array of length {action_dimensionality} as discrete_action_magnitude. \
                            Received {discrete_action_magnitude=} with lenght {len(discrete_action_magnitude)}."
                    )

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
        self.num_thread_tracking_points = num_thread_tracking_points
        self.thread_tracking_point_indices: Union[np.ndarray, None]  # set in self._init_sim

        if observation_type == ObservationType.STATE:
            # ptsd_state -> 4
            # gripper_pose -> 7
            # thread_center_of_mass_position -> 3
            # thread_tracking_point_positions -> num_thread_tracking_points * 3
            # hole_opening_position -> 3
            observations_size = 4 + 7 + 3 + num_thread_tracking_points * 3 + 3
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
        super()._init_sim()

        self.gripper: ArticulatedGripper = self.scene_creation_result["gripper"]
        self.thread: Rope = self.scene_creation_result["thread"]
        self.hole: Cavity = self.scene_creation_result["hole"]
        self.camera: Camera = self.scene_creation_result["camera"]
        self.contact_listeners: Tuple = self.scene_creation_result["contact_listeners"]

        # Factor for normalizing the distances in the reward function.
        # Based on the workspace of the gripper.
        self._distance_normalization_factor = 1.0 / np.linalg.norm(self.gripper.cartesian_workspace["high"] - self.gripper.cartesian_workspace["low"])

        # Save the gripper position for velocity calculation
        self.previous_gripper_position = self.gripper.get_pose()[:3].copy()

        # Identify the indices of points on the torus that should be used for describing the state of the rope. If num_thread_tracking_points is -1, use all of them.
        if self.num_thread_tracking_points > 0:
            if self.thread.num_points < self.num_thread_tracking_points:
                raise Exception(f"The number of thread tracking points ({self.num_thread_tracking_points}) is larger than the number of points on the thread ({self.thread.num_points}).")
            self.thread_tracking_point_indices = np.linspace(0, self.thread.num_points - 1, num=self.num_thread_tracking_points, endpoint=True, dtype=np.int16)
        elif self.num_thread_tracking_points == -1:
            self.thread_tracking_point_indices = np.array(range(self.thread.num_points), dtype=np.int16)
        elif self.num_thread_tracking_points == 0:
            self.thread_tracking_point_indices = None
        else:
            raise ValueError(f"num_thread_tracking_points must be > 0 or == -1 (to use them all). Received {self.num_thread_tracking_points}.")

        # Limit insertion_ratio_threshold such that the thread can actually be inserted to the desired ratio
        if self.thread.length * self.insertion_ratio_threshold >= self.hole.height:
            old_insertion_ratio_threshold = self.insertion_ratio_threshold
            segment_length = self.thread.length / (self.thread.num_points - 1)
            # Round down to the next integer, because the success and ratio checks are done on discrete points on the thread
            segments_to_fit_in_hole = int(self.hole.height / segment_length)
            self.insertion_ratio_threshold = segments_to_fit_in_hole / (self.thread.num_points - 1)
            print(f"[WARNING] insertion_ratio_threshold was set to {self.insertion_ratio_threshold} to ensure that the thread can actually fit in the hole to solve the task. Previously, it was set to {old_insertion_ratio_threshold}.")

        self.indices_to_check_for_success = list(range(self.thread.num_points))[int((1 - self.insertion_ratio_threshold) * self.thread.num_points) :]

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

        self.gripper.set_state(self.gripper.get_state() + self._scale_action(action))

    def _get_reward_features(self, previous_reward_features: dict) -> dict:
        """Get the features that may be used to assemble the reward function

        Features:
            - thread_tip_distance_to_hole (float): Distance between the tip of the thread and the hole opening on the top.
            - delta_thread_tip_distance_to_hole (float): Change in distance between the tip of the thread and the hole opening on the top.
            - thread_center_distance_to_hole (float): Distance between the center of mass of the thread and the hole opening on the top.
            - delta_thread_center_distance_to_hole (float): Change in distance between the center of mass of the thread and the hole opening on the top.
            - thread_points_distance_to_hole (np.ndarray): Distances between the tracking points on the thread and the hole opening on the top.
            - delta_thread_points_distance_to_hole (np.ndarray): Change in distances between the tracking points on the thread and the hole opening on the top.
            - unstable_deformation (bool): An approximation of whether the thread deformation might be unstable. Checked by looking at the velocities of the thread.
            - thread_velocity (float): Mean velocity of all points on the thread.
            - gripper_velocity (float): Cartesian velocity of the gripper.
            - action_violated_state_limits (bool): Whether the last action would have violated the state limits of the gripper.
            - action_violated_cartesian_workspace (bool): Whether the last action would have violated the Cartesian workspace limits of the gripper.
            - successful_task (float): Whether the center of mass of the rope is within the hole.
            - ratio_rope_in_hole (float): Ratio of the rope that is in the hole.
            - delta_ratio_rope_in_hole (float): Change in ratio of the rope that is in the hole.
            - gripper_collisions (int): Number of collisions between gripper and cylinder.
        """

        reward_features = {}

        thread_center_of_mass = self.thread.get_center_of_mass()
        thread_positions = self.thread.get_positions()
        hole_position = self.hole.get_center_of_opening_position()

        # Distances between thread's tip and the hole's opening
        reward_features["thread_tip_distance_to_hole"] = np.linalg.norm(hole_position - thread_positions[-1])

        # Change in distances
        reward_features["delta_thread_tip_distance_to_hole"] = reward_features["thread_tip_distance_to_hole"] - previous_reward_features["thread_tip_distance_to_hole"]

        # Distances between thread's center of mass and the hole's opening
        reward_features["thread_center_distance_to_hole"] = np.linalg.norm(hole_position - thread_center_of_mass)

        # Change in distances
        reward_features["delta_thread_center_distance_to_hole"] = reward_features["thread_center_distance_to_hole"] - previous_reward_features["thread_center_distance_to_hole"]

        if self.thread_tracking_point_indices is not None:
            # Distance between the hole opening and thread tracking points
            reward_features["thread_points_distance_to_hole"] = np.linalg.norm(hole_position - thread_positions[self.thread_tracking_point_indices], axis=1)
            # Change in distance between the hole opening and thread tracking points
            reward_features["delta_thread_points_distance_to_hole"] = reward_features["thread_points_distance_to_hole"] - previous_reward_features["thread_points_distance_to_hole"]
        else:
            reward_features["thread_points_distance_to_hole"] = 0.0
            reward_features["delta_thread_points_distance_to_hole"] = 0.0

        # Is the deformable model of the thread unstable?
        # The thread is probably unstable, if the mean velocity is larger than 150% of the gripper speed, and larger than 25 mm/s
        mean_thread_velocity = np.mean(np.linalg.norm(self.thread.mechanical_object.velocity.array()[:, :3], axis=1))
        gripper_position = self.gripper.get_pose()[:3]
        gripper_velocity = np.linalg.norm(self.previous_gripper_position - gripper_position) / (self.time_step * self.frame_skip)
        thread_probably_unstable = mean_thread_velocity > 1.5 * gripper_velocity and mean_thread_velocity > 25.0
        self.previous_gripper_position[:] = gripper_position
        reward_features["unstable_deformation"] = thread_probably_unstable

        # Cartesian speed of gripper and torus center
        reward_features["thread_velocity"] = mean_thread_velocity
        reward_features["gripper_velocity"] = gripper_velocity

        # Did the action violate state or workspace limits?
        reward_features["action_violated_cartesian_workspace"] = self.gripper.last_set_state_violated_workspace_limits
        reward_features["action_violated_state_limits"] = self.gripper.last_set_state_violated_state_limits

        # How much of the rope is inside the hole?
        if self.simple_success_check:
            reward_features["ratio_rope_in_hole"] = np.sum(np.logical_and(thread_positions[:, 2] < self.hole.height, np.linalg.norm(thread_positions[:, :2], axis=1) < self.hole.inner_radius)) / len(thread_positions)
            # Task is successful, if a desired ratio of points on the thread are within the hollow cylinder
            points_to_check = thread_positions[self.indices_to_check_for_success]
            # Z coordinate below the cylinder's height AND
            # XY coordinate within the inner radius
            reward_features["successful_task"] = np.all(points_to_check[:, 2] < self.hole.height) and np.all(np.linalg.norm(points_to_check[:, :2], axis=1) < self.hole.inner_radius)

        else:
            thread_positions_are_in_hole = self.hole.are_in_cavity(thread_positions)
            reward_features["ratio_rope_in_hole"] = np.sum(thread_positions_are_in_hole) / len(thread_positions)
            reward_features["successful_task"] = np.all(thread_positions_are_in_hole[self.indices_to_check_for_success])

        reward_features["delta_ratio_rope_in_hole"] = reward_features["ratio_rope_in_hole"] - previous_reward_features["ratio_rope_in_hole"]

        # Collisions between gripper and cylinder
        reward_features["gripper_collisions"] = 0
        for listener in self.contact_listeners:
            reward_features["gripper_collisions"] += listener.getNumberOfContacts()

        return reward_features

    def _get_reward(self) -> float:
        """Retrieve the reward features and scale them with the ``reward_amount_dict``."""
        reward = 0.0
        self.reward_info = {}

        reward_features = self._get_reward_features(self.reward_features)
        # we change the values of the dict -> do a copy (deepcopy not necessary, because the value itself is not manipulated)
        self.reward_features = reward_features.copy()

        # These values are arrays -> reduce them to a scalar
        if self.thread_tracking_point_indices is not None:
            reward_features["thread_points_distance_to_hole"] = np.min(reward_features["thread_points_distance_to_hole"])
            reward_features["delta_thread_points_distance_to_hole"] = np.min(reward_features["delta_thread_points_distance_to_hole"])

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
            state_dict = {}
            state_dict["ptsd_state"] = self.gripper.get_state()
            state_dict["gripper_pose"] = self.gripper.get_pose()
            state_dict["thread_center_of_mass_position"] = self.thread.get_center_of_mass()
            if self.thread_tracking_point_indices is not None:
                state_dict["thread_tracking_point_positions"] = self.thread.get_positions()[self.thread_tracking_point_indices].ravel()
            state_dict["hole_opening_position"] = self.hole.get_center_of_opening_position()

            observation = np.concatenate(tuple(state_dict.values()))
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
        super().reset(seed)

        # Seed the instruments
        if self.unconsumed_seed:
            seeds = self.seed_sequence.spawn(2)
            self.gripper.seed(seed=seeds[0])
            self.hole.seed(seed=seeds[1])
            self.unconsumed_seed = False

        # Reset the gripper
        self.gripper.reset_gripper()
        self.previous_gripper_position[:] = self.gripper.get_pose()[:3]

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

        if self.call_hole_reset:
            self.hole.reset_cavity()

        thread_center_of_mass = self.thread.get_center_of_mass()
        thread_positions = self.thread.get_positions()
        hole_position = self.hole.get_center_of_opening_position()

        self.reward_features["thread_tip_distance_to_hole"] = np.linalg.norm(hole_position - thread_positions[-1])
        self.reward_features["thread_center_distance_to_hole"] = np.linalg.norm(hole_position - thread_center_of_mass)
        if self.simple_success_check:
            self.reward_features["ratio_rope_in_hole"] = np.sum(np.logical_and(thread_positions[:, 2] < self.hole.height, np.linalg.norm(thread_positions[:, :2], axis=1) < self.hole.inner_radius)) / len(thread_positions)
        else:
            self.reward_features["ratio_rope_in_hole"] = np.sum(self.hole.are_in_cavity(thread_positions)) / len(thread_positions)
        if self.thread_tracking_point_indices is not None:
            self.reward_features["thread_points_distance_to_hole"] = np.linalg.norm(hole_position - thread_positions[self.thread_tracking_point_indices], axis=1)

        # Execute any callbacks passed to the env.
        for callback in self.on_reset_callbacks:
            callback(self)

        # Animate several timesteps without actions until simulation settles
        for _ in range(self._settle_steps):
            self.sofa_simulation.animate(self._sofa_root_node, self._sofa_root_node.getDt())

        # If after reset, any points of the thread are stuck in the cylinder, reset again
        thread_positions = self.thread.get_positions()
        if self.simple_success_check:
            invalid_starting_configuration = np.any(np.logical_and(thread_positions[:, 2] < self.hole.height, np.linalg.norm(thread_positions[:, :2], axis=1) < self.hole.outer_radius))
        else:
            invalid_starting_configuration = np.any(self.hole.are_in_cavity(thread_positions))
        if invalid_starting_configuration:
            print(f"[WARNING]: Peg started inside cylinder on reset after {self._settle_steps} steps of letting the simulation settle. Will reset again.")
            self.reset()

        return self._get_observation(maybe_rgb_observation=self._maybe_update_rgb_buffer()), {}


if __name__ == "__main__":
    import pprint
    import time

    pp = pprint.PrettyPrinter()

    env = ThreadInHoleEnv(
        observation_type=ObservationType.RGB,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.CONTINUOUS,
        image_shape=(800, 800),
        frame_skip=3,
        time_step=0.01,
        settle_steps=50,
        num_thread_tracking_points=-1,
        # camera_reset_noise=np.array([20, 20, 20, 20, 20, 20]),
        hole_rotation_reset_noise=np.array([10, 10, 10]),
        hole_position_reset_noise=np.array([30, 30, 30]),
    )

    env.reset()
    done = False

    fps_list = deque(maxlen=100)
    counter = 0
    while not done:
        for _ in range(100):
            start = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
            if counter == 20:
                env.reset()
                counter = 0
            counter += 1
            end = time.perf_counter()
            fps = 1 / (end - start)
            fps_list.append(fps)
            # pp.pprint(info)
            print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")

        env.reset()
