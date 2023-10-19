import cv2
import gymnasium.spaces as spaces
import numpy as np

from collections import deque, defaultdict
from enum import Enum, unique
from pathlib import Path
from functools import reduce

from typing import Union, Tuple, Optional, Any, List, Callable, Dict

from sofa_env.base import SofaEnv, RenderMode, RenderFramework
from sofa_env.scenes.reach.sofa_objects.end_effector import EndEffector
from sofa_env.sofa_templates.rigid import ControllableRigidObject

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


class ReachEnv(SofaEnv):
    """Reaching Task Environment

    The goal of this environment is to control a robotic end-effector in Cartesian space to reach a desired position.

    Args:
        scene_path (Union[str, Path]): Path to the scene description script that contains this environment's ``createScene`` function.
        image_shape (Tuple[int, int]): Height and Width of the rendered images.
        observation_type (ObservationType): Observation type of the env. Can be ``RGB`` (color), ``STATE`` (state vector), ``DEPTH`` (depth image), or ``RGBD`` (color and depth image).
        observe_target_position (bool): Whether to include the position of the target point into the observation. Observation space turns into a ``dict``, if ``observation_type`` is not ``ObservationType.STATE``
        distance_to_target_threshold (float): Distance between target position and end effector position in meters at which episode is marked as done.
        minimum_distance_to_target_at_start (float): Minimum distance between end effector and target in meters when resetting the environment.
        time_step (float): size of simulation time step in seconds.
        frame_skip (int): number of simulation time steps taken (call _do_action and advance simulation) each time step is called.
        maximum_robot_velocity (float): Maximum per Cartesian direction robot velocity in millimeters per second. Used for scaling the actions that are passed to ``env.step(action)``.
        discrete_action_magnitude (float): Discrete delta motion in millimeters per second applied to the end effector for the discrete action type.
        action_type (ActionType): Discrete or continuous actions to define the action space of the environment.
        render_mode (RenderMode): create a window (RenderMode.HUMAN) or run headless (RenderMode.HEADLESS).
        render_framework (RenderFramework): choose between pyglet and pygame for rendering
        reward_amount_dict (dict): Dictionary to weigh the components of the reward function.
        create_scene_kwargs (Optional[dict]): A dictionary to pass additional keyword arguments to the ``createScene`` function.
        render_mode (RenderMode): Create a window (``RenderMode.HUMAN``), run headless (``RenderMode.HEADLESS``), or do not create a render buffer at all (``RenderMode.NONE``).
        on_reset_callbacks (Optional[List[Callable]]): A list of callables to call after the environment is reset.
        sphere_radius (float): Radius of the target sphere in meters.
    """

    def __init__(
        self,
        scene_path: Union[str, Path] = SCENE_DESCRIPTION_FILE_PATH,
        image_shape: Tuple[int, int] = (124, 124),
        observation_type: ObservationType = ObservationType.RGB,
        observe_target_position: bool = False,
        distance_to_target_threshold: float = 0.003,
        minimum_distance_to_target_at_start: float = 0.05,
        time_step: float = 0.1,
        frame_skip: int = 1,
        maximum_robot_velocity: float = 35.0,
        discrete_action_magnitude: float = 10.0,
        action_type: ActionType = ActionType.CONTINUOUS,
        render_mode: RenderMode = RenderMode.HEADLESS,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        reward_amount_dict={
            "distance_to_target": -0.0,
            "delta_distance_to_target": -0.0,
            "time_step_cost": -0.0,
            "workspace_violation": -0.0,
            "successful_task": 0.0,
        },
        create_scene_kwargs: Optional[dict] = None,
        on_reset_callbacks: Optional[List[Callable]] = None,
        sphere_radius: float = 0.008,
    ) -> None:
        # Pass image shape to the scene creation function
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape
        create_scene_kwargs["sphere_radius"] = sphere_radius

        self.sphere_radius = sphere_radius

        super().__init__(
            scene_path=scene_path,
            time_step=time_step,
            frame_skip=frame_skip,
            render_mode=render_mode,
            render_framework=render_framework,
            create_scene_kwargs=create_scene_kwargs,
        )

        ##############
        # Action Space
        ##############
        action_dimensionality = 3
        self.action_type = action_type
        if action_type == ActionType.CONTINUOUS:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dimensionality,), dtype=np.float32)
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
            self._discrete_action_lookup *= self.time_step * 0.001  # mm to m
            self._discrete_action_lookup.flags.writeable = False

        self._maximum_robot_velocity = maximum_robot_velocity
        self._discrete_action_magnitude = discrete_action_magnitude

        ###################
        # Observation Space
        ###################
        self._observe_target_position = observe_target_position
        # State observations
        if observation_type == ObservationType.STATE:
            dim_states = 3
            if observe_target_position:
                dim_states += 3
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(dim_states,), dtype=np.float32)

        # Image observations
        elif observation_type == ObservationType.RGB:
            if observe_target_position:
                self.observation_space = spaces.Dict(
                    {
                        "rgb": spaces.Box(low=0, high=255, shape=image_shape + (3,), dtype=np.uint8),
                        "target_position": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                    }
                )
            else:
                self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (3,), dtype=np.uint8)

        elif observation_type == ObservationType.RGBD:
            if observe_target_position:
                self.observation_space = spaces.Dict(
                    {
                        "rgbd": spaces.Box(low=0, high=255, shape=image_shape + (4,), dtype=np.uint8),
                        "target_position": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                    }
                )
            else:
                self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (4,), dtype=np.uint8)

        elif observation_type == ObservationType.DEPTH:
            if observe_target_position:
                self.observation_space = spaces.Dict(
                    {
                        "depth": spaces.Box(low=0, high=255, shape=image_shape + (1,), dtype=np.uint8),
                        "target_position": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                    }
                )
            else:
                self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (1,), dtype=np.uint8)

        else:
            raise Exception(f"Please set observation_type to a value of ObservationType. Received {observation_type}.")

        self._observation_type = observation_type

        #########################
        # Episode specific values
        #########################

        self._last_action_violated_workspace = False

        # Infos per episode
        self.episode_info = defaultdict(float)

        # Infos from the reward
        self.reward_info = {}
        self.reward_features = {}

        # Task specific values
        self._distance_to_target_threshold = distance_to_target_threshold
        self._minimum_distance_to_target_at_start = minimum_distance_to_target_at_start

        # All parameters of the reward function that are not passed default to 0.0
        self.reward_amount_dict = defaultdict(float) | reward_amount_dict

        # Callback functions called on reset
        if on_reset_callbacks is not None:
            self.on_reset_callbacks = on_reset_callbacks
        else:
            self.on_reset_callbacks = []

    def _init_sim(self):
        "Initialise simulation and calculate values for reward scaling."
        super()._init_sim()

        self._workspace: dict = self.scene_creation_result["workspace"]
        self.end_effector: EndEffector = self.scene_creation_result["interactive_objects"]["end_effector"]
        self._visual_target: ControllableRigidObject = self.scene_creation_result["interactive_objects"]["visual_target"]

        # Scale the all distance values to an interval of [0, 1] based on the end effector workspace
        self._distance_normalization_factor = 1.0 / np.linalg.norm(self._workspace["high"][:3] - self._workspace["low"][:3])

    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        """Step function of the environment that applies the action to the simulation and returns observation, reward, done signal, and info."""

        maybe_rgb_observation = super().step(action)

        observation = self._get_observation(maybe_rgb_observation=maybe_rgb_observation)
        reward = self._get_reward()
        terminated = self._get_done()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _scale_discrete_action(self, action: int) -> np.ndarray:
        """Maps action indices to a motion delta."""
        return self._discrete_action_lookup[action]

    def _scale_continuous_action(self, action: np.ndarray) -> np.ndarray:
        """
        Policy is output is clipped to [-1, 1] e.g. [-0.3, 0.8, 1].
        We want to scale that to [-max_rob_vel, max_rob_vel] in mm/s,
        we then have to scale it to m/s (SOFA scene is in meter),
        and further to m/sim_step (because delta T is not 1 second).
        """
        return self.time_step * 0.001 * self._maximum_robot_velocity * action

    def _do_action(self, action: Union[int, np.ndarray]) -> None:
        """Scale action and set new poses in simulation"""

        # scale the action according to the function determined during init
        sofa_action = self._scale_action(action)

        # The action corresponds to delta in XYZ, but sofa objects want XYZ + rotation as quaternion.
        # We keep the current orientation, by appending the rotation read from simulation.
        current_pose = self.end_effector.get_pose()
        current_position = current_pose[:3]
        current_orientation = current_pose[3:]
        new_position = current_position + sofa_action
        new_pose = np.append(new_position, current_orientation)
        invalid_poses_mask = self.end_effector.set_pose(new_pose)
        self._last_action_violated_workspace = any(invalid_poses_mask)

    def _get_reward_features(self, previous_reward_features: dict) -> dict:
        """Get the features that may be used to assemble the reward function

        Features:
            - distance_to_target (float): Distance between end effector and target in meters.
            - delta_distance_to_target (float): Change in distance between end effector and target in meters since the last step.
            - time_step_cost (float): 1.0 for every step.
            - workspace_violation (float): 1.0 if the action would have violated the workspace.
            - successful_task (float): 1.0 if the distance between the end effector and the target is below the threshold ``distance_to_target_threshold``.
        """

        reward_features = {}
        current_position = self.end_effector.get_pose()[:3]
        target_position = self._visual_target.get_pose()[:3]

        reward_features["distance_to_target"] = np.linalg.norm(current_position - target_position)
        reward_features["delta_distance_to_target"] = reward_features["distance_to_target"] - previous_reward_features["distance_to_target"]

        reward_features["time_step_cost"] = 1.0

        reward_features["workspace_violation"] = float(self._last_action_violated_workspace)

        reward_features["successful_task"] = float(reward_features["distance_to_target"] <= self._distance_to_target_threshold)

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
                value = self._distance_normalization_factor * value
            self.reward_info[f"reward_{key}"] = self.reward_amount_dict[key] * value
            reward += self.reward_info[f"reward_{key}"]

        self.reward_info["reward"] = reward

        return float(reward)

    def _get_done(self) -> bool:
        """Look up if the episode is finished."""
        return bool(self.reward_features["successful_task"])

    def _normalize_position(self, position: np.ndarray) -> np.ndarray:
        """Normalizes a position with the environments workspace."""
        return 2 * (position - self._workspace["low"][:3]) / (self._workspace["high"][:3] - self._workspace["low"][:3]) - 1

    def _get_observation(self, maybe_rgb_observation: Union[np.ndarray, None]) -> Union[np.ndarray, dict]:
        """Assembles the correct observation based on the ``ObservationType`` and ``observe_phase_state``."""

        if self._observation_type == ObservationType.RGB:
            if self._observe_target_position:
                observation = {
                    "rgb": maybe_rgb_observation,
                    "target_position": self._normalize_position(self._visual_target.get_pose()[:3]),
                }
            else:
                observation = maybe_rgb_observation

        elif self._observation_type == ObservationType.RGBD:
            if self._observe_target_position:
                observation = self.observation_space.sample()
                observation["rgbd"][:, :, :3] = maybe_rgb_observation
                observation["rgbd"][:, :, 3:] = self.get_depth()
                observation["target_position"][:] = self._normalize_position(self._visual_target.get_pose()[:3])
            else:
                observation = self.observation_space.sample()
                observation[:, :, :3] = maybe_rgb_observation
                observation[:, :, 3:] = self.get_depth()

        elif self._observation_type == ObservationType.DEPTH:
            if self._observe_target_position:
                observation = self.observation_space.sample()
                observation["depth"][:] = self.get_depth()
                observation["target_position"][:] = self._normalize_position(self._visual_target.get_pose()[:3])
            else:
                observation = self.observation_space.sample()
                observation[:] = self.get_depth()

        else:
            observation = self.observation_space.sample()
            observation[:3] = self._normalize_position(self.end_effector.get_pose()[:3])
            if self._observe_target_position:
                observation[3:] = self._normalize_position(self._visual_target.get_pose()[:3])

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

        # Seed the instrument
        if self.unconsumed_seed:
            # Seed the end effector's random number generator with the seed sequence from the env
            seeds = self.seed_sequence.spawn(1)
            self.end_effector.seed(seed=seeds[0])
            self.unconsumed_seed = False

        # Reset end_effector
        self.end_effector.reset()

        # Place the target at a new position, that is at least minimum_distance_to_target_at_start away from the end effector
        new_target_position_found = False
        while not new_target_position_found:
            target_pose = self.rng.uniform(self._workspace["low"], self._workspace["high"])
            initial_distance = np.linalg.norm(self.end_effector.initial_pose[:3] - target_pose[:3])
            new_target_position_found = initial_distance >= self._minimum_distance_to_target_at_start
        self._visual_target.set_pose(target_pose)
        self.target_position = target_pose[:3]

        # Clear the episode info values
        for key in self.episode_info:
            self.episode_info[key] = 0.0

        # and the reward info dict
        self.reward_info = {}

        # and fill the first values used as previous_reward_features
        self.reward_features = {}
        self.reward_features["distance_to_target"] = np.linalg.norm(self.end_effector.get_pose()[:3] - self._visual_target.get_pose()[:3])

        # Execute any callbacks passed to the env.
        for callback in self.on_reset_callbacks:
            callback(self)

        return self._get_observation(maybe_rgb_observation=self._maybe_update_rgb_buffer()), {}


if __name__ == "__main__":
    import pprint
    import time

    pp = pprint.PrettyPrinter()

    env = ReachEnv(
        observation_type=ObservationType.RGBD,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.DISCRETE,
        observe_target_position=False,
        image_shape=(480, 480),
        frame_skip=1,
        time_step=0.1,
        reward_amount_dict={
            "distance_to_target": -1.0,
            "delta_distance_to_target": -1.0,
            "time_step_cost": -1.0,
            "workspace_violation": -1.0,
            "successful_task": 1.0,
        },
    )

    env.reset()

    fps_list = deque(maxlen=100)

    done = False
    counter = 0
    while not done:
        start = time.time()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if env._observation_type == ObservationType.DEPTH:
            cv2.imshow("Depth Image", obs)
            cv2.waitKey(1)
        if env._observation_type == ObservationType.RGBD:
            cv2.imshow("Depth Image", obs[:, :, 3])
            cv2.waitKey(1)
        end = time.time()
        fps = 1 / (end - start)
        fps_list.append(fps)
        if counter % 100 == 0:
            env.reset()
        counter += 1

        print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")
