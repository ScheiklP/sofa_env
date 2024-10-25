import gymnasium.spaces as spaces
import numpy as np

from collections import defaultdict, deque
from enum import Enum, unique
from pathlib import Path
from functools import reduce

from typing import Callable, Union, Tuple, Optional, List, Any, Dict
from sofa_env.base import SofaEnv, RenderMode, RenderFramework

from sofa_env.scenes.rope_cutting.sofa_objects.rope import CuttableRope
from sofa_env.scenes.tissue_dissection.sofa_objects.cauter import PivotizedCauter
from sofa_env.sofa_templates.camera import PhysicalCamera

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


class RopeCuttingEnv(SofaEnv):
    """Rope Cutting Environment

    The goal of this environment is to cut a series of ropes with a cauter tool that can be turned on and off.
    At any time, one of the remaining ropes is highlighted as active and thus marked as the next rope to be cut.
    If the rope cuts an inactive rope, the episode ends, if not enough ropes are left to reach the desired
    number of active ropes to be cut.

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
        activation_deadzone (float): Defines the deadzone in which the action for changing the state of the cauter (on/off) is ignored.
        create_scene_kwargs (Optional[dict]): A dictionary to pass additional keyword arguments to the ``createScene`` function.
        on_reset_callbacks (Optional[List[Callable]]): Functions that are called as the last step of the ``env.reset()`` function.
        num_tracking_points_ropes (int): Number of points per rope to include in the state observation. If set to -1, use all points.
        num_ropes (int): Number of ropes to add to the scene.
        num_ropes_to_cut (int): Number of ropes that have to be cut before the task is considered done.
        normalize_observations (bool): Whether to normalize the ``ptsd_state`` with the cauter's state limits, the pose of the cauter and positions
        of the ropes with the cauter's Cartesian workspace.
    """

    def __init__(
        self,
        scene_path: Union[str, Path] = SCENE_DESCRIPTION_FILE_PATH,
        image_shape: Tuple[int, int] = (124, 124),
        observation_type: ObservationType = ObservationType.RGB,
        action_type: ActionType = ActionType.CONTINUOUS,
        time_step: float = 0.1,
        frame_skip: int = 10,
        settle_steps: int = 50,
        settle_step_dt: float = 0.1,
        render_mode: RenderMode = RenderMode.HEADLESS,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        reward_amount_dict={
            "distance_cauter_active_rope": -0.0,
            "delta_distance_cauter_active_rope": -0.0,
            "cut_active_rope": 0.0,
            "cut_inactive_rope": -0.0,
            "workspace_violation": -0.0,
            "state_limits_violation": -0.0,
            "successful_task": 0.0,
            "failed_task": -0.0,
        },
        maximum_state_velocity: Union[np.ndarray, float] = np.array([20.0, 20.0, 20.0, 20.0]),
        discrete_action_magnitude: Union[np.ndarray, float] = np.array([15.0, 15.0, 15.0, 10.0]),
        activation_deadzone: float = 0.1,
        create_scene_kwargs: Optional[dict] = None,
        on_reset_callbacks: Optional[List[Callable]] = None,
        num_tracking_points_ropes: int = 3,
        num_ropes: int = 5,
        num_ropes_to_cut: int = 2,
        normalize_observations: bool = True,
    ) -> None:
        # Pass image shape to the scene creation function
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape
        create_scene_kwargs["num_ropes"] = num_ropes

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
        self._settle_step_dt = settle_step_dt

        self.maximum_state_velocity = maximum_state_velocity
        self.activation_deadzone = activation_deadzone
        self.num_ropes_to_cut = num_ropes_to_cut
        self.num_ropes = num_ropes
        self.num_cut_ropes = 0
        self.num_cut_inactive_ropes = 0
        self.rope_indices = list(range(num_ropes))
        self.num_tracking_points_ropes = num_tracking_points_ropes
        self.active_rope_color = (0, 255, 0)
        self.inactive_rope_color = (255, 0, 0)
        self.normalize_observations = normalize_observations

        ##############
        # Action Space
        ##############
        action_dimensionality = 5
        self.action_type = action_type
        if action_type == ActionType.CONTINUOUS:
            self._scale_action = self._scale_continuous_action
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dimensionality,), dtype=np.float32)
        else:
            self._scale_action = self._scale_discrete_action
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

            self._discrete_action_lookup = np.array(action_list)
            self._discrete_action_lookup *= self.time_step
            self._discrete_action_lookup.flags.writeable = False

        ###################
        # Observation Space
        ###################

        if observation_type == ObservationType.STATE:
            # ptsd_state -> 4
            # active_index -> 1
            # rope_tracking_points -> num_tracking_points_ropes * 3 * num_ropes
            # rope_tracking_points_active -> num_tracking_points_ropes * 3
            # pose -> 7
            observations_size = 4 + 1 + num_tracking_points_ropes * 3 * (num_ropes + 1) + 7
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

        self.cauter: PivotizedCauter = self.scene_creation_result["cauter"]
        self.ropes: List[CuttableRope] = self.scene_creation_result["ropes"]
        self.camera: PhysicalCamera = self.scene_creation_result["physical_camera"]

        # Factor for normalizing the distances in the reward function.
        # Based on the workspace of the cauter.
        self._distance_normalization_factor = 1.0 / np.linalg.norm(self.cauter.cartesian_workspace["high"] - self.cauter.cartesian_workspace["low"])
        self._corner_position = self.cauter.cartesian_workspace["high"].copy()

        example_rope = self.ropes[0]
        self.rope_points = example_rope.number_of_points
        # Tracking points on the loop
        if self.num_tracking_points_ropes > 0:
            if self.rope_points < self.num_tracking_points_ropes:
                raise Exception(f"The number of rope tracking points ({self.num_tracking_points_ropes}) is larger than the number of points on the rope ({self.rope_points}).")
            self.ropes_tracking_point_indices = np.linspace(0, self.rope_points - 1, num=self.num_tracking_points_ropes, endpoint=True, dtype=np.int16)
        elif self.num_tracking_points_ropes == -1:
            self.ropes_tracking_point_indices = np.array(range(self.rope_points), dtype=np.int16)
        elif self.num_tracking_points_ropes == 0:
            self.ropes_tracking_point_indices = None
        else:
            if self.observation_type == ObservationType.STATE:
                raise ValueError(f"num_tracking_points_ropes must be > 0 or == -1 (to use them all). Received {self.num_tracking_points_ropes}.")

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
        in ``maximum_state_velocity`` in [angle, angle, angle, mm] / step.
        and further to per second (because delta T is not 1 second).
        """
        return self.time_step * self.maximum_state_velocity * action

    def _scale_discrete_action(self, action: int) -> np.ndarray:
        """Maps action indices to a motion delta."""
        return self._discrete_action_lookup[action]

    def _do_action(self, action: np.ndarray) -> None:
        """Apply action to the simulation."""
        if abs(action[-1]) > self.activation_deadzone:
            self.cauter.set_activation(True if action[-1] > 0 else False)
        self.cauter.set_state(self.cauter.get_state() + self._scale_action(action[:-1]))

    def _update_rope_buffers(self) -> None:
        """Read the current rope positions and topological changes.

        Writes the results into ``self.rope_positions`` and ``self.rope_topological_changes``.

        Notes:
            - By cutting, some of the points of the ropes could drift very far away.
            The increased bounding box may reduce simulation accuracy or worse, crash the simulation.
            Checks, if any points of the ropes are far away from the actual scene, and set their positions to zero.
            - If points are removed, the missing point positions are filled with the corner point of the workspace

        TODO:
            - Benchmark what is faster in this context: a) assign the existing array, or b) write into the buffer.
        """

        # Retrieve rope positions and topological changes of the ropes
        rope_positions = np.empty((self.num_ropes, self.rope_points, 3), dtype=np.float32)
        rope_topology_changes = np.empty((self.num_ropes), dtype=np.int16)
        for index, rope in enumerate(self.ropes):
            # If necessary, forces points that floated far away from the scene to the corner position.
            read_positions = rope.get_positions()
            faraway_indices = np.where(np.linalg.norm(read_positions, axis=-1) > 10 / self._distance_normalization_factor)[0]
            if len(faraway_indices) > 0:
                rope.set_positions(self._corner_position, faraway_indices)
                # NOTE: This set_positions technically also updates read_positions, because it is the same array.
                # However this behavior relies on the current implementation of array() and writeable() in SofaPython3.
                # Making a copy here is a bit slower, but it is safer and less prone to errors, should the SofaPython3 implementation change.
                # Make read positions writeable
                read_positions = read_positions.copy()
                read_positions[faraway_indices] = self._corner_position

            if read_positions.shape[0] < self.rope_points:
                # If points are removed, the missing point positions are filled with the corner point of the workspace
                all_positions = np.repeat(self._corner_position[None, :], self.rope_points, axis=0)
                all_positions[: read_positions.shape[0]] = read_positions
                rope_positions[index, :, :] = all_positions
            else:
                rope_positions[index, :, :] = read_positions

            rope_topology_changes[index] = rope.consume_topology_change_buffer()

        # Save the rope positions for eventual reuse in _get_observation()
        self.rope_position_buffer = rope_positions
        self.rope_topology_change_buffer = rope_topology_changes

    def _get_reward_features(self, previous_reward_features: dict) -> dict:
        """Get the features that may be used to assemble the reward function

        Features:
            - distance_cauter_active_rope (float): Minimal distance between the cauter and the active rope.
            - delta_distance_cauter_active_rope (float): Change in distance between the cauter and the active rope.
            - cut_active_rope (int): Sum of topological changes of the active rope.
            - cut_inactive_rope (int): Sum of topological changes of the inactive ropes.
            - workspace_violation (float): 1.0 if the cauter action would have violated the workspace, 0.0 otherwise.
            - state_limits_violation (float): 1.0 if the cauter action would have violated the state limits, 0.0 otherwise.
        """

        reward_features = {}

        # Read the current rope positions and topological changes.
        self._update_rope_buffers()

        cauter_cutting_center_position = self.cauter.get_cutting_center_position()

        # Distance between cauter tip and the rope to cut
        reward_features["distance_cauter_active_rope"] = np.min(np.linalg.norm(self.rope_position_buffer[self.active_rope_index, :, :] - cauter_cutting_center_position, axis=1))
        reward_features["delta_distance_cauter_active_rope"] = reward_features["distance_cauter_active_rope"] - previous_reward_features["distance_cauter_active_rope"]

        # Cut on the active rope
        reward_features["cut_active_rope"] = self.rope_topology_change_buffer[self.active_rope_index] > 0

        # Cut on non-active ropes
        reward_features["inactive_cut_rope_indices"] = []
        reward_features["cut_inactive_rope"] = 0
        for index, change in enumerate(self.rope_topology_change_buffer):
            if index == self.active_rope_index:
                continue
            else:
                # Check if the rope was cut, and it is still in the indices.
                # If the rope was already cut, but the cauter removes points from the remains, we do not add that to the reward
                if change > 0 and index in self.rope_indices:
                    reward_features["cut_inactive_rope"] += 1
                    reward_features["inactive_cut_rope_indices"].append(index)

        # State and workspace limits
        reward_features["workspace_violation"] = float(self.cauter.last_set_state_violated_workspace_limits)
        reward_features["state_limits_violation"] = float(self.cauter.last_set_state_violated_state_limits)

        return reward_features

    def _get_reward(self) -> float:
        """Retrieve the reward features and scale them with the ``reward_amount_dict``."""
        reward = 0.0
        self.reward_info = {}

        reward_features = self._get_reward_features(self.reward_features)

        # Update the remaining ropes
        if reward_features["cut_inactive_rope"] > 0:
            self.num_cut_inactive_ropes += reward_features["cut_inactive_rope"]
            for index in reward_features["inactive_cut_rope_indices"]:
                self.rope_indices.remove(index)
        reward_features["inactive_cut_rope_indices"] = 0.0

        # Chose a new active rope
        if reward_features["cut_active_rope"]:
            self.num_cut_ropes += 1
            self.rope_indices.remove(self.active_rope_index)
            if len(self.rope_indices) > 0:
                self.active_rope_index = self.rng.choice(self.rope_indices)
            for index, rope in enumerate(self.ropes):
                if index == self.active_rope_index:
                    rope.set_color(self.active_rope_color)
                else:
                    rope.set_color(self.inactive_rope_color)

        # Successful task if desired number of ropes is cut
        if self.num_cut_ropes >= self.num_ropes_to_cut:
            reward_features["successful_task"] = True
        else:
            reward_features["successful_task"] = False

        # Failed task if there are more ropes to cut than left in the scene
        if self.num_ropes_to_cut - self.num_cut_ropes > len(self.rope_indices):
            reward_features["failed_task"] = True
        else:
            reward_features["failed_task"] = False

        # We change the values of the dict -> do a copy (deepcopy not necessary, because the value itself is not manipulated)
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
        return self.reward_features["successful_task"] or self.reward_features["failed_task"]

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
            state_dict["ptsd_state"] = self.cauter.get_state()
            state_dict["active"] = np.asarray(self.cauter.active)[None]  # 1 -> [1]
            state_dict["pose"] = self.cauter.get_physical_pose().copy()

            # Reuse the rope positions from the reward function
            rope_positions = self.rope_position_buffer[:, self.ropes_tracking_point_indices, :]

            # normalize values between -1 and 1
            if self.normalize_observations:
                state_dict["ptsd_state"] = 2 * (state_dict["ptsd_state"] - self.cauter.state_limits["low"]) / (self.cauter.state_limits["high"] - self.cauter.state_limits["low"]) - 1
                state_dict["pose"][:3] = 2 * (state_dict["pose"][:3] - self.cauter.cartesian_workspace["low"]) / (self.cauter.cartesian_workspace["high"] - self.cauter.cartesian_workspace["low"]) - 1
                rope_positions = 2 * (rope_positions - self.cauter.cartesian_workspace["low"]) / (self.cauter.cartesian_workspace["high"] - self.cauter.cartesian_workspace["low"]) - 1

            state_dict["rope_positions"] = rope_positions.ravel()
            state_dict["active_rope_positions"] = rope_positions[self.active_rope_index].ravel()

            observation = np.concatenate(tuple(state_dict.values()))
            # Overwrite possible NaNs with the maximum distance in the workspace
            observation = np.where(np.isnan(observation), 1.0 / self._distance_normalization_factor, observation)

        return observation

    def _get_info(self) -> dict:
        """Assemble the info dictionary."""

        self.info = {}
        self.info["num_cut_ropes"] = self.num_cut_ropes
        self.info["num_cut_inactive_ropes"] = self.num_cut_inactive_ropes

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

        if self._initialized:
            self.sofa_simulation.unload(self._sofa_root_node)
            if hasattr(self, "_window"):
                super().close()
            self._init_sim()

        super().reset(seed)

        # Seed the instrument
        if self.unconsumed_seed:
            seeds = self.seed_sequence.spawn(1)
            self.cauter.seed(seed=seeds[0])
            self.unconsumed_seed = False

        self.active_rope_index = self.rng.integers(0, self.num_ropes)
        self.num_cut_ropes = 0
        self.num_cut_inactive_ropes = 0
        self.rope_indices = list(range(self.num_ropes))
        for index, rope in enumerate(self.ropes):
            if index == self.active_rope_index:
                rope.set_color(self.active_rope_color)
            else:
                rope.set_color(self.inactive_rope_color)

        self.cauter.reset_cauter()

        # Clear the episode info values
        for key in self.episode_info:
            self.episode_info[key] = 0.0

        # and the reward info dict
        self.reward_info = {}

        # and fill the first values used as previous_reward_features
        self.reward_features = {}

        # Read the current rope positions and topological changes.
        self._update_rope_buffers()

        cauter_cutting_center_position = self.cauter.get_cutting_center_position()
        self.reward_features["distance_cauter_active_rope"] = np.min(np.linalg.norm(self.rope_position_buffer[self.active_rope_index, :, :] - cauter_cutting_center_position, axis=1))

        # Execute any callbacks passed to the env.
        for callback in self.on_reset_callbacks:
            callback(self)

        # Animate several timesteps without actions until simulation settles
        for _ in range(self._settle_steps):
            self.sofa_simulation.animate(self._sofa_root_node, self._settle_step_dt)
            self._maybe_update_rgb_buffer()

        return self._get_observation(maybe_rgb_observation=self._maybe_update_rgb_buffer()), {}


if __name__ == "__main__":
    import pprint
    import time

    pp = pprint.PrettyPrinter()

    env = RopeCuttingEnv(
        observation_type=ObservationType.STATE,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.CONTINUOUS,
        image_shape=(800, 800),
        frame_skip=1,
        time_step=0.1,
        settle_steps=10,
        settle_step_dt=0.1,
    )

    env.reset()
    done = False

    fps_list = deque(maxlen=100)
    while not done:
        for _ in range(100):
            start = time.perf_counter()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            end = time.perf_counter()
            fps = 1 / (end - start)
            fps_list.append(fps)
            # pp.pprint(info)
            print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")

        env.reset()
