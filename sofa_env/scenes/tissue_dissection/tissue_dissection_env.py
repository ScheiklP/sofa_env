import gymnasium.spaces as spaces
import numpy as np

from collections import defaultdict, deque
from enum import Enum, unique
from pathlib import Path
from functools import reduce

from typing import Callable, Union, Tuple, Optional, List, Any, Dict
from sofa_env.base import SofaEnv, RenderMode, RenderFramework

from sofa_env.scenes.tissue_dissection.sofa_objects.cauter import PivotizedCauter
from sofa_env.sofa_templates.deformable import SimpleDeformableObject
from sofa_env.sofa_templates.camera import Camera
from sofa_env.utils.math_helper import farthest_point_sampling


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


class TissueDissectionEnv(SofaEnv):
    """Tissue Dissection Environment

    The goal of this environment is dissecting a tissue with a dissection electrode. The red tissue is connected to a rigid board through
    blue connective tissue. The goal is to cut the blue connective tissue without damaging the red tissue.

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
        camera_reset_noise (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Limits to uniformly sample noise that is added to the camera's initial state at reset.
        activation_deadzone (float): Defines the deadzone in which the action for changing the state of the cauter (on/off) is ignored.
        with_board_collision (bool): Whether to add a collision model to the board. This will influence the behavior of the cauter whenn touching the board.
        With a collision model, the remote center of motion will be violated, when the cauter tries to move into the board and the cauter will be deflected.
        rows_to_cut (int): How many rows along the tissue should be simulated by cuttable connective tissue.
        The tissue is 10 rows long and the first 2 rows are not attached to the board. The rest of the tissue is connected to the board.
        ``rows_to_cut`` controls how many of the remaining rows are connected by deformable, cuttable connective tissue. The rest is fixed to the board.
        num_tracking_points_tissue (int): How many points on the tissue to add to the state observation.
        num_tracking_points_connective_tissue (int): How many points on the connective tissue to add to the state observation.
        randomize_cauter (bool): Whether to randomize the cauter's initial state at reset.
        control_retraction_force (bool): Whether to control the retraction force on the tissue in the action space.
        The action space is extended by 3 values that add a delte to the XYZ values of the force applied on the tissue.
    """

    def __init__(
        self,
        scene_path: Union[str, Path] = SCENE_DESCRIPTION_FILE_PATH,
        image_shape: Tuple[int, int] = (124, 124),
        observation_type: ObservationType = ObservationType.RGB,
        time_step: float = 0.01,
        frame_skip: int = 1,
        settle_steps: int = 50,
        render_mode: RenderMode = RenderMode.HEADLESS,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        reward_amount_dict: Dict[str, float] = {
            "unstable_deformation": -0.0,
            "distance_cauter_border_point": -0.0,
            "delta_distance_cauter_border_point": -0.0,
            "cut_connective_tissue": 0.0,
            "cut_tissue": -0.0,
            "workspace_violation": -0.0,
            "state_limits_violation": -0.0,
            "rcm_violation_xyz": -0.0,
            "rcm_violation_rpy": -0.0,
            "collision_with_board": -0.0,
            "successful_task": 0.0,
        },
        maximum_state_velocity: Union[np.ndarray, float] = np.array([20.0, 20.0, 30.0, 20.0]),
        discrete_action_magnitude: Union[np.ndarray, float] = np.array([15.0, 15.0, 15.0, 10.0]),
        create_scene_kwargs: Optional[dict] = None,
        on_reset_callbacks: Optional[List[Callable]] = None,
        action_type: ActionType = ActionType.CONTINUOUS,
        camera_reset_noise: Optional[np.ndarray] = None,
        activation_deadzone: float = 0.1,
        with_board_collision: bool = True,
        rows_to_cut: int = 2,
        num_tracking_points_tissue: int = 10,
        num_tracking_points_connective_tissue: int = 10,
        randomize_cauter: bool = True,
        control_retraction_force: bool = False,
    ) -> None:
        # Pass image shape to the scene creation function
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape
        create_scene_kwargs["with_board_collision"] = with_board_collision
        create_scene_kwargs["rows_to_cut"] = rows_to_cut

        self.randomize_cauter = randomize_cauter
        if not randomize_cauter:
            create_scene_kwargs["cauter_reset_noise"] = None

        self.with_board_collision = with_board_collision
        self.rows_to_cut = rows_to_cut

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

        self.activation_deadzone = activation_deadzone
        self.maximum_state_velocity = maximum_state_velocity

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
        action_dimensionality = 8 if control_retraction_force else 5
        self.action_type = action_type
        self.control_retraction_force = control_retraction_force
        if action_type == ActionType.CONTINUOUS:
            self._scale_action = self._scale_continuous_action
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dimensionality,), dtype=np.float32)
        else:
            self._scale_action = self._scale_discrete_action
            self.action_space = spaces.Discrete(action_dimensionality * 2 + 1)
            if control_retraction_force:
                raise NotImplementedError("Discrete actions are currently not supported for controlling the retraction force.")

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
        # State observations
        self.num_tracking_points_tissue = num_tracking_points_tissue
        self.num_tracking_points_connective_tissue = num_tracking_points_connective_tissue
        if observation_type == ObservationType.STATE:
            # ptsda_state -> 5
            # cauter_pose -> 7
            # tracking_points_tissue -> 3 * num_tracking_points_tissue
            # tracking_points_connective_tissue -> 3 * num_tracking_points_connective_tissue
            # border_point -> 3
            # retraction_force if control_retraction_force else 0
            observations_size = 5 + 7 + 3 * num_tracking_points_tissue + 3 * num_tracking_points_connective_tissue + 3 + 3 * control_retraction_force
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

        self.cauter: PivotizedCauter = self.scene_creation_result["cauter"]
        self.tissue: SimpleDeformableObject = self.scene_creation_result["tissue"]
        self.connective_tissue: SimpleDeformableObject = self.scene_creation_result["connective_tissue"]
        self.camera: Camera = self.scene_creation_result["camera"]
        self.contact_listener = self.scene_creation_result["contact_listener"]
        self.border_point_mechanical_object = self.scene_creation_result["border_point_mechanical_object"]
        self.retraction_force = self.scene_creation_result["retraction_force"]
        self.topology_info: dict = self.scene_creation_result["topology_info"]

        # Factor for normalizing the distances in the reward function.
        # Based on the workspace of the gripper.
        self._distance_normalization_factor = 1.0 / np.linalg.norm(self.cauter.cartesian_workspace["high"] - self.cauter.cartesian_workspace["low"])

        # Identify the indices of points on the (connective) tissue that should be used for describing the state of the scene.
        tissue_points = self.tissue.mechanical_object.position.array()
        if len(tissue_points) < self.num_tracking_points_tissue:
            raise ValueError(f"Cannot track {self.num_tracking_points_tissue} points on the tissue. Only {len(tissue_points)} points are available.")
        self.tissue_tracking_point_indices = farthest_point_sampling(
            points=tissue_points,
            num_samples=self.num_tracking_points_tissue,
            return_indices=True,
        )

        connective_tissue_points = self.connective_tissue.mechanical_object.position.array()
        if len(connective_tissue_points) < self.num_tracking_points_connective_tissue:
            raise ValueError(f"Cannot track {self.num_tracking_points_connective_tissue} points on the connective_tissue. Only {len(connective_tissue_points)} points are available.")
        self.connective_tissue_tracking_point_indices = farthest_point_sampling(
            points=connective_tissue_points,
            num_samples=self.num_tracking_points_connective_tissue,
            return_indices=True,
        )

    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        """Step function of the environment that applies the action to the simulation and returns observation, reward, done signal, and info."""

        maybe_rgb_observation = super().step(action)

        reward = self._get_reward()
        # Calculate reward first, to update the tracking indices
        observation = self._get_observation(maybe_rgb_observation=maybe_rgb_observation)
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
        if abs(action[4]) > self.activation_deadzone:
            self.cauter.set_activation(True if action[-1] > 0 else False)
        self.cauter.set_state(self.cauter.get_state() + self._scale_action(action[:4]))
        if self.control_retraction_force:
            with self.retraction_force.totalForce.writeable() as force:
                new_force = np.clip(force + action[5:] * 3e2, a_min=-1e5, a_max=1e5)
                force[:] = new_force

    def _get_reward_features(self, previous_reward_features: dict) -> dict:
        """Get the features that may be used to assemble the reward function

        Features:
            - unstable_deformation (bool): Whether the tissue is unstable (unstable simulation).
            - distance_cauter_border_point (float): Minimal distance between the cauter and the border point of the connective tissue.
            - delta_distance_cauter_border_point (float): Change in distance between the cauter and the border point of the connective tissue.
            - cut_connective_tissue (int): Number of tetrahedra  of the connective tissue that were cut since the last step.
            - cut_tissue (int): Number of tetrahedra of the tissue that were cut since the last step.
            - workspace_violation (float): 1.0 if the cauter action would have violated the workspace, 0.0 otherwise.
            - state_limits_violation (float): 1.0 if the cauter action would have violated the state limits, 0.0 otherwise.
            - rcm_violation_xyz (float): Cartesian difference between desired and actual cauter position.
            - rcm_violation_rpy (float): Rotation angle difference between desired and actual cauter orientation in Degrees.
            - collision_with_board (Union[bool, int]): Whether the cauter collided with the board (or number of collisions, if ``self.with_board_collision``).
            - successful_task (bool): Whether the task was successful in removing all tetrahedra of the connective tissue.
        """

        reward_features = {}

        cauter_position = self.cauter.get_cutting_tip_position()
        connective_tissue_positions = self.connective_tissue.mechanical_object.position.array()
        tissue_positions = self.tissue.mechanical_object.position.array()

        nonempty_connective_tissue = len(self.connective_tissue.mechanical_object.position) > 0 and len(self.connective_tissue.topology_container.tetrahedra) > 0
        nonempty_tissue = len(tissue_positions) > 0 and len(self.tissue.topology_container.tetrahedra) > 0

        # By cutting, some of the points of the (connective) tissue could drift very far away.
        # The increased bounding box may reduce simulation accuracy or worse, crash the simulation.
        # Check, if any points of the (connective) tissue are far away from the actual scene, and set their positions to zero.
        faraway_indices_tissue = np.where(np.linalg.norm(tissue_positions, axis=1) > 10 / self._distance_normalization_factor)[0]
        faraway_indices_connective_tissue = np.where(np.linalg.norm(connective_tissue_positions, axis=1) > 10 / self._distance_normalization_factor)[0]

        if len(faraway_indices_tissue) > 0:
            print(f"WARNING: {len(faraway_indices_tissue)} tissue points are far away from the scene. Setting their positions to [0, 0, 0].")
            with self.tissue.mechanical_object.position.writeable() as writable_array:
                writable_array[faraway_indices_tissue] = [0.0, 0.0, 0.0]

        if len(faraway_indices_connective_tissue) > 0:
            print(f"WARNING: {len(faraway_indices_connective_tissue)} points of the connective tissue are far away from the actual scene. Setting their positions to [0, 0, 0].")
            with self.connective_tissue.mechanical_object.position.writeable() as writable_array:
                writable_array[faraway_indices_connective_tissue] = [0.0, 0.0, 0.0]

        # Is the connective tissue unstable? 500 is an empirical value.
        reward_features["unstable_deformation"] = np.amax(self.connective_tissue.mechanical_object.velocity.array()) > 500 if nonempty_connective_tissue > 0 else False

        # Heuristically, the next point to cut is the one that is furthest in y direction (direction of the board).
        if nonempty_connective_tissue:
            border_points = connective_tissue_positions[connective_tissue_positions.max(axis=0)[1] == connective_tissue_positions[:, 1]]
            min_distance_index = np.argmin(np.linalg.norm(border_points - cauter_position, axis=1))
            border_point = border_points[min_distance_index]
            reward_features["distance_cauter_border_point"] = np.linalg.norm(cauter_position - border_point) * self._distance_normalization_factor
            reward_features["delta_distance_cauter_border_point"] = reward_features["distance_cauter_border_point"] - previous_reward_features["distance_cauter_border_point"]
            # Visualization of the border point.
            with self.border_point_mechanical_object.position.writeable() as position:
                position[0, :3] = border_point

        else:
            reward_features["distance_cauter_border_point"] = 0.0
            reward_features["delta_distance_cauter_border_point"] = 0.0

        # Cut connective tissue
        reward_features["num_tetra_in_connective_tissue"] = len(self.connective_tissue.topology_container.tetrahedra)
        reward_features["cut_connective_tissue"] = previous_reward_features["num_tetra_in_connective_tissue"] - reward_features["num_tetra_in_connective_tissue"]
        # Update the tracking indices
        if reward_features["cut_connective_tissue"] > 0 and nonempty_connective_tissue:
            self.connective_tissue_tracking_point_indices = farthest_point_sampling(
                points=connective_tissue_positions,
                num_samples=min(self.num_tracking_points_connective_tissue, len(connective_tissue_positions)),
                return_indices=True,
            )

        # Cut tissue
        reward_features["num_tetra_in_tissue"] = len(self.tissue.topology_container.tetrahedra)
        reward_features["cut_tissue"] = previous_reward_features["num_tetra_in_tissue"] - reward_features["num_tetra_in_tissue"]
        # Update the tracking indices
        if reward_features["cut_tissue"] > 0 and nonempty_tissue:
            self.tissue_tracking_point_indices = farthest_point_sampling(
                points=tissue_positions,
                num_samples=min(self.num_tracking_points_tissue, len(tissue_positions)),
                return_indices=True,
            )

        # State and workspace limits
        reward_features["workspace_violation"] = float(self.cauter.last_set_state_violated_workspace_limits)
        reward_features["state_limits_violation"] = float(self.cauter.last_set_state_violated_state_limits)
        rcm_difference = self.cauter.get_pose_difference(position_norm=True)
        reward_features["rcm_violation_xyz"] = rcm_difference[0] * self._distance_normalization_factor
        reward_features["rcm_violation_rotation"] = rcm_difference[1]

        # Collision with the board
        if self.with_board_collision:
            reward_features["collision_with_board"] = self.contact_listener.getNumberOfContacts()
        else:
            # very simple approximation of board collision based on the cauter height
            reward_features["collision_with_board"] = cauter_position[2] < 0.0

        reward_features["successful_task"] = len(self.connective_tissue.topology_container.tetrahedra) == 0

        return reward_features

    def _get_reward(self) -> float:
        """Retrieve the reward features and scale them with the ``reward_amount_dict``."""
        reward = 0.0
        self.reward_info = {}

        reward_features = self._get_reward_features(self.reward_features)
        # we change the values of the dict -> do a copy (deepcopy not necessary, because the value itself is not manipulated)
        self.reward_features = reward_features.copy()

        self.episode_info["total_cut_connective_tissue"] += reward_features["cut_connective_tissue"]
        self.episode_info["total_cut_tissue"] += reward_features["cut_tissue"]
        self.episode_info["total_unstable_deformation"] += reward_features["unstable_deformation"]
        self.episode_info["ratio_cut_connective_tissue"] = self.episode_info["total_cut_connective_tissue"] / self.initial_tetra_in_connective_tissue
        self.episode_info["ratio_cut_tissue"] = self.episode_info["total_cut_tissue"] / self.initial_tetra_in_tissue

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
            state_dict["ptsd_state"] = self.cauter.get_state()
            state_dict["active"] = np.asarray(self.cauter.active)[None]  # 1 -> [1]
            state_dict["cauter_pose"] = self.cauter.get_pose()
            tissue_positions = self.tissue.mechanical_object.position.array()
            connective_tissue_positions = self.connective_tissue.mechanical_object.position.array()
            if len(tissue_positions):
                state_dict["tissue_tracking_point_positions"] = tissue_positions[self.tissue_tracking_point_indices].ravel()
            if len(connective_tissue_positions):
                state_dict["connective_tissue_tracking_point_positions"] = connective_tissue_positions[self.connective_tissue_tracking_point_indices].ravel()
            state_dict["border_point"] = self.border_point_mechanical_object.position.array()[0, :3]

            if self.control_retraction_force:
                state_dict["retraction_force"] = self.retraction_force.totalForce.array()

            observation_stack = np.concatenate(tuple(state_dict.values()))
            # We could have fewer points left than specified in the number of tracking points -> fill the rest with zeros
            observation[: len(observation_stack)] = observation_stack
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
            seeds = self.seed_sequence.spawn(1)
            self.cauter.seed(seed=seeds[0])
            self.unconsumed_seed = False

        # Reset the gripper
        self.cauter.reset_cauter()
        if self.control_retraction_force:
            with self.retraction_force.totalForce.writeable() as force:
                force[:] = [0, -1e5, 1e5]

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
        self.reward_features["num_tetra_in_connective_tissue"] = len(self.connective_tissue.topology_container.tetrahedra)
        self.reward_features["num_tetra_in_tissue"] = len(self.tissue.topology_container.tetrahedra)
        self.initial_tetra_in_tissue = self.reward_features["num_tetra_in_tissue"]
        self.initial_tetra_in_connective_tissue = self.reward_features["num_tetra_in_connective_tissue"]

        # Execute any callbacks passed to the env.
        for callback in self.on_reset_callbacks:
            callback(self)

        # Animate several timesteps without actions until simulation settles
        for _ in range(self._settle_steps):
            self.sofa_simulation.animate(self._sofa_root_node, self._sofa_root_node.getDt())

        connective_tissue_positions = self.connective_tissue.mechanical_object.position.array()
        nonempty_connective_tissue = len(connective_tissue_positions) > 0 and len(self.connective_tissue.topology_container.tetrahedra.array()) > 0

        if nonempty_connective_tissue:
            # Heuristically, the next point to cut is the one that is furthest in y direction (direction of the board).
            cauter_position = self.cauter.get_cutting_tip_position()
            border_points = connective_tissue_positions[connective_tissue_positions.max(axis=0)[1] == connective_tissue_positions[:, 1]]
            min_distance_index = np.argmin(np.linalg.norm(border_points - cauter_position, axis=1))
            border_point = border_points[min_distance_index]
            self.reward_features["distance_cauter_border_point"] = np.linalg.norm(cauter_position - border_point) * self._distance_normalization_factor
        else:
            self.reward_features["distance_cauter_border_point"] = 0.0

        return self._get_observation(maybe_rgb_observation=self._maybe_update_rgb_buffer()), {}


if __name__ == "__main__":
    import pprint
    import time

    pp = pprint.PrettyPrinter()

    env = TissueDissectionEnv(
        observation_type=ObservationType.STATE,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.CONTINUOUS,
        image_shape=(800, 800),
        frame_skip=3,
        time_step=0.01,
        settle_steps=50,
        control_retraction_force=True,
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
            if counter == 200:
                env.reset()
                counter = 0
            counter += 1
            end = time.perf_counter()
            fps = 1 / (end - start)
            fps_list.append(fps)
            # pp.pprint(info)
            print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")

        env.reset()
