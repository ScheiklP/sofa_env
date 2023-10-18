import cv2
import gymnasium as gym
import gymnasium.spaces as spaces
import time
import pprint
import numpy as np

from enum import Enum, unique
from pathlib import Path
from collections import deque
from typing import Union, Tuple, Optional, Any, List, Dict

from sofa_env.base import SofaEnv, RenderMode, RenderFramework
from sofa_env.utils.camera import world_to_pixel_coordinates

from sofa_env.scenes.tissue_manipulation.sofa_robot_functions import WorkspaceType, Workspace
from sofa_env.scenes.tissue_manipulation.scene_description import COLOR_VIS_TARGET
from sofa_env.scenes.tissue_manipulation.sofa_objects.gripper import AttachedGripper
from sofa_env.scenes.tissue_manipulation.sofa_objects.rigidified_tissue import Tissue
from sofa_env.scenes.tissue_manipulation.sofa_objects.visual_target import VisualTarget
from sofa_env.scenes.tissue_manipulation.utils.logging import TissueManipulationSceneLogger

SCENE_DESCRIPTION_FILE_PATH = Path(__file__).resolve().parent / "scene_description.py"
DT = 0.1
VEL = 0.002  # 2 mm/s


@unique
class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


@unique
class ObservationType(Enum):
    RGB = 0
    STATE = 1
    RGBD = 2


@unique
class EpisodeEndCriteria(Enum):
    """Four criteria to evaluate if an Episode is *DONE*

    Notes:
        - values: Tuple[name: str, if_indicator: bool]
        - special case: STUCK
            - values: Tuple[name: str, if_indicator: bool, steps: int, distance_delta: float, active_after_x_episodes: int]
    """

    SUCCESS = ("goal_reached", True)
    STABILITY = ("stable_deformation", False)
    STUCK = ("is_stuck", True, 50, 5 * 1e-4, 0)  # check for delta in distance to target


class TissueManipulationEnv(SofaEnv):
    """Tissue Manipulation Environment

    The goal of this environment is to manipulate a tissue with a robot gripper such that a visual manipulation target on the tissue is
    moved to overlap with a visual target in the image observation.

    Notes:
        - Nomenclature:
            - Tissue Target Point (TTP) / manipulation target: point on the deformable tissue, which is the target for alignment with a goal point.
            - Tissue Grasping Point (TGP): point on the tissue, where the gripper is attached.
            - Image Desired Point (IDP) / visual target: point inside workspace, which represents the goal position for the TTP.
        - Scene consists of a deformable tissue with a rigidified part at the top where a gripper is attached. The tissue is fixated in the liver.
        - Metrics:
            - "is_success": success of current episode; either 1 or 0
            - "motion_efficiency": (# steps in current episode) / (initial distance to target in mm)
            - "motion_smoothness":  (Mean Squared 2nd order gradient of trajectory [mm]) / (initial distance(IDP, TTP) [m])

    Args:
        scene_path (Union[str, Path]): Path to the scene description script that contains this environment's ``createScene`` function.
        image_shape (Tuple[int, int]): Height and Width of the rendered images.
        observation_type (ObservationType): Whether to return RGB images or an array of states as the observation.
        time_step (float): Size of simulation time step in seconds (default: 0.01).
        frame_skip (int): Number of simulation time steps taken (call ``_do_action`` and advance simulation) each time step is called (default: 1).
        settle_steps (int): How many steps to simulate without returning an observation after resetting the environment.
        action_type (ActionType): Specify discrete or continuous action space
        state_observation_space (Optional[gymnasium.Space]): Specify dimensions of state space. Leave empty for automatic detection.
        render_mode (RenderMode): create a window (``RenderMode.HUMAN``) or run headless (``RenderMode.HEADLESS``).
        render_framework (RenderFramework): choose between pyglet and pygame for rendering
        action_space (Optional[gymnasium.spaces.Box]): An optional ``Box`` action space to set the limits for clipping.
        end_position_threshold (float): Distance to the ``end_position`` at which episode success is triggered (in meters [m]).
        maximum_robot_velocity (float): Maximum velocity of TGP in millimeters per second [mm/s].
        squared_distance (float): Use squared distance in loss calculation.
        distance_calculation_2d (bool): Calculate distance in XZ-coordinates only. Distanced projected to workspace plane.
        reward_amount_dict (dict): Dictionary to weigh the components of the reward function.
        create_scene_kwargs (Optional[dict]): A dictionary to pass additional keyword arguments to the ``createScene`` function.
        end_episode_criteria (list(EpisodeEndCriteria)): List of EpisodeEndCriterias, which specify when a episode is *DONE*".
        minimum_target_distance (float): Minimum initial distance between TTP/TGP and IDP in meters [m].
        log_episode_env_info (bool): Log TTP, TGP, IDP, Distance in separate file.
        log_path (str): Path to where the episode env info is logged.
        deactivate_overlay (bool): Deactivates the overlay of the IDP. Activate this option for sim2real.
        debug (bool): Activates debug rendering with cv2.
        print_init_info (bool): Print information about the environment at initialization.
    """

    def __init__(
        self,
        scene_path: Union[str, Path] = SCENE_DESCRIPTION_FILE_PATH,
        image_shape: Tuple[int, int] = (128, 128),
        observation_type: ObservationType = ObservationType.RGB,
        time_step: float = 0.01,
        frame_skip: int = 1,
        settle_steps: int = 20,
        action_type: ActionType = ActionType.CONTINUOUS,
        render_mode: RenderMode = RenderMode.HEADLESS,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        state_observation_space: Optional[gym.Space] = None,
        visual_target_position: Optional[np.ndarray] = None,
        action_space: Optional[gym.Space] = None,
        discrete_action_magnitude: Optional[float] = 0.5,
        end_position_threshold: float = 0.002,
        maximum_robot_velocity: float = 1.0,
        stability_threshold: float = 1e-3 * 2,
        squared_distance: bool = False,
        distance_calculation_2d: bool = True,
        reward_amount_dict: Dict[str, float] = {
            "distance_to_target": -0.0,
            "one_time_reward_goal": 0.0,
            "one_time_penalty_is_stuck": -0.0,
            "one_time_penalty_invalid_action": -0.0,
            "one_time_penalty_unstable_simulation": -0.0,
        },
        create_scene_kwargs: Optional[dict] = None,
        end_episode_criteria: Optional[List[EpisodeEndCriteria]] = None,
        minimum_target_distance: float = 0.01,
        log_episode_env_info: bool = True,
        log_path: Optional[str] = None,
        deactivate_overlay: bool = False,
        debug: bool = False,
        print_init_info: bool = False,
    ) -> None:
        # Pass image shape to the scene creation function
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape
        self.image_shape = image_shape

        self._deactivate_overlay = deactivate_overlay
        self._debug = debug
        if debug:
            render_mode = RenderMode.HEADLESS
            print("[DEBUG] Overwriting RenderMode --> HEADLESS")
        super().__init__(
            scene_path=scene_path,
            time_step=time_step,
            frame_skip=frame_skip,
            render_mode=render_mode,
            render_framework=render_framework,
            create_scene_kwargs=create_scene_kwargs,
        )

        self.print_init_info = print_init_info

        ##############
        # Action Space
        ##############
        self._workspace_type = self.create_scene_kwargs.get("workspace_kwargs", {}).get("workspace_type", WorkspaceType.TISSUE_ALIGNED)
        dim_action = self._workspace_type.value[-1]
        if action_type == ActionType.CONTINUOUS:
            if action_space is None:
                action_space = spaces.Box(low=-1.0, high=1.0, shape=(dim_action,), dtype=np.float64)
            else:
                if type(action_space) != spaces.Box:
                    raise ValueError(f"Specified wrong action_space {action_space} for action_type == ActionType.CONTINUOUS")
                if action_space.shape != (dim_action,):
                    raise ValueError(f"Specified wrong action dim {action_space.shape} for workspace dim = {(dim_action,)}")
            self._scale_action = self._scale_normalized_action

        elif action_type == ActionType.DISCRETE:
            if action_space is None:
                action_space = spaces.Discrete(dim_action * 2 + 1)
            else:
                if not (isinstance(action_space, spaces.Discrete) and action_space.n == dim_action * 2 + 1):
                    raise ValueError(f"If setting a manual discrete action space, please pass it as a gymnasium.spaces.Discrete with {dim_action * 2 + 1} elements.")

            self._scale_action = self._scale_discrete_action

            if discrete_action_magnitude is None:
                raise ValueError("For discrete actions, please pass a step size.")

            # [step, 0, 0, ...], [-step, 0, 0, ...], [0, step, 0, ...], [0, -step, 0, ...]
            action_list = []
            for i in range(dim_action * 2):
                action = [0.0] * dim_action
                step_size = discrete_action_magnitude
                action[int(i / 2)] = (1 - 2 * (i % 2)) * step_size
                action_list.append(action)

            # Noop action
            action_list.append([0.0] * dim_action)

            self._discrete_action_lookup = np.array(action_list)

            self._discrete_action_lookup *= self.time_step * 0.001
            self._discrete_action_lookup.flags.writeable = False

        self.action_space = action_space
        self.action_type = action_type

        ###################
        # Observation Space
        ###################
        # State observations
        if observation_type == ObservationType.STATE:
            if state_observation_space is None:
                # TGP: 3
                # TTP: 3
                # IDP: 3
                dim_states = 3 + 3 + 3
                observation_space = spaces.Box(low=-1.0, high=1.0, shape=(dim_states,), dtype=np.float64)
            else:
                observation_space = state_observation_space

        # Image observations
        elif observation_type == ObservationType.RGB:
            observation_space = spaces.Box(low=0, high=255, shape=image_shape + (3,), dtype=np.uint8)

        # Depth Image observations
        elif observation_type == ObservationType.RGBD:
            observation_space = spaces.Box(low=0, high=255, shape=image_shape + (4,), dtype=np.uint8)

        else:
            raise Exception(f"Please set observation_type to a value of ObservationType. Received {observation_type}.")

        self.observation_space = observation_space
        self.observation_type = observation_type

        ##############
        # Placeholders
        ##############
        self._gripper: AttachedGripper
        self._visual_target: VisualTarget
        self._tissue: Tissue
        self._reward_scaling_factor: float
        self._workspace: Workspace
        self.T_S_R = None
        self.T_R_S = None

        #########
        # Logging
        #########
        self.log_episode_env_info = log_episode_env_info
        if log_episode_env_info:
            self._log_episode_info_flag = False
            self._initial_env_state = dict()
            self.logger = TissueManipulationSceneLogger(log_path=log_path)

        ######################
        # Task specific values
        ######################

        self._reward_amount_dict = reward_amount_dict
        # Criteria to determine when to end an episode
        self.end_episode_criteria = [EpisodeEndCriteria.SUCCESS] if end_episode_criteria is None else end_episode_criteria
        self._settle_steps = settle_steps
        self._maximum_robot_velocity = maximum_robot_velocity
        self._stability_threshold = stability_threshold  # 1 mm / dt
        self._minimum_target_distance = minimum_target_distance
        self._target_threshold = end_position_threshold

        #########################
        # Episode specific values
        #########################
        self._reset_counter = 0
        self._visual_target_position = visual_target_position
        self._current_step: int
        self._current_trajectory: List
        self._last_action_valid: bool
        self._last_action_norm: float
        self._last_delta_pos_norm: float
        self._last_observation: np.ndarray
        self._delta_distance = deque(maxlen=EpisodeEndCriteria.STUCK.value[2])
        self.ws_box_coordinates: Tuple[float, float, float, float]

        self.distance_calculation_2d = distance_calculation_2d
        if squared_distance:
            self._distance_exponent = 2
        else:
            self._distance_exponent = 1

    def _init_sim(self):
        """Initialise simulation and calculate values for reward scaling. Also prints additional scene information."""
        super()._init_sim()

        self._gripper: AttachedGripper = self.scene_creation_result["interactive_objects"]["gripper"]
        self._tissue: Tissue = self.scene_creation_result["interactive_objects"]["tissue"]
        self._visual_target: VisualTarget = self.scene_creation_result["interactive_objects"]["target"]
        self._workspace: Workspace = self.scene_creation_result["workspace"]
        self._sofa_camera = self.scene_creation_result["camera"].sofa_object

        # scale the rewards to an interval of [-1, 0]
        self._reward_scaling_factor = float(1.0 / np.linalg.norm(self._workspace.get_high() - self._workspace.get_low()))

        if self._visual_target_position is None:
            self._visual_target_position = np.array(self._visual_target.get_pose()[:3])
        else:
            self._visual_target.set_position(self._visual_target_position)

        if self.print_init_info:
            print(
                "\n--- Environment Settings --- \nDT:",
                self.time_step,
                "\nMax TCP velocity per coordinate [mm/s]:",
                self._maximum_robot_velocity,
                "\nFrames skipped between observations:",
                self.frame_skip,
                "\nObservation Frequency [Hz]:",
                1 / ((self.frame_skip + 1) * self.time_step),
                "\nTime between observations [s]:",
                (self.frame_skip + 1) * self.time_step,
                "\nMax TCP delta per coordinate between observations [mm]:",
                (self.frame_skip + 1) * self.time_step * self._maximum_robot_velocity,
                "\nRequired number of steps to overcome 50 mm of distance in one coordinate direction:",
                (50.0 / ((self.frame_skip + 1) * self.time_step * self._maximum_robot_velocity)),
                "\n",
            )

    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        """Step function of the environment that applies the action to the simulation and returns observation, reward, done signal, and info."""

        maybe_rgb_observation = super().step(action)
        self._current_step += 1

        observation = self._get_observation(maybe_rgb_observation=maybe_rgb_observation)
        reward = self._get_reward()
        terminated = self._get_done()
        info = self._get_info()

        if self._debug:
            self.debug_render(observation)

        return observation, reward, terminated, False, info

    def _get_new_target_position(self) -> np.ndarray:
        """Returns a new target position that is not too close to the current target position and within the reachable points.

        TODO:
            - Add support for WorkspaceType.GENERAL.
        """

        if not self._workspace_type == WorkspaceType.TISSUE_ALIGNED:
            # TODO: To support the GENERAL case, we need to generate a volumetric mesh of points
            # that are reachable by the robot and then sample from that.
            raise NotImplementedError("Only tissue aligned workspace is implemented.")

        try:
            from sofa_env.scenes.tissue_manipulation.sampling import POSITIONS, point_is_reachable
        except ImportError:
            raise RuntimeError(
                "Could not import POSITIONS from sofa_env.scenes.tissue_manipulation.sampling. \
                    Please make sure that POSITIONS were correctly created for the current workspace configuration."
            )

        # Get the id of the sampled manipulation point on the tissue.
        manipulation_point_index = self._tissue.target_idx_id

        # Get the position of the sampled manipulation point on the tissue.
        # This array is already valid, as we animate at least one step in reset.
        tissue_position = self._tissue.get_manipulation_target_pose()[:3]

        # Lookup the bounding box, in which potations target points lie
        min_points = np.array((np.min(POSITIONS[manipulation_point_index, :, 0]), np.min(POSITIONS[manipulation_point_index, :, 1])))
        max_points = np.array((np.max(POSITIONS[manipulation_point_index, :, 0]), np.max(POSITIONS[manipulation_point_index, :, 1])))

        sampling_attempts = 0
        found_valid_point = False
        candidate_point = np.empty(2)

        # Try finding a valid target point that lies withing the reachable polygon of the target point
        # and is at least self._minimum_target_distance away from the current TTP.
        while not found_valid_point:
            candidate_point = self.rng.uniform(low=min_points, high=max_points)
            sampling_attempts += 1
            distance_to_manipulation_point = np.linalg.norm(candidate_point - tissue_position[[0, 2]])
            if distance_to_manipulation_point > self._minimum_target_distance and point_is_reachable(candidate_point, manipulation_point_index):
                found_valid_point = True
            else:
                if sampling_attempts > 1000:
                    raise RuntimeError(
                        f"Could not find a valid point after 1000 attempts. \
                            Please check the sampling space and the minimum distance ({self._minimum_target_distance})."
                    )

        return np.array((candidate_point[0], tissue_position[1], candidate_point[1]))

    def reset(self, seed: Union[int, np.random.SeedSequence, None] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Union[np.ndarray, None], Dict]:
        super().reset(seed)

        # Seed the instruments
        if self.unconsumed_seed:
            seeds = self.seed_sequence.spawn(3)
            self._gripper.seed(seed=seeds[0])
            self._tissue.seed(seed=seeds[1])
            self._workspace.seed(seed=seeds[2])
            self.unconsumed_seed = False

        self._gripper.reset()
        self._tissue.reset()

        self._grasping_position = self._gripper.get_pose()[:3]
        self._visual_target_position = self._visual_target.get_pose()[:3]
        self._last_action_valid = True
        self._last_action_norm = 0.0
        self._last_delta_pos_norm = 0.0
        self._delta_distance = deque(maxlen=EpisodeEndCriteria.STUCK.value[2])

        if self.log_episode_env_info and self._log_episode_info_flag:
            self.log_episode_info()

        # Animate several timesteps without actions until simulation settles
        for _ in range(max(self._settle_steps - 1, 1)):
            # Manipulation target is moved on 'onAnimateEndEvent', so we call at least 1 AnimationStep here
            self.sofa_simulation.animate(self._sofa_root_node, self._sofa_root_node.getDt())

        target_position = self._get_new_target_position()
        self._visual_target.reset(new_position=target_position)

        if self.log_episode_env_info:
            self.set_initial_env_state()

        # Reset step counter
        self._current_step = 0
        # Reset trajectory memory
        self._current_trajectory = []
        # Update episode counter
        self._reset_counter += 1

        # Hide the visual model of the SOFA object.
        # The target point will be overlayed onto the observation.
        self._visual_target.set_visibility(False)

        return self._get_observation(self._maybe_update_rgb_buffer()), {}

    def overlay_target(self, observation: np.ndarray, radius: int = 7, scaling_factor: float = 1.0) -> np.ndarray:
        """Adds visual target as overlay to observation. cv2.circle(...) is used."""
        pixel_row, pixel_col = world_to_pixel_coordinates(self._visual_target.get_pose()[:3], self._sofa_camera)
        color = np.asarray([int(x * 255) for x in COLOR_VIS_TARGET]).tolist()
        img = observation.astype(np.int32).copy()
        cv2.circle(img, center=(round(pixel_col * scaling_factor), round(pixel_row * scaling_factor)), radius=radius, color=color, thickness=-1)

        return img.astype(np.uint8)

    def set_initial_env_state(self) -> None:
        """Set IDP, TTP, TGP, distance(TTP, IDP)"""
        self._initial_env_state["IDP"] = self._visual_target_position.copy() if self._visual_target_position is not None else np.array([-1, -1, -1])
        self._initial_env_state["TTP"] = np.array(self._tissue.get_manipulation_target_pose()[:3]).copy()
        self._initial_env_state["TGP"] = self._grasping_position.copy()
        self._initial_env_state["distance_ttp_idp"] = self._calc_distance_to_target()

        self._log_episode_info_flag = True

    def log_episode_info(self) -> None:
        """Log env info for current episode: TTP, IDP, TGP, Distance(TTP, IDP) + success"""
        env_info = self._initial_env_state
        try:
            success = bool(self._get_info()["is_success"])
            reward_available = True
        except Exception:
            success = False
            reward_available = False

        self._log_episode_info_flag = False
        if reward_available:
            self.logger.log(success=success, env_info=env_info)

    def _do_action(self, action) -> None:
        """Scale action and set new poses in simulation"""
        self._last_action_valid = True
        # Multiply with 0.001 (meter to millimeter) -  actions are in [-1, 1]
        # Then multiply with the dt of the simulation, to reach a conversion to millimeter per second:
        if self.action_type == ActionType.DISCRETE:
            delta_pos = self._scale_action(action)
            delta_pos = self._workspace.transform_action(delta_pos)
        elif self.action_type == ActionType.CONTINUOUS:
            action = self._workspace.transform_action(action)  # transforms 2D action in 3D space w.r.t. Workspace.Type
            delta_pos = self._scale_action(action)

        # log last action norm for debugging training
        self._last_action_norm = np.linalg.norm(action)
        self._last_delta_pos_norm = np.linalg.norm(delta_pos)

        # The action corresponds to delta in XYZ, but sofa objects want XYZ + rotation as quaternion.
        current_pose = self._gripper.get_pose()
        current_position = current_pose[:3]
        current_orientation = current_pose[3:]

        new_position = current_position + delta_pos
        new_pose = np.append(new_position, current_orientation)

        # Validate Poses here, throws sofa error in gripper object
        invalid_poses_mask = (self._workspace.get_low() > new_pose[:3]) | (new_pose[:3] > self._workspace.get_high())
        if any(invalid_poses_mask):
            self._last_action_valid = False if any(invalid_poses_mask[[0, 2]]) else True
            invalid_poses_mask = np.append(invalid_poses_mask, [False] * 4)
            new_pose[invalid_poses_mask] = current_pose[invalid_poses_mask]
        self._gripper.set_pose(new_pose, validate=False)

        # Append new position to trajectory
        self._current_trajectory.append(self._gripper.get_pose()[:3].copy())

    def _get_observation(self, maybe_rgb_observation: Union[np.ndarray, None]) -> np.ndarray:
        """Assembles the correct observation based on the ``ObservationType``."""

        if self.observation_type == ObservationType.STATE:
            observation = self.observation_space.sample()

            tgp = self._gripper.get_pose()[:3]
            idp = self._visual_target.get_pose()[:3]
            ttp = self._tissue.get_manipulation_target_pose()[:3]

            # Normalize TTP / TGP withing workspace to [-1, 1]
            observation[:3] = self._workspace.safe_normalization(tgp)
            observation[3:6] = self._workspace.safe_normalization(idp)
            observation[6:] = self._workspace.safe_normalization(ttp)

        elif self.observation_type == ObservationType.RGB:
            if self._deactivate_overlay:
                observation = maybe_rgb_observation
            else:
                observation = self.overlay_target(maybe_rgb_observation, radius=int(round(self.observation_space.shape[0] / 64)))
            self._last_observation = observation

        elif self.observation_type == ObservationType.RGBD:
            observation = self.observation_space.sample()
            if self._deactivate_overlay:
                observation[:, :, :3] = maybe_rgb_observation
            else:
                observation[:, :, :3] = self.overlay_target(maybe_rgb_observation, radius=int(round(self.observation_space.shape[0] / 64)))
            observation[:, :, 3:] = self.get_depth()
            self._last_observation = observation

        return observation

    def _get_done(self) -> bool:
        """Returns if Episode is DONE"""
        for criteria in self.end_episode_criteria:
            if self.reward_info[criteria.value[0]] == criteria.value[1]:
                return True
        return False

    def render(self, mode: Optional[str] = None) -> np.ndarray:
        """Returns the rgb observation from the simulation with overlayed target."""
        image = super().render(mode)
        return self.overlay_target(image, radius=int(round(self.image_shape[0] / 64)))

    def _get_info(self) -> dict:
        """Returns info dict for current episode.

        Notes:
            - for logging in stable baselines 3 env use: ``env = VecMonitor(env, info_keywords=("is_success",))``
            - To make ``VecMonitor`` work see: https://github.com/DLR-RM/stable-baselines3/issues/728
            Issue is fixed in sb3 1.5.0 (https://github.com/DLR-RM/stable-baselines3/releases/tag/v1.5.0)
        """
        ret = {
            "is_success": float(self.reward_info["goal_reached"]),
            "max_def": self._tissue.get_displacement_norm(),
            "action_norm": float(self._last_action_norm),
            "delta_pos_norm": float(self._last_delta_pos_norm),
            "motion_efficiency": self._current_step / (self._initial_env_state["distance_ttp_idp"] * 1e3),
            "motion_smoothness": self.calculate_smoothness(self._current_trajectory, scaling_factor=1e3) / (self._initial_env_state["distance_ttp_idp"]) if len(self._current_trajectory) > 1 else 0.0,
            **self.reward_info,
        }

        if self.log_episode_env_info:
            ret.update(self._initial_env_state)

        return ret

    def _get_reward(self) -> float:
        """Reward function of the Tissue Manipulation Scene

        Reward = - scaling_factor * dist(IDP, TTP) ** distance_exponent
                    + One Time Reward (Goal Reached) (POSITIVE (+))
                    + One Time Penalty (Invalid Action) (NEGATIVE (-))
                    + One Time Penalty (Is Stuck) (NEGATIVE (-))
                    + One Time Penalty (Unstable Tissue Handling) (NEGATIVE (-))
        """
        reward = 0.0
        unstable = self._tissue.get_displacement_norm() > self._stability_threshold

        self.reward_info = {
            "distance_to_target_position": None,
            "distance_to_target_position_with_exponent": None,
            "penalty_invalid_action": 0.0,
            "penalty_is_stuck": 0.0,
            "penalty_unstable_simulation": 0.0,
            "goal_reached": False,
            "valid_action": self._last_action_valid,
            "stable_deformation": not unstable,
            "is_stuck": False,
        }

        distance_to_target_position = self._calc_distance_to_target()
        reward += self._reward_amount_dict["distance_to_target"] * self._reward_scaling_factor * distance_to_target_position**self._distance_exponent

        # Add distance to info dict
        self.reward_info["distance_to_target_position"] = distance_to_target_position
        self.reward_info["distance_to_target_position_with_exponent"] = distance_to_target_position**self._distance_exponent

        # Calculate if goal is reached
        if distance_to_target_position <= self._target_threshold:
            self.reward_info["goal_reached"] = True
            reward += self._reward_amount_dict["one_time_reward_goal"]

        # Penalty if agent is stuck
        self._delta_distance.append(distance_to_target_position)
        if len(self._delta_distance) == self._delta_distance.maxlen:
            stuck = np.max(self._delta_distance) - np.min(self._delta_distance) < EpisodeEndCriteria.STUCK.value[3]
            if stuck and self._reset_counter > EpisodeEndCriteria.STUCK.value[4]:
                self.reward_info["is_stuck"] = True
                self.reward_info["penalty_is_stuck"] = self._reward_amount_dict["one_time_penalty_is_stuck"]
                reward += self.reward_info["penalty_is_stuck"]

        # Penalty for invalid action
        if not self._last_action_valid:
            self.reward_info["penalty_invalid_action"] = self._reward_amount_dict["one_time_penalty_invalid_action"]
            reward += self.reward_info["penalty_invalid_action"]

        # Penalty for forcing sofa to be unstable
        if unstable:
            self.reward_info["penalty_unstable_simulation"] = self._reward_amount_dict["one_time_penalty_unstable_simulation"]
            reward += self.reward_info["penalty_unstable_simulation"]

        self.reward_info["reward"] = reward
        self.annotate_pyglet_window()

        return reward

    def _calc_distance_to_target(self) -> float:
        """Calculates the distance(TTP, IDP). Uses current values of SofaObjects for calculation."""
        current_position = self._gripper.get_pose()[:3]
        tissue_position = self._tissue.get_manipulation_target_pose()[:3]
        target_position = self._visual_target.get_pose()[:3]

        if self.distance_calculation_2d:
            # Remove the y element of the positions to calculate distances in XZ-coordinates only
            current_position = current_position[[0, 2]]
            tissue_position = tissue_position[[0, 2]]
            target_position = target_position[[0, 2]]

        # Distance calculated between the targets
        distance_to_target_position = np.linalg.norm(target_position - tissue_position)

        return float(distance_to_target_position)

    def _scale_discrete_action(self, action: int) -> np.ndarray:
        """Maps action indices to a motion delta."""

        return self._discrete_action_lookup[action]

    def _scale_normalized_action(self, action: np.ndarray) -> np.ndarray:
        """
        Policy is output is clipped to [-1, 1] e.g. [-0.3, 0.8, 1].
        We want to scale that to [-max_rob_vel, max_rob_vel] in mm/s,
        we then have to scale it to m/s (SOFA scene is in meter),
        and further to m/sim_step (because delta T is not 1 second).
        """
        return self.time_step * 0.001 * self._maximum_robot_velocity * action

    def annotate_pyglet_window(self) -> None:
        """Visualize distance as pyglet window caption."""
        if self.internal_render_mode == RenderMode.HUMAN:
            distance_to_show = self.reward_info["distance_to_target_position"]
            if distance_to_show is None:
                distance_to_show = -1.0
            if self.render_framework == RenderFramework.PYGLET:
                self._window.set_caption(f"D:{distance_to_show:.5f}")
            elif self.render_framework == RenderFramework.PYGAME:
                self.pygame.display.set_caption(f"D:{distance_to_show:.5f}")

    def calculate_smoothness(self, trajectory: List, scaling_factor: float = 1e3) -> float:
        """Calculate 2nd order gradient for x and z direction respectively. Then computes Mean Square of Gradients."""
        trajectory = np.asarray(trajectory)
        dx = np.gradient(trajectory[:, 0])
        ddx = np.gradient(dx)
        dz = np.gradient(trajectory[:, -1])
        ddz = np.gradient(dz)

        mean_squared = np.mean([[(x * scaling_factor) ** 2 for x in ddx], [(z * scaling_factor) ** 2 for z in ddz]])

        return float(mean_squared)

    def get_last_observation(self) -> np.ndarray:
        if self.internal_render_mode == RenderMode.NONE:
            raise ValueError("Calling env.render() is invalid when render_mode was set to RenderMode.NONE.")
        return self._last_observation

    @property
    def delta_distances(self) -> List:
        """Returns List of distance changes (deltas)."""
        return list(self._delta_distance)

    def debug_render(self, observation: np.ndarray) -> None:
        """Utility function to render the observation in OpenCV."""
        if self.observation_type == ObservationType.STATE:
            print("STATE observation:", observation)
            return None

        img_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)

        if self.render_framework == RenderFramework.PYGLET:
            cv2.imshow(f"Sofa_{id(self)}", img_bgr)
            cv2.waitKey(1)
        elif self.render_framework == RenderFramework.PYGAME:
            img_bgr = self.pygame.surfarray.make_surface(img_bgr)
            self._window.blit(img_bgr, (self._camera_object.heightViewport.value / 2, self._camera_object.widthViewport.value / 2))
            self.pygame.display.flip()


if __name__ == "__main__":
    pp = pprint.PrettyPrinter()

    flat = True
    bounds = [-0.055, 0.045, -0.05, 0.05, 0.1075 - 0.025, 0.1075 + 0.015]
    if flat:
        # remove y axis
        bounds = [bounds[i] for i in [0, 1, 4, 5]]

    env = TissueManipulationEnv(
        scene_path=SCENE_DESCRIPTION_FILE_PATH,
        render_mode=RenderMode.HUMAN,
        observation_type=ObservationType.RGB,
        action_type=ActionType.CONTINUOUS,
        discrete_action_magnitude=0.5,
        image_shape=(600, 600),
        frame_skip=3,
        time_step=DT,
        maximum_robot_velocity=VEL * 1e3,
        squared_distance=False,
        distance_calculation_2d=True,
        create_scene_kwargs={
            "camera_position": (0.0, -0.2, 0.15),
            "camera_look_at": (0.0, 0.0, 0.12),
            "camera_field_of_view_vertical": 57,
            # https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view-
            "randomize_grasping_point": True,
            "randomize_manipulation_target": True,
            "show_workspace_bounding_box": True,
            "workspace_kwargs": {
                "bounds": bounds,
                "workspace_type": WorkspaceType.TISSUE_ALIGNED if flat else WorkspaceType.GENERAL,
            },
        },
        end_episode_criteria=[EpisodeEndCriteria.SUCCESS, EpisodeEndCriteria.STABILITY, EpisodeEndCriteria.STUCK],
        debug=True,
    )
    env.reset()

    # move gripper to each corner of the workspace
    low, high = env._workspace.get_low(), env._workspace.get_high()
    targets = [low[:], [low[0], low[1], high[2]], high[:], [high[0], high[1], low[2]], low[:]]

    gripper = env._gripper
    gripper.motion_path = gripper.create_linear_motion(targets.pop(0), dt=DT, velocity=VEL)

    rewards = []
    max_deformation = []
    distance = []
    done = False
    random_action = env.action_space.sample()
    fps_list = deque(maxlen=100)

    try:
        # while not done:
        while True:
            start = time.time()
            obs, reward, terminated, truncated, info = env.step(random_action)
            done = terminated or truncated

            if len(gripper.motion_path) == 0 and len(targets) > 0:
                hit_corner = True
                motion_path = gripper.create_linear_motion(targets.pop(0), dt=DT, velocity=VEL)
                gripper.add_to_motion_path(motion_path)
            elif len(gripper.motion_path) == 0 and len(targets) == 0:
                print("simulation has finished")
                break

            end = time.time()
            fps = 1 / (end - start)
            fps_list.append(fps)
            rewards.append(reward)
            distance.append(info["distance_to_target_position"] / 1e-2)  # in cm
            max_deformation.append(info["max_def"] / env._stability_threshold)  # norm with threshold

            info["fps"] = fps

    except KeyboardInterrupt:
        pass
