import cv2
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np

from collections import deque, defaultdict
from enum import Enum, unique
from pathlib import Path

from typing import Callable, Union, Tuple, Optional, List, Any, Dict

from sofa_env.base import SofaEnv, RenderMode, RenderFramework
from sofa_env.scenes.tissue_retraction.sofa_objects.end_effector import EndEffector, is_in, add_waypoints_to_end_effector

HERE = Path(__file__).resolve().parent
SCENE_DESCRIPTION_FILE_PATH = HERE / Path("scene_description.py")


@unique
class ObservationType(Enum):
    """Observation type specifies whether the environment step returns RGB images or a defined state"""

    RGB = 0
    STATE = 1
    RGBD = 2


@unique
class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


@unique
class Phase(Enum):
    """The environment has to be in one of two phases, grasping the tissue or retracting the tissue."""

    GRASPING = 0
    RETRACTING = 1


@unique
class CollisionPunishmentMode(Enum):
    """Two types of how to punish collisions.

    ``MOTIONINTISSUE`` penalizes moving in the tissue. An initial collision is not punished. ``CONTACTDISTANCE`` immediately punishes collisions proportional to their distance to the ``grasping_position``.
    """

    MOTIONINTISSUE = 0
    CONTACTDISTANCE = 1


class TissueRetractionEnv(SofaEnv):
    """Tissue Retraction Environment

    Args:
        scene_path (Union[str, Path]): Path to the scene description script that contains this environment's ``createScene`` function.
        image_shape (Tuple[int, int]): Height and Width of the rendered images.
        observation_type (ObservationType): Whether to return RGB images or an array of states as the observation.
        observe_phase_state (bool): Whether to include the current ``Phase`` in the observation. For ``ObservationType.RGB`` this will return a dictionary observation. For ``ObservationType.STATE`` it will add another value to the array.
        time_step (float): size of simulation time step in seconds (default: 0.01).
        frame_skip (int): number of simulation time steps taken (call ``_do_action`` and advance simulation) each time step is called (default: 1).
        settle_steps (int): How many steps to simulate without returning an observation after resetting the environment.
        maximum_robot_velocity (float): Maximum per direction robot velocity in meters per second. Used for scaling the actions that are passed to ``env.step(action)``.
        render_mode (RenderMode): create a window (``RenderMode.HUMAN``) or run headless (``RenderMode.HEADLESS``).
        render_framework (RenderFramework): choose between pyglet and pygame for rendering
        action_space (Optional[gymnasium.spaces.Box]): An optional Box action space to set the limits for clipping.
        grasping_position (Union[Tuple[float, float, float], np.ndarray]): World coordinates of the point that should be reached during grasping.
        end_position (Union[Tuple[float, float, float], np.ndarray]): World coordinates of the point that should be reached during retracting.
        grasping_threshold (float): Distance to the ``grasping_position`` at which grasping is triggered.
        end_position_threshold (float): Distance to the ``end_position`` at which episode success is triggered.
        maximum_grasp_height (float): A maximum Y-Value to further constrain when grasping is triggered.
        collision_punishment_mode (CollisionPunishmentMode): How to punish collisions.
        maximum_in_tissue_travel (float): Parameter for ``CollisionPunishmentMode.MOTIONINTISSUE`` that controls at what distance travelled in collision a punishment is triggered.
        collision_tolerance (float): Distance to the ``grasping_position`` until which collisions are ignored.
        reward_amount_dict (dict): Dictionary to weigh the components of the reward function.
        create_scene_kwargs (Optional[dict]): A dictionary to pass additional keyword arguments to the ``createScene`` function.
        on_reset_callbacks (Optional[List[Callable]]): Functions that are called as the last step of the ``env.reset()`` function.
    """

    def __init__(
        self,
        scene_path: Union[str, Path] = SCENE_DESCRIPTION_FILE_PATH,
        image_shape: Tuple[int, int] = (84, 84),
        observation_type: ObservationType = ObservationType.RGB,
        observe_phase_state: bool = False,
        time_step: float = 0.1,
        frame_skip: int = 3,
        settle_steps: int = 20,
        action_type: ActionType = ActionType.CONTINUOUS,
        maximum_robot_velocity: float = 5.0,
        discrete_action_magnitude: Optional[float] = 3.0,
        render_mode: RenderMode = RenderMode.HEADLESS,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        action_space: Optional[gym.spaces.Box] = None,
        grasping_position: Union[Tuple[float, float, float], np.ndarray] = (-0.0485583, 0.0085, 0.0356076),
        end_position: Union[Tuple[float, float, float], np.ndarray] = (-0.019409, 0.062578, -0.00329643),
        grasping_threshold: float = 0.003,
        end_position_threshold: float = 0.003,
        maximum_grasp_height: float = np.inf,
        collision_punishment_mode: CollisionPunishmentMode = CollisionPunishmentMode.MOTIONINTISSUE,
        maximum_in_tissue_travel: float = 0.003,  # for CollisionPunishmentMode.MOTIONINTISSUE
        collision_tolerance: float = 0.003,
        reward_amount_dict={
            "one_time_reward_grasped": 1.0,
            "one_time_reward_goal": 1.0,
            "time_step_cost_scale_in_grasp_phase": 1.2,
            "target_visible_scaling": 0,
            "control_cost_factor": 0.0,
            "workspace_violation_cost": 0.1,
            "collision_cost_factor": 0.1,
        },
        create_scene_kwargs: Optional[dict] = None,
        on_reset_callbacks: Optional[List[Callable]] = None,
    ) -> None:
        # Pass image shape to the scene creation function
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape

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
        action_dimensionality = 3
        if action_type == ActionType.CONTINUOUS:
            self._maximum_robot_velocity = maximum_robot_velocity
            if action_space is None:
                action_space = spaces.Box(low=-maximum_robot_velocity, high=maximum_robot_velocity, shape=(action_dimensionality,), dtype=np.float32)
            else:
                if not (isinstance(action_space, spaces.Box) and isinstance(action_space.low, np.ndarray) and isinstance(action_space.high, np.ndarray)):
                    raise ValueError("If setting a manual continuous action space, please pass it as a gymnasium.spaces.Box. with correct shape.")

            # Check if the action space is in [-1, 1] to determine if we have to scale the actions in the env
            self._action_space_is_normalized = all(action_space.low == -1.0) and all(action_space.high == 1.0)
            self._action_space_is_robot_velocity = all(action_space.low == -maximum_robot_velocity) and all(action_space.high == maximum_robot_velocity)

            # Determine how to convert the policy output (e.g. [-0.8, 1.0, -0.2]) to the value passed to sofa (e.g. [-0.00008, 0.0001, -0.00002])
            if self._action_space_is_normalized:
                self._scale_action = self._scale_normalized_action
            elif self._action_space_is_robot_velocity:
                self._scale_action = self._scale_robot_vel_action
            else:
                self._scale_action = self._scale_unnormalized_action
        else:
            if action_space is None:
                action_space = spaces.Discrete(action_dimensionality * 2 + 1)
            else:
                if not (isinstance(action_space, spaces.Discrete) and action_space.n == action_dimensionality * 2 + 1):
                    raise ValueError(f"If setting a manual discrete action space, please pass it as a gymnasium.spaces.Discrete with {action_dimensionality * 2 + 1} elements.")

            self._scale_action = self._scale_discrete_action

            if discrete_action_magnitude is None:
                raise ValueError("For discrete actions, please pass a step size.")

            # [step, 0, 0, ...], [-step, 0, 0, ...], [0, step, 0, ...], [0, -step, 0, ...]
            action_list = []
            for i in range(action_dimensionality * 2):
                action = [0.0] * action_dimensionality
                step_size = discrete_action_magnitude
                action[int(i / 2)] = (1 - 2 * (i % 2)) * step_size
                action_list.append(action)

            # Noop action
            action_list.append([0.0] * action_dimensionality)

            self._discrete_action_lookup = np.array(action_list)

            self._discrete_action_lookup *= self.time_step * 0.001
            self._discrete_action_lookup.flags.writeable = False

        self.action_space = action_space

        ###################
        # Observation Space
        ###################

        # Whether to include the current phase to the observation
        self._observe_phase = observe_phase_state

        # State observations
        if observation_type == ObservationType.STATE:
            dim_states = 3
            if observe_phase_state:
                dim_states += 1
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(dim_states,), dtype=np.float32)

        # Image observations
        elif observation_type == ObservationType.RGB:
            if observe_phase_state:
                self.observation_space = spaces.Dict(
                    {
                        "rgb": spaces.Box(low=0, high=255, shape=image_shape + (3,), dtype=np.uint8),
                        "phase": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                    }
                )
            else:
                self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (3,), dtype=np.uint8)

        elif observation_type == ObservationType.RGBD:
            if observe_phase_state:
                self.observation_space = spaces.Dict(
                    {
                        "rgbd": spaces.Box(low=0, high=255, shape=image_shape + (4,), dtype=np.uint8),
                        "phase": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                    }
                )
            else:
                self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (4,), dtype=np.uint8)

        else:
            raise Exception(f"Please set observation_type to a value of ObservationType. Received {observation_type}.")

        self._observation_type = observation_type

        #########################
        # Episode specific values
        #########################

        # Keep information about the last action for control cost
        self._last_grasping_distance = np.nan
        self._last_action_violated_workspace = False

        # Current phase starts in grasping
        self._phase = Phase.GRASPING

        # Temporary values for collision punishment
        self._initial_tissue_collision_point = np.zeros(3, dtype=np.float32)
        self._collision_is_fresh = False

        # Parameters for collision punishment
        self._collision_tolerance = collision_tolerance
        self._collision_punishment_mode = collision_punishment_mode
        self._maximum_in_tissue_travel = maximum_in_tissue_travel

        # Infos per episode
        self.episode_info = {}
        self.episode_info["episode_control_cost"] = 0
        self.episode_info["episode_collision_cost"] = 0
        self.episode_info["episode_workspace_violation_cost"] = 0
        self.episode_info["steps_in_grasping_phase"] = 0
        self.episode_info["steps_in_retraction_phase"] = 0
        self.episode_info["steps_in_collision"] = 0
        self.episode_info["steps_in_workspace_violation"] = 0
        self.episode_info["return_from_grasping"] = 0
        self.episode_info["return_from_retracting"] = 0

        # All parameters of the reward function that are not passed default to 0.0
        self.reward_amount_dict = defaultdict(float) | reward_amount_dict

        # Task specific parameters
        self._grasping_position = np.array(grasping_position)
        self._end_position = np.array(end_position)

        self._end_position_threshold = end_position_threshold
        self._grasping_threshold = grasping_threshold
        self._maximum_grasp_height = maximum_grasp_height

        # Parameters for checking visibility of the target
        self.target_color_rgb = np.array([150, 0, 150], dtype=np.uint8)
        self.target_color_hsv = cv2.cvtColor(self.target_color_rgb.reshape((1, 1, 3)), cv2.COLOR_RGB2HSV).reshape((3,))
        self.color_delta = np.array([30, 80, 80])
        # mask is {0, 255} per pixel.
        if image_shape == (84, 84):
            self.max_sum_visibility_mask = 6630
        elif image_shape == (128, 128):
            self.max_sum_visibility_mask = 23460
        elif image_shape == (480, 480):
            self.max_sum_visibility_mask = 337365
        else:
            print(f"Visibility ratio not defined for resolution {image_shape}.")
            if reward_amount_dict["target_visible_scaling"] > 0.0:
                raise ValueError(f"Was asked to reward target visibility (target_visible_scaling) but could not find a value for self.max_sum_visibility_mask for image shape {image_shape}. Please use the debug_render method of TissueRetractionEnv to determine the value for self.max_sum_visibility_mask at image shape {image_shape}. Currently available image shapes are 84x84, 128x128, and 480x480.")
            self.max_sum_visibility_mask = np.inf

        # Callback functions called on reset
        if on_reset_callbacks is not None:
            self.on_reset_callbacks = on_reset_callbacks
        else:
            self.on_reset_callbacks = []

    def _init_sim(self):
        "Initialise simulation and calculate values for reward scaling."
        super()._init_sim()

        self.workspace = self.scene_creation_result["workspace"]
        self.tissue_box = self.scene_creation_result["tissue_box"]
        self.end_effector = self.scene_creation_result["interactive_objects"]["end_effector"]

        # Scale the distances to an interval of [0, 1]
        self._reward_scaling_factor = 1.0 / np.linalg.norm(self.workspace["high"][:3] - self.workspace["low"][:3])
        # Time step cost is *time_step_cost_scale_in_grasp_phase of the distance from grasping point to end point
        grapsing_to_end_distance = np.linalg.norm(self._grasping_position - self._end_position)

        self._time_step_cost_in_grasping_phase = -self._reward_scaling_factor * grapsing_to_end_distance * self.reward_amount_dict["time_step_cost_scale_in_grasp_phase"]

    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        """Step function of the environment that applies the action to the simulation and returns observation, reward, done signal, and info."""

        maybe_rgb_observation = super().step(action)

        observation = self._get_observation(maybe_rgb_observation=maybe_rgb_observation)
        reward = self._get_reward(maybe_rgb_observation)
        terminated = self._get_done()
        info = self._get_info()

        return observation, reward, terminated, False, info

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

    def _scale_robot_vel_action(self, action: np.ndarray) -> np.ndarray:
        """
        Policy is output is clipped to [-max_rob_vel, max_rob_vel].
        We have to scale it to m/s (SOFA scene is in meter),
        and further to m/sim_step (because delta T is not 1 second).
        """
        return self.time_step * 0.001 * action

    def _scale_unnormalized_action(self, action: np.ndarray) -> np.ndarray:
        """
        Policy is output is not clipped at all.
        We have to normalize it, where the norm of the action is bigger than max_rob_vel,
        and further to m/sim_step (because delta T is not 1 second).
        """
        # Norm of the velocity vector in [-inf, inf]
        action_norm = np.linalg.norm(action)
        # If the norm is bigger than the maximum robot velocity (in mm/s),
        # scale it down -> ||action||_2 in [0, max_rob_vel].
        # Make sure, that the max_rob_vel is >= 1, otherwise action/action_norm makes action bigger.
        clipped_action = np.where(action_norm < self._maximum_robot_velocity, action, action / (action_norm + 1e-12))

        # Scene is in meters, which results in very small action values
        # Multiply with 0.001 (millimeter to meter)
        # Then multiply with the dt of the simulation, to reach a conversion to meter per time step.
        # action was clipped in a way, that the norm of the action vector is at max the max_rob_vel in m/s
        return 0.001 * self.time_step * clipped_action

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

        # original action as predicted by the policy to calculate the control cost (punish high velocities).
        self._last_policy_action = action
        self._last_action_violated_workspace = any(invalid_poses_mask)

    def set_reward_coefficients(
        self,
        collision_cost_factor: Optional[float] = None,
        workspace_violation_cost: Optional[float] = None,
    ) -> None:
        """Change the weights of collision cost and workspace violation cost. Used for Curriculum Learning."""

        if collision_cost_factor is not None:
            self.reward_amount_dict["collision_cost_factor"] = collision_cost_factor

        if workspace_violation_cost is not None:
            self.reward_amount_dict["workspace_violation_cost"] = workspace_violation_cost

    def _get_reward(self, rgb_observation: Optional[np.ndarray] = None) -> float:
        """Reward function of the Tissue Retraction task.

        Grasping:
            - time step cost (constant)
            - collision (distance to grasping point, scaled with workspace factor, only added for distances greater than the end effectors grasping distance)
            - distance to grapsing point
            - control cost proportional to action magnitude
            - cost for trying to leave the workspace
        Retracting:
            - distance to end point
            - control cost proportional to action magnitude
            - cost for trying to leave the workspace
            - visibility of the target
        """

        reward = 0.0
        current_position = self.end_effector.get_pose()[:3]

        self.reward_info = {
            "distance_to_grasping_position": None,
            "distance_to_grasping_position_reward": None,
            "distance_to_end_position": None,
            "distance_to_end_position_reward": None,
            "collision_cost": None,
            "workspace_violation_cost": None,
            "goal_reached": False,
            "target_visible": None,
            "control_cost": None,
        }

        # Calculate control cost based on sum of squared velocities
        control_cost = -np.sum(np.square(self._last_policy_action)) * self.reward_amount_dict["control_cost_factor"]

        reward += control_cost
        self.reward_info["control_cost"] = control_cost

        # Calculate cost for trying to leave the workspace
        workspace_violation_cost = -float(self._last_action_violated_workspace) * self.reward_amount_dict["workspace_violation_cost"]
        self.reward_info["workspace_violation_cost"] = workspace_violation_cost
        reward += workspace_violation_cost

        if self._phase == Phase.GRASPING:
            self.episode_info["steps_in_grasping_phase"] += 1

            # Distance to grasping point
            distance_to_grasping_position = np.linalg.norm(self._grasping_position - current_position)
            self._last_grasping_distance = distance_to_grasping_position
            distance_to_grasping_position_reward = -self._reward_scaling_factor * distance_to_grasping_position + self._time_step_cost_in_grasping_phase
            reward += distance_to_grasping_position_reward

            self.reward_info["distance_to_grasping_position"] = distance_to_grasping_position
            self.reward_info["distance_to_grasping_position_reward"] = distance_to_grasping_position_reward

            ############
            # COLLISIONS
            ############
            if self.reward_amount_dict["collision_cost_factor"] > 0.0:
                #####################
                # CHECK FOR COLLISION
                #####################

                collision_relevant_points = [
                    current_position,
                    current_position + np.array([0.00763673, 0.00214612, 0.0]),  # right jaw of the open gripper
                    current_position + np.array([-0.00763673, 0.00214612, 0.0]),  # left jaw of the open gripper
                ]

                is_in_collision = any(all(map(is_in, point, self.tissue_box["low"], self.tissue_box["high"])) for point in collision_relevant_points)

                is_outside_tolerance = distance_to_grasping_position > self._collision_tolerance
                distance = distance_to_grasping_position

                ######################
                # COLLISION_PUNISHMENT
                ######################
                if is_in_collision and is_outside_tolerance:
                    if self._collision_punishment_mode == CollisionPunishmentMode.CONTACTDISTANCE:
                        collision_cost = -self.reward_amount_dict["collision_cost_factor"] * distance * self._reward_scaling_factor

                    elif self._collision_punishment_mode == CollisionPunishmentMode.MOTIONINTISSUE:
                        if not self._collision_is_fresh:
                            self._initial_tissue_collision_point[:] = current_position[:]
                            self._collision_is_fresh = True

                        if self._collision_is_fresh:
                            current_xz = np.array([current_position[0], current_position[2]])
                            initial_xz = np.array([self._initial_tissue_collision_point[0], self._initial_tissue_collision_point[2]])
                            motion_in_tissue = np.linalg.norm(current_xz - initial_xz)
                            if motion_in_tissue > self._maximum_in_tissue_travel:
                                collision_cost = -self.reward_amount_dict["collision_cost_factor"] * motion_in_tissue * self._reward_scaling_factor
                            else:
                                collision_cost = 0.0
                        else:
                            collision_cost = 0.0

                    else:
                        raise ValueError
                else:
                    self._initial_tissue_collision_point[:] = 0.0
                    self._collision_is_fresh = False
                    collision_cost = 0.0

                self.reward_info["collision_cost"] = collision_cost
                reward += collision_cost

            # Trigger grasping and change the Phase
            if distance_to_grasping_position <= self._grasping_threshold and current_position[1] <= self._maximum_grasp_height:
                self.end_effector.has_grasped = True
                self._phase = Phase.RETRACTING
                reward += self.reward_amount_dict["one_time_reward_grasped"]

        else:
            self.episode_info["steps_in_retraction_phase"] += 1

            # Distance to end position
            distance_to_end_position = np.linalg.norm(self._end_position - current_position)
            distance_to_end_position_reward = -self._reward_scaling_factor * distance_to_end_position

            reward += distance_to_end_position_reward

            self.reward_info["distance_to_end_position"] = distance_to_end_position
            self.reward_info["distance_to_grasping_position"] = self._last_grasping_distance
            self.reward_info["distance_to_end_position_reward"] = distance_to_end_position_reward

            # Target visibility
            if self.reward_amount_dict["target_visible_scaling"] > 0.0:
                if rgb_observation is None:
                    raise ValueError(
                        f"For including target visibility in the reward function, please set the render_mode to RenderMode.HUMAN or RenderMode.HEADLESS. \
                                     If you do not want to visually render the scene, set ``target_visible_scaling`` in the reward_amount_dict to ``0`` \
                                     (currently at {self.reward_amount_dict['target_visible_scaling']})."
                    )
                hsv_image = cv2.cvtColor(rgb_observation, cv2.COLOR_RGB2HSV)
                visibility_mask = cv2.inRange(hsv_image, self.target_color_hsv - self.color_delta, self.target_color_hsv + self.color_delta)
                ratio_visible = np.sum(visibility_mask) / self.max_sum_visibility_mask
                self.reward_info["target_visible"] = ratio_visible
                reward += self.reward_amount_dict["target_visible_scaling"] * ratio_visible

            if distance_to_end_position <= self._end_position_threshold:
                self.reward_info["goal_reached"] = True
                reward += self.reward_amount_dict["one_time_reward_goal"]

        # Annotate the pyglet window
        if self.internal_render_mode == RenderMode.HUMAN:
            distance_to_show = self.reward_info["distance_to_end_position"] if self._phase == Phase.RETRACTING else self.reward_info["distance_to_grasping_position"]
            if distance_to_show is None:
                distance_to_show = -1.0
            if self.render_framework == RenderFramework.PYGLET:
                self._window.set_caption(f"{self._phase.name} {distance_to_show:.5f}")
            elif self.render_framework == RenderFramework.PYGAME:
                self.pygame.display.set_caption(f"{self._phase.name} {distance_to_show:.5f}")

        self.reward_info["reward"] = reward
        return reward

    def _get_done(self) -> bool:
        """Look up if the episode is finished."""
        return self.reward_info["goal_reached"]

    def _get_observation(self, maybe_rgb_observation: Union[np.ndarray, None]) -> Union[np.ndarray, dict]:
        """Assembles the correct observation based on the ``ObservationType`` and ``observe_phase_state``."""

        if self._observation_type == ObservationType.RGB:
            assert maybe_rgb_observation is not None
            if self._observe_phase:
                observation = {
                    "rgb": maybe_rgb_observation,
                    "phase": np.array(self._phase.value, dtype=self.observation_space["phase"].dtype).reshape(self.observation_space["phase"].shape),
                }
            else:
                observation = maybe_rgb_observation
        elif self._observation_type == ObservationType.RGBD:
            if self._observe_phase:
                observation = self.observation_space.sample()
                observation["rgbd"][:, :, :3] = maybe_rgb_observation
                observation["rgbd"][:, :, 3:] = self.get_depth()
                observation["phase"] = np.array(self._phase.value, dtype=self.observation_space["phase"].dtype).reshape(self.observation_space["phase"].shape)
            else:
                observation = self.observation_space.sample()
                observation[:, :, :3] = maybe_rgb_observation
                observation[:, :, 3:] = self.get_depth()
        else:
            observation = self.observation_space.sample()
            end_effector_position = self.end_effector.get_pose()[:3]
            observation[:3] = 2 * (end_effector_position - self.workspace["low"][:3]) / (self.workspace["high"][:3] - self.workspace["low"][:3]) - 1

            if self._observe_phase:
                observation[-1] = self._phase.value

        return observation

    def _get_info(self) -> dict:
        """Assemble the info dictionary."""
        self.episode_info["episode_control_cost"] += self.reward_info["control_cost"]
        self.episode_info["episode_collision_cost"] += self.reward_info["collision_cost"] if self.reward_info["collision_cost"] is not None else 0
        self.episode_info["episode_workspace_violation_cost"] += self.reward_info["workspace_violation_cost"] if self.reward_info["workspace_violation_cost"] is not None else 0

        self.episode_info["steps_in_collision"] += 1 if self.reward_info["collision_cost"] is not None and self.reward_info["collision_cost"] < 0.0 else 0
        self.episode_info["steps_in_workspace_violation"] += 1 if self.reward_info["workspace_violation_cost"] < 0.0 else 0

        self.episode_info["return_from_grasping"] += self.reward_info["distance_to_grasping_position_reward"] if self.reward_info["distance_to_grasping_position_reward"] is not None else 0
        self.episode_info["return_from_retracting"] += self.reward_info["distance_to_end_position_reward"] if self.reward_info["distance_to_end_position_reward"] is not None else 0

        self.info = {
            "phase": self._phase.value,
            "phase_was_switched": True if self._phase == Phase.RETRACTING else False,
            "in_collision": self.reward_info["collision_cost"] is not None,
        }

        return {**self.info, **self.reward_info, **self.episode_info}

    def reset(self, seed: Union[int, np.random.SeedSequence, None] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Union[np.ndarray, None], Dict]:
        super().reset(seed)

        # Seed the instrument
        if self.unconsumed_seed:
            seeds = self.seed_sequence.spawn(1)
            self.end_effector.seed(seeds[0])
            self.unconsumed_seed = False

        # Reset end_effector and phase
        self._phase = Phase.GRASPING
        self.end_effector.reset()
        self._initial_tissue_collision_point = np.zeros(3, dtype=np.float32)
        self._collision_is_fresh = False

        # Clear the episode info values
        for key in self.episode_info:
            self.episode_info[key] = 0

        # Execute any callbacks passed to the env.
        for callback in self.on_reset_callbacks:
            callback(self)

        # Animate several timesteps without actions until simulation settles
        for _ in range(self._settle_steps):
            self.sofa_simulation.animate(self._sofa_root_node, self._sofa_root_node.getDt())

        return self._get_observation(maybe_rgb_observation=self._maybe_update_rgb_buffer()), {}

    def get_gripper(self) -> EndEffector:
        """Return the EndEffector of the scene."""
        return self.end_effector

    def get_grasping_position(self) -> np.ndarray:
        """Return the grasping_position of the scene."""
        return self._grasping_position

    def get_end_position(self) -> np.ndarray:
        """Return the end_position of the scene."""
        return self._end_position

    def debug_render(self) -> None:
        """Utility function to render the observation in OpenCV and determine the thresholds for target visibility."""
        rgb_observation = self._maybe_update_rgb_buffer()
        hsv_image = cv2.cvtColor(rgb_observation, cv2.cv2.COLOR_RGB2HSV)
        visibility_mask = cv2.inRange(hsv_image, self.target_color_hsv - self.color_delta, self.target_color_hsv + self.color_delta)
        print(np.sum(visibility_mask))
        result = cv2.bitwise_and(rgb_observation, rgb_observation, mask=visibility_mask)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.cvtColor(rgb_observation, cv2.COLOR_RGB2BGR)
        display_img = np.hstack([img_bgr, result_bgr])
        cv2.imshow(f"Sofa_{id(self)}", display_img)
        cv2.waitKey(1)


if __name__ == "__main__":
    import pprint
    import time

    pp = pprint.PrettyPrinter()

    env = TissueRetractionEnv(
        observation_type=ObservationType.RGB,
        action_type=ActionType.CONTINUOUS,
        render_mode=RenderMode.HUMAN,
        collision_punishment_mode=CollisionPunishmentMode.CONTACTDISTANCE,
        observe_phase_state=False,
        image_shape=(480, 480),
        frame_skip=3,
        time_step=0.1,
        maximum_robot_velocity=3,
        maximum_grasp_height=0.0088819,
        reward_amount_dict={
            "one_time_reward_grasped": 1.0,
            "one_time_reward_goal": 1.0,
            "time_step_cost_scale_in_grasp_phase": 1.2,
            "target_visible_scaling": 1.0,
            "control_cost_factor": 0.0,
            "workspace_violation_cost": 0.1,
            "collision_cost_factor": 2.0,
        },
        create_scene_kwargs={
            "show_floor": True,
            "texture_objects": False,
            "workspace_height": 0.09,
            "workspace_width": 0.075,
            "workspace_depth": 0.09,
            "camera_field_of_view_vertical": 42,
        },
    )

    env.reset()

    end_effector = env.get_gripper()
    start_position = end_effector.initial_pose[:3]

    way_points = [[0, 0.007, 0], [0.05, 0.007, 0.0], [0, 0.06, 0], env._grasping_position, env._end_position]

    add_waypoints_to_end_effector(way_points, end_effector)

    no_action = np.array([0.0] * 3)
    done = False

    fps_list = deque(maxlen=100)

    while not done:
        start = time.time()
        obs, reward, terminated, truncated, info = env.step(no_action)
        done = terminated or truncated
        end = time.time()
        fps = 1 / (end - start)
        fps_list.append(fps)
        # env.debug_render()

        if done:
            env.end_effector.set_pose(np.append(env._end_position, (0.0, 0.0, 0.0, 1.0)))

        print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")
