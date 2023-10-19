import gymnasium.spaces as spaces
import numpy as np

from sofa_env.scenes.search_for_point.sofa_objects.point_of_interest import PointOfInterest
from sofa_env.scenes.search_for_point.scene_description import POIMode
from sofa_env.scenes.grasp_lift_touch.sofa_objects.cauter import Cauter

from enum import Enum, unique
from pathlib import Path
from typing import Optional, Tuple, Union, List, Callable, Any, Dict
from collections import defaultdict
from functools import reduce

import Sofa
import Sofa.Core

from sofa_env.base import RenderMode, SofaEnv, RenderFramework
from sofa_env.sofa_templates.camera import PivotizedCamera
from sofa_env.sofa_templates.rigid import PivotizedArticulatedInstrument

from sofa_env.utils.camera import world_to_pixel_coordinates


HERE = Path(__file__).resolve().parent
SCENE_DESCRIPTION_FILE_PATH = HERE / "scene_description.py"


@unique
class ActionType(Enum):
    """Action type specifies whether the actions are continuous values that are added to the camera state, or discrete actions that increment the camera state."""

    DISCRETE = 0
    CONTINUOUS = 1


@unique
class ActiveVision(Enum):
    """Active vision instrument specifies which instrument is used to do the task"""

    DEACTIVATED = 0  # Standard task where the camera needs to find the poi
    CAUTER = 1


@unique
class ObservationType(Enum):
    RGB = 0
    RGBD = 1
    STATE = 2


class SearchForPointEnv(SofaEnv):
    """Search-for-Point Environment

    The Search for Point Environment is an environment, that simulates a minimally invasive surgery.
    It consists of a laparoscopic 30 degree oblique viewing camera, an endoscopic cauter and two grippers which
    are placed on fixed trocar positions and a randomized orientation at the abdomen of a human.
    By default active vision is deactivated and the goal of the environment is to strategically find and visualize a green point of interest within the
    abdominal cavity by controlling the camera.
    The position of the point of interest is sampled from the visible surface of the organs.
    If active vision is activated the goal is to touch the green point with the selected active vision instrument.
    In that case the green point is sampled in an area that is reachable for the instrument.

    Args:
        image_shape (Tuple[int, int]): Height and Width of the rendered images.
        render_mode (RenderMode): Create a window (``RenderMode.HUMAN``), run headless (``RenderMode.HEADLESS``), or do not create a render buffer at all (``RenderMode.NONE``).
        render_framework (RenderFramework): choose between pyglet and pygame for rendering
        time_step (float): size of simulation time step in seconds (default: 0.01).
        frame_skip (int): number of simulation time steps taken (call ``_do_action`` and advance simulation) each time step is called (default: 1).
        scene_path (Union[str, Path]): Path to the scene description script that contains this environment's ``createScene`` function.
        create_scene_kwargs (Optional[dict]): A dictionary to pass additional keyword arguments to the ``createScene`` function.
        maximum_state_velocity (Union[np.ndarray, float]): Velocity in deg/s for pts and mm/s for d in state space which are applied with a normalized action of value 1.
        discrete_action_magnitude (Union[np.ndarray, float]): Discrete change in state space in deg/s for pts and mm/s for d.
        on_reset_callbacks (Optional[List[Callable]]): Functions that are called as the last step of the ``env.reset()`` function.
        action_type: Whether the actions are continuous values or discrete actions
        reward_amount_dict (dict): Dictionary to weigh the components of the reward function.
        tolerance_image_center (float): Tolerated distance from the POI to the center of the image in ``pixel / maximum_pixel_distance_to_center`` for which the task is successful.
        tolerance_target_distance (float): Tolerated Cartesian distance from camera to POI in millimeters for which the task is considered as successful.
        active_vision_mode (ActiveVision): Specifies the active vision mode, by deactivating it or selecting a instrument.
        target_distance_camera_poi (float): Desired distance between camera and target point in millimeters.
        instrument_touch_tolerance (float): How close the instrument needs to be to the target to be considered a successful touch.
        cauter_activation (bool): If ``True`` the cauter needs to be activate when touching the target.
        hide_surgeon_gripper (bool): If ``True`` the surgeon gripper is hidden in the image.
        hide_assistant_gripper (bool): If ``True`` the assistant gripper is hidden in the image.
        hide_cauter (bool): If ``True`` the cauter is hidden in the image. Is set to ``False`` if active vision is activated for the cauter.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int] = (84, 84),
        render_mode: RenderMode = RenderMode.HEADLESS,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        observation_type: ObservationType = ObservationType.RGB,
        time_step: float = 0.1,
        frame_skip: int = 3,
        scene_path: Union[str, Path] = SCENE_DESCRIPTION_FILE_PATH,
        create_scene_kwargs: Optional[dict] = None,
        maximum_state_velocity: Union[np.ndarray, float] = np.array([20.0, 20.0, 30.0, 35.0]),
        discrete_action_magnitude: Union[np.ndarray, float] = np.array([5.0, 5.0, 5.0, 5.0]),
        on_reset_callbacks: Optional[List[Callable]] = None,
        action_type: ActionType = ActionType.CONTINUOUS,
        reward_amount_dict: dict = {
            ActiveVision.DEACTIVATED: {
                "poi_is_in_frame": 0.0,
                "relative_camera_distance_error_to_poi": -0.0,
                "delta_relative_camera_distance_error_to_poi": -0.0,
                "relative_pixel_distance_poi_to_image_center": -0.0,
                "delta_relative_pixel_distance_poi_to_image_center": -0.0,
                "successful_task": 0.0,
            },
            ActiveVision.CAUTER: {
                "collision_cauter": -0.0,
                "relavtive_distance_cauter_target": -0.0,
                "relavtive_delta_distance_cauter_target": -0.0,
                "cauter_touches_target": 0.0,
                "successful_task": 0.0,
                "cauter_action_violated_state_limits": -0.0,
            },
        },
        target_distance_camera_poi: float = 120.0,
        tolerance_image_center: float = 0.2,
        tolerance_target_distance: float = 10.0,
        instrument_touch_tolerance: float = 5.0,
        transparent_abdominal_wall: bool = False,
        hide_surgeon_gripper: bool = True,
        hide_assistant_gripper: bool = True,
        hide_cauter: bool = True,
        check_collision: bool = True,
        active_vision_mode: ActiveVision = ActiveVision.DEACTIVATED,
        individual_agents: bool = False,
        cauter_activation: bool = False,
    ):
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        # Pass image shape to the scene creation function
        create_scene_kwargs["image_shape"] = image_shape

        # Pass transparent_abdominal flag to scene creation function
        create_scene_kwargs["transparent_abdominal_wall"] = transparent_abdominal_wall
        create_scene_kwargs["hide_surgeon_gripper"] = hide_surgeon_gripper
        create_scene_kwargs["hide_assistant_gripper"] = hide_assistant_gripper

        if active_vision_mode == ActiveVision.CAUTER:
            hide_cauter = False
        create_scene_kwargs["hide_cauter"] = hide_cauter

        self.active_vision_mode = active_vision_mode

        # Set active vision
        self.active_vision = True if self.active_vision_mode != ActiveVision.DEACTIVATED else False
        self.individual_agents = individual_agents
        if self.active_vision and self.active_vision_mode != ActiveVision.CAUTER:
            raise ValueError(f"SearchForPointEnv currently only supports active vision for the cauter. {active_vision_mode=}")

        if self.active_vision_mode == ActiveVision.CAUTER:
            create_scene_kwargs["poi_mode"] = POIMode.CAUTER_REACHABLE
        else:
            create_scene_kwargs["poi_mode"] = POIMode.CAMERA_VISIBLE_WITHOUT_ABDOMINAL_WALL

        self.image_shape = image_shape

        # Center of the image recorded by the camera
        self.center_pixel = np.round(np.array(image_shape) / 2.0).astype(np.uint16)

        # Distance from the corner to the center of the image
        self.maximum_pixel_distance_to_center = np.linalg.norm(np.asarray(self.image_shape) / 2.0)
        self.target_distance_camera_poi = target_distance_camera_poi
        self.tolerance_image_center = tolerance_image_center * self.maximum_pixel_distance_to_center
        self.tolerance_target_distance = tolerance_target_distance

        # Check collisions
        self.check_collision = False if active_vision_mode == ActiveVision.DEACTIVATED else check_collision
        # Pass check collison flag to scene creation function
        if self.check_collision:
            create_scene_kwargs["check_collision"] = True
        # How close the instrument needs to be to the target to be considered successful
        self.instrument_touch_tolerance = instrument_touch_tolerance

        self.cauter_activation = cauter_activation

        # Target position
        self.target_position = np.empty(3, dtype=np.float32)
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
        # Camera = 4
        # Cauter = 5 if the cauter can be activated, else 4
        # Assistant Gripper, Surgeon Gripper = 5
        self.camera_action_dimensionality = camera_action_dimensionality = 4
        self.cauter_action_dimensionality = cauter_action_dimensionality = 5 if self.cauter_activation else 4
        setable_actions = camera_action_dimensionality + cauter_action_dimensionality
        setable_actions -= 1 if self.cauter_activation else 0

        # Maximum State Velocity
        if self.active_vision:
            # Check for type
            if isinstance(maximum_state_velocity, np.ndarray):
                # Use one array for both velocities

                if len(maximum_state_velocity) == camera_action_dimensionality:
                    self.camera_max_state_velocity = maximum_state_velocity
                    # The activation of the cauter action is scaled to fall into [-1, 1]
                    self.cauter_max_state_velocity = np.append(maximum_state_velocity, 1.0 / self.time_step) if self.cauter_activation else maximum_state_velocity

                # Split the velocities into two arrays for camera and cauter
                elif len(maximum_state_velocity) == setable_actions:
                    self.camera_max_state_velocity = maximum_state_velocity[:camera_action_dimensionality]
                    # The activation of the cauter action is scaled to fall into [-1, 1]
                    self.cauter_max_state_velocity = maximum_state_velocity[camera_action_dimensionality:]
                    if self.cauter_activation:
                        # The activation (last element of the cauter) is not scaled -> - 1
                        self.cauter_max_state_velocity = np.append(self.cauter_max_state_velocity, 1.0 / self.time_step)
                else:
                    if self.cauter_activation:
                        raise ValueError(f"Invalid maximum_state_velocity array {maximum_state_velocity=} of length {len(maximum_state_velocity)}. Length needs to be {camera_action_dimensionality} or {camera_action_dimensionality + cauter_action_dimensionality - 1}")
                    else:
                        raise ValueError(f"Invalid maximum_state_velocity array {maximum_state_velocity=} of length {len(maximum_state_velocity)}. Length needs to be {camera_action_dimensionality} or {camera_action_dimensionality + cauter_action_dimensionality}")
            elif isinstance(maximum_state_velocity, float):
                self.camera_max_state_velocity = maximum_state_velocity
                # The activation of the cauter action is scaled to fall into [-1, 1]
                self.cauter_max_state_velocity = np.array([maximum_state_velocity] * cauter_action_dimensionality)
                if self.cauter_activation:
                    self.cauter_max_state_velocity = np.append(self.cauter_max_state_velocity, 1.0 / self.time_step)
            else:
                raise ValueError("Invalid maximum_state_velocity type. Needs to be float or np.ndarray. Got {type(maximum_state_velocity)}")
        else:
            if isinstance(maximum_state_velocity, np.ndarray):
                if not len(maximum_state_velocity) == camera_action_dimensionality:
                    raise ValueError(f"If you want to use individual maximal state limits, please pass an array of length {camera_action_dimensionality} as maximum_state_velocity. Received {maximum_state_velocity=} with lenght {len(maximum_state_velocity)}.")
            self.camera_max_state_velocity = maximum_state_velocity
            self.cauter_max_state_velocity = np.NaN

        self.action_type = action_type
        # Continuous action
        if action_type == ActionType.CONTINUOUS:
            self._scale_camera_action = self._scale_continuous_camera_action
            # Active vision -> The camera and an instrument are controlled
            if self.active_vision:
                self._scale_cauter_action = self._scale_continuous_cauter_action

                if self.individual_agents:
                    self._do_action = self._do_action_active_vision_dict
                    self.action_space = spaces.Dict(
                        {
                            "camera": spaces.Box(low=-1.0, high=1.0, shape=(camera_action_dimensionality,), dtype=np.float32),
                            "cauter": spaces.Box(low=-1.0, high=1.0, shape=(cauter_action_dimensionality,), dtype=np.float32),
                        }
                    )
                else:
                    self._do_action = self._do_continuous_action_active_vision_array
                    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(camera_action_dimensionality + cauter_action_dimensionality,), dtype=np.float32)
            # No active vision -> Only the camera is controlled
            else:
                self._do_action = self._do_action_camera
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(camera_action_dimensionality,), dtype=np.float32)
        # Discrete action
        else:
            # Discrete action magnitude
            if isinstance(discrete_action_magnitude, np.ndarray):
                # Use one array for both step sizes
                if len(discrete_action_magnitude) == camera_action_dimensionality:
                    discrete_action_magnitude_camera = discrete_action_magnitude
                    discrete_action_magnitude_cauter = discrete_action_magnitude
                    # The activation (last element of the cauter) is not scaled -> - 1
                    if self.cauter_activation:
                        discrete_action_magnitude_cauter = np.append(discrete_action_magnitude_cauter, 1.0 / self.time_step)

                # Split the step sizes into two arrays for camera and cauter
                # The activation (last element of the cauter) is not scaled -> - 1
                elif len(discrete_action_magnitude) == setable_actions:
                    discrete_action_magnitude_camera = discrete_action_magnitude[:camera_action_dimensionality]
                    discrete_action_magnitude_cauter = discrete_action_magnitude[camera_action_dimensionality:]
                    if self.cauter_activation:
                        discrete_action_magnitude_cauter = np.append(discrete_action_magnitude_cauter, 1.0 / self.time_step)

                else:
                    if self.cauter_activation:
                        raise ValueError(f"Invalid discrete_action_magnitude array {discrete_action_magnitude=} of length {len(discrete_action_magnitude)}. Length needs to be {camera_action_dimensionality} or {camera_action_dimensionality + cauter_action_dimensionality - 1}")
                    else:
                        raise ValueError(f"Invalid discrete_action_magnitude array {discrete_action_magnitude=} of length {len(discrete_action_magnitude)}. Length needs to be {camera_action_dimensionality} or {camera_action_dimensionality + cauter_action_dimensionality}")
            elif isinstance(discrete_action_magnitude, float):
                discrete_action_magnitude_camera = discrete_action_magnitude
                discrete_action_magnitude_cauter = np.array([discrete_action_magnitude] * cauter_action_dimensionality)
                if self.cauter_activation:
                    discrete_action_magnitude_cauter = np.append(discrete_action_magnitude_cauter, 1.0 / self.time_step)
            else:
                raise ValueError("Invalid discrete_action_magnitude type. Needs to be float or np.ndarray. Got {type(discrete_action_magnitude)}")

            self._scale_camera_action = self._scale_discrete_camera_action
            self._discrete_camera_action_lookup = self.create_discrete_action_lookup(camera_action_dimensionality, discrete_action_magnitude_camera)
            self._discrete_camera_action_lookup.flags.writeable = False

            if self.active_vision:
                if self.individual_agents:
                    self.action_space = spaces.Dict(
                        {
                            "camera": spaces.Discrete(camera_action_dimensionality * 2 + 1),
                            "cauter": spaces.Discrete(cauter_action_dimensionality * 2 + 1),
                        }
                    )
                    self._do_action = self._do_action_active_vision_dict
                else:
                    # +/- for camera, +/- for cauter, noop
                    self.action_space = spaces.Discrete(camera_action_dimensionality * 2 + cauter_action_dimensionality * 2 + 1)
                    self._do_action = self._do_discret_action_active_vision_array

                self._scale_cauter_action = self._scale_discrete_cauter_action
                self._discrete_cauter_action_lookup = self.create_discrete_action_lookup(cauter_action_dimensionality, discrete_action_magnitude_cauter)
                self._discrete_cauter_action_lookup.flags.writeable = False

            else:
                self.action_space = spaces.Discrete(camera_action_dimensionality * 2 + 1)
                self._do_action = self._do_action_camera

        if self.active_vision_mode not in (ActiveVision.CAUTER, ActiveVision.DEACTIVATED):
            raise NotImplementedError("Active vision is currently only implemented for the cauter")

        ###################
        # Observation Space
        ###################

        # State observations
        if observation_type == ObservationType.STATE:
            # Position of poi -> 3
            # Pose of camera -> 7
            # Ptsd state of camera -> 4
            observations_size = 3 + 7 + 4

            if self.active_vision:
                # Pose of active vision instrument -> 7
                # Ptsda of active vision instrument -> 5
                observations_size += 7 + 5

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
        self.reward_features = {}
        self.reward_info = {}

        # All parameters of the reward function that are not passed default to 0.0
        self.reward_amount_dict = defaultdict(float) | reward_amount_dict[self.active_vision_mode]

        if self.active_vision:
            self._get_reward_features = self._get_reward_features_active_vision

        # Callback functions called on reset
        self.on_reset_callbacks = on_reset_callbacks if on_reset_callbacks is not None else []

    def _init_sim(self) -> None:
        super()._init_sim()

        self.camera: PivotizedCamera = self.scene_creation_result["camera"]
        self.assistant_gripper: Union[PivotizedArticulatedInstrument, None] = self.scene_creation_result["assistant_gripper"]
        self.surgeon_gripper: Union[PivotizedArticulatedInstrument, None] = self.scene_creation_result["surgeon_gripper"]
        self.cauter: Union[Cauter, None] = self.scene_creation_result["cauter"]
        self.poi: PointOfInterest = self.scene_creation_result["poi"]
        self.contact_listener: Dict[str, Sofa.Core.ContactListener] = self.scene_creation_result["contact_listener"]
        if self.check_collision and "cauter" not in self.contact_listener:
            raise KeyError("Cauter contact listener not found in contact_listener dict, but collision checking is enabled.")

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

    def _scale_continuous_camera_action(self, action: np.ndarray) -> np.ndarray:
        """
        Policy is output is clipped to [-1, 1] e.g. [-0.3, 0.8, 1].
        We want to scale that to the maximum velocities defined
        in ``maximum_state_velocity`` in [angle, angle, angle, mm, angle] / step.
        and further to per second (because delta T is not 1 second).
        """
        continuous_action = self.time_step * self.camera_max_state_velocity * action[: self.camera_action_dimensionality]
        return continuous_action

    def _scale_continuous_cauter_action(self, action: np.ndarray) -> np.ndarray:
        """
        Policy is output is clipped to [-1, 1] e.g. [-0.3, 0.8, 1].
        We want to scale that to the maximum velocities defined
        in ``maximum_state_velocity`` in [angle, angle, angle, mm, angle] / step.
        and further to per second (because delta T is not 1 second).
        """
        continuous_action = self.time_step * self.cauter_max_state_velocity * action[-self.cauter_action_dimensionality :]
        return continuous_action

    def _scale_discrete_camera_action(self, action: int) -> np.ndarray:
        """Maps camera action indices to a motion delta."""
        return self._discrete_camera_action_lookup[action]

    def _scale_discrete_cauter_action(self, action: int) -> np.ndarray:
        """Maps cauter action indices to a motion delta."""
        return self._discrete_cauter_action_lookup[action]

    def _do_action(self, action) -> None:
        """Satisfy ABC."""
        pass

    def _do_action_active_vision_dict(self, action: Dict[str, Union[np.ndarray, int]]) -> None:
        self._do_action_camera(action["camera"])
        self.cauter.do_action(self._scale_cauter_action(action["cauter"]))

    def _do_continuous_action_active_vision_array(self, action: np.ndarray) -> None:
        self._do_action_camera(action)
        self.cauter.do_action(self._scale_cauter_action(action))

    def _do_discret_action_active_vision_array(self, action: int) -> None:
        """Apply the action to the selected instrument.

        Action_space: [camera_actions, cauter_actions, no_action].
        """
        no_action = self.camera_action_dimensionality * 2 + self.cauter_action_dimensionality * 2
        cauters_first_action = self.camera_action_dimensionality * 2

        # If the selected action equals the last index of the action space no action will be executed
        if action == no_action:
            pass
        # If the selected action is within the camera actions, the camera executes the action
        elif action < cauters_first_action:
            self._do_action_camera(action)
        # Else the cauter will execute the action
        elif action < no_action:
            action = action - cauters_first_action
            self.cauter.do_action(self._scale_cauter_action(action))
        else:
            raise ValueError(f"The selected discrete action is greater than the action space. {action=} {self.action_space.n}")

    def _do_action_camera(self, action: Union[np.ndarray, int]) -> None:
        """Apply the ``camera`` action to the simulation."""
        self.camera.set_state(self.camera.get_state() + self._scale_camera_action(action))

    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        """Step function of the environment that applies the action to the simulation and returns observation, reward, done signal, and info."""
        maybe_rgb_observation = super().step(action)

        observation = self._get_observation(maybe_rgb_observation)
        reward = self._get_reward()
        terminated = self._get_done()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _get_reward_features(self, previous_reward_features: dict) -> dict:
        """Get the features that may be used to assemble the reward function

        Features:
            - poi_is_in_frame: Point of interest is in the camera frame.
            - relative_camera_distance_error_to_poi: Distance between the desired and actual camera to point of interest distance normalized by the desired distance (``abs(distance)/target_distance``).
            - delta_relative_camera_distance_error_to_poi: Change in distance between the desired and actual camera to poi distance.
            - relative_pixel_distance_poi_to_image_center: Distance between the image center and the poi in pixels normalized by the maximum pixel distance in frame.
            - delta_relative_pixel_distance_poi_to_image_center: Change in distance between the center and the poi in the image.
            - sucessful_task: Task is successful if the poi is in the camera frame, the camera is close enough to the poi and the poi is centered in the image.
        """

        reward_features = {}

        cartesian_camera_position = self.camera.get_pose()[:3]
        cartesian_poi_position = self.target_position
        cartesian_distance_poi_to_camera = np.linalg.norm(cartesian_poi_position - cartesian_camera_position)
        pixel_poi_position = np.asarray(world_to_pixel_coordinates(cartesian_poi_position, self.camera.sofa_object))
        poi_is_in_frame = np.all(np.logical_and(pixel_poi_position < self.image_shape, np.all(pixel_poi_position > 0)))
        distance_poi_to_center = np.linalg.norm(pixel_poi_position - self.center_pixel)

        reward_features["poi_is_in_frame"] = poi_is_in_frame
        reward_features["relative_camera_distance_error_to_poi"] = abs(self.target_distance_camera_poi - cartesian_distance_poi_to_camera) / self.target_distance_camera_poi
        reward_features["delta_relative_camera_distance_error_to_poi"] = reward_features["relative_camera_distance_error_to_poi"] - previous_reward_features["relative_camera_distance_error_to_poi"]
        reward_features["relative_pixel_distance_poi_to_image_center"] = distance_poi_to_center / self.maximum_pixel_distance_to_center
        reward_features["delta_relative_pixel_distance_poi_to_image_center"] = reward_features["relative_pixel_distance_poi_to_image_center"] - previous_reward_features["relative_pixel_distance_poi_to_image_center"]

        camera_position_in_tolerance = abs(self.target_distance_camera_poi - cartesian_distance_poi_to_camera) < self.tolerance_target_distance
        poi_in_image_center = distance_poi_to_center < self.tolerance_image_center
        reward_features["successful_task"] = camera_position_in_tolerance and poi_in_image_center and poi_is_in_frame

        return reward_features

    def cauter_to_target_distance(self):
        """Calculates the distance between the ``cauter`` and the target distance"""
        return np.linalg.norm(self.target_position - self.cauter.get_collision_center_position())

    def _get_reward_features_active_vision(self, previous_reward_features: dict):
        """
        Get the features for active vision that may be used to assemble the reward function.

        Features:
            - collision_cauter: Cauter has a collision.
            - relative_distance_cauter_target: Relative distance between the cauter and the point of interest.
            - delta_distance_cauter_target: Difference between the current the the previous relative distance between the cauter and the point of interest.
            - cauter_touches_target: ``True`` if the distance between the tip position of the cauter and the poi is smaller than ``instrument_touch_tolerance``.
            - successful_task: ``True`` if the cauter touches the target and is activated.
            - cauter_action_violated_state_limits: Cauter violated the state limits.
        """

        reward_features = {}

        cauter_to_target_distance = self.cauter_to_target_distance()

        if self.active_vision_mode == ActiveVision.CAUTER:
            reward_features["collision_cauter"] = self.contact_listener["cauter"].getNumberOfContacts() if self.check_collision else 0.0
            reward_features["relative_distance_cauter_target"] = cauter_to_target_distance / self.distance_cauter_rcm_to_poi
            reward_features["relative_delta_distance_cauter_target"] = reward_features["relative_distance_cauter_target"] - previous_reward_features["relative_distance_cauter_target"]
            reward_features["cauter_touches_target"] = cauter_to_target_distance < self.instrument_touch_tolerance
            reward_features["cauter_action_violated_state_limits"] = self.cauter.last_set_state_violated_state_limits
            if self.cauter_activation:
                reward_features["successful_task"] = reward_features["cauter_touches_target"] and self.cauter.active
            else:
                reward_features["successful_task"] = reward_features["cauter_touches_target"]

        return reward_features

    def _get_reward(self) -> float:
        """Reward function of the Search for Point Task."""
        reward = 0.0
        self.reward_info = {}

        reward_features = self._get_reward_features(self.reward_features) if self.active_vision_mode == ActiveVision.DEACTIVATED else self._get_reward_features_active_vision(self.reward_features)

        # we change the values of the dict -> do a copy (deepcopy not necessary, because the value itself is not manipulated)
        self.reward_features = reward_features.copy()

        for key, value in reward_features.items():
            self.reward_info[f"reward_{key}"] = self.reward_amount_dict[key] * value
            reward += self.reward_info[f"reward_{key}"]

        self.reward_info["reward"] = reward

        return float(reward)

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

    def _get_done(self) -> bool:
        """Look up if the episode is finished."""
        return self.reward_features["successful_task"]

    def _get_observation(self, rgb_observation: Union[np.ndarray, None]) -> Union[np.ndarray, dict]:
        """Assembles the correct observation based on the ``ObservationType``."""

        if self.observation_type == ObservationType.RGB:
            observation = rgb_observation
        elif self.observation_type == ObservationType.RGBD:
            observation = self.observation_space.sample()
            observation[:, :, :3] = rgb_observation
            observation[:, :, 3:] = self.get_depth()
        else:
            state_dict = {}
            state_dict["poi_position"] = self.poi.get_position()
            state_dict["camera_pose"] = self.camera.get_pose()
            state_dict["camera_ptsd"] = self.camera.get_state()
            if self.active_vision:
                state_dict["cauter_pose"] = self.cauter.get_pose()
                state_dict["cauter_ptsda"] = self.cauter.get_ptsda_state()

            observation = np.concatenate(tuple(state_dict.values()))

        return observation

    def collision(self) -> bool:
        """Checks for collision.

        Returns:
            in_collision (bool): ``True`` if collision else ``False``.
        """
        for key in self.contact_listener:
            if self.contact_listener[key].getNumberOfContacts():
                return True
        return False

    def reset(self, seed: Union[int, np.random.SeedSequence, None] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Union[np.ndarray, None], Dict]:
        super().reset(seed)

        # Seed the instruments
        if self.unconsumed_seed:
            seeds = self.seed_sequence.spawn(5)
            self.camera.seed(seed=seeds[0])
            if self.assistant_gripper is not None:
                self.assistant_gripper.seed(seed=seeds[1])
            if self.surgeon_gripper is not None:
                self.surgeon_gripper.seed(seed=seeds[2])
            if self.cauter is not None:
                self.cauter.seed(seed=seeds[3])
            self.poi.seed(seed=seeds[4])
            self.unconsumed_seed = False

        # Reset the instruments
        if self.assistant_gripper is not None:
            self.assistant_gripper.reset_state()
        if self.surgeon_gripper is not None:
            self.surgeon_gripper.reset_state()
        if self.cauter is not None:
            self.cauter.reset_state()
        self.camera.reset_state()

        # Reset the poi
        self.poi.reset()
        self.target_position = self.poi.get_position()

        # Clear the episode info values
        for key in self.episode_info:
            self.episode_info[key] = 0.0

        # and the reward info dict
        self.reward_info = {}

        # and the reward_features, except the ones needed to calculate the delta
        self.reward_features = {}

        # recalculate the previous reward
        if self.active_vision:
            self.distance_cauter_rcm_to_poi = np.linalg.norm(self.poi.get_position() - self.cauter.get_rcm_position())
            self.reward_features["relative_distance_cauter_target"] = self.cauter_to_target_distance() / self.distance_cauter_rcm_to_poi
        else:
            cartesian_camera_position = self.camera.get_pose()[:3]
            cartesian_poi_position = self.poi.get_pose()[:3]
            pixel_poi_position = np.asarray(world_to_pixel_coordinates(self.poi.get_pose()[:3], self.camera.sofa_object))
            cartesian_distance_poi_to_camera = np.linalg.norm(cartesian_poi_position - cartesian_camera_position)
            distance_poi_to_center = np.linalg.norm(pixel_poi_position - self.center_pixel)

            self.reward_features["relative_camera_distance_error_to_poi"] = abs(self.target_distance_camera_poi - cartesian_distance_poi_to_camera) / self.target_distance_camera_poi
            self.reward_features["relative_pixel_distance_poi_to_image_center"] = distance_poi_to_center / self.maximum_pixel_distance_to_center

        # Execute any callbacks passed to the env.
        for callback in self.on_reset_callbacks:
            callback(self)

        return self._get_observation(rgb_observation=self._maybe_update_rgb_buffer()), {}


if __name__ == "__main__":
    import pprint
    import time
    from collections import deque

    pp = pprint.PrettyPrinter()

    env = SearchForPointEnv(
        render_mode=RenderMode.HUMAN,
        image_shape=(600, 600),
        observation_type=ObservationType.STATE,
        active_vision_mode=ActiveVision.CAUTER,
        maximum_state_velocity=np.array([20.0, 20.0, 20.0, 20.0]),
        discrete_action_magnitude=np.array([20.0, 20.0, 20.0, 20.0]),
        action_type=ActionType.CONTINUOUS,
        individual_agents=True,
        check_collision=True,
        cauter_activation=False,
    )
    env.reset()
    done = False
    fps_list = deque(maxlen=100)

    counter = 0
    while not done:
        for _ in range(500):
            start = time.perf_counter()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if counter % 300 == 0:
                env.reset()
                counter = 0
            counter += 1
            end = time.perf_counter()
            fps = 1 / (end - start)
            fps_list.append(fps)
            print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")

        env.reset()
