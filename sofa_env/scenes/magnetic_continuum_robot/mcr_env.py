from typing import Union, Tuple, Optional, Any, Dict
from pathlib import Path
from enum import Enum, unique
import gymnasium.spaces as spaces
import numpy as np
from collections import defaultdict

from sofa_env.base import SofaEnv, RenderMode, RenderFramework
from sofa_env.scenes.magnetic_continuum_robot.mcr_sim.mcr_controller_sofa import ControllerSofa

HERE = Path(__file__).resolve().parent
FLAT_SCENE_DESCRIPTION_FILE_PATH = HERE / "scene_description_2d.py"
AORTIC_SCENE_DESCRIPTION_FILE_PATH = HERE / "scene_description_3d.py"
FLAT_CATHETER_DESTINATION_EXIT_POINT = np.array([0.101129, 0.0238015, 0.002])
AORTIC_CATHETER_DESTINATION_EXIT_POINT = np.array([-0.0101583, -0.180636, 0.0345185])


@unique
class ObservationType(Enum):
    RGB = 0
    STATE = 1


@unique
class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


@unique
class EnvType(Enum):
    FLAT = 0
    AORTIC = 1


class MCREnv(SofaEnv):
    """Magnetic Continuum Robot Environment (MCREnv)

    The goal of this environment is to rotate and move the catheter through the artery to a specified destination.
    The workspace size may be adapted through the ``create_scene_kwargs``. See ``scene_description.py`` for more details.
    There are two scenes available (flat or aortic) which can be selected through the ``env_type`` parameter.

    Args:
        image_shape (Tuple[int, int]): Height and Width of the rendered images.
        create_scene_kwargs (Optional[dict]): A dictionary to pass additional keyword arguments to the ``createScene`` function.
        observation_type (ObservationType): Whether to return RGB images or an array of states as the observation.
        action_type (ActionType): Discrete or continuous actions to define the action space of the environment.
        time_step (float): size of simulation time step in seconds (default: 0.1).
        frame_skip (int): number of simulation time steps taken (call ``_do_action`` and advance simulation) each time step is called (default: 1).
        settle_steps (int): How many steps to simulate without returning an observation after resetting the environment.
        render_mode (RenderMode): Create a window (``RenderMode.HUMAN``), run headless (``RenderMode.HEADLESS``), or do not create a render buffer at all (``RenderMode.NONE``).
        render_framework (RenderFramework): choose between pyglet and pygame for rendering
        reward_amount_dict (dict): Dictionary to weigh the components of the reward function.
        target_position (Optional[np.ndarray]): Target position of the catheter tip in the scene.
        env_type (EnvType): Whether to use the flat (``EnvType.FLAT``) or aortic (``EnvType.AORTIC``) scene.
        target_distance_threshold (float): Distance threshold for the reward function (default: 0.015).
        num_catheter_tracking_points (int): Number of points on the catheter to track (default: 4).
    """

    def __init__(
        self,
        image_shape: Tuple[int, int] = (400, 400),
        create_scene_kwargs: Optional[dict] = None,
        observation_type: ObservationType = ObservationType.STATE,
        action_type: ActionType = ActionType.CONTINUOUS,
        time_step: float = 0.1,
        frame_skip: int = 1,
        settle_steps: int = 10,
        render_mode: RenderMode = RenderMode.HUMAN,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        reward_amount_dict={
            "tip_pos_distance_to_dest_pos": -0.0,
            "delta_tip_pos_distance_to_dest_pos": -0.0,
            "workspace_constraint_violation": -0.0,
            "successful_task": 0.0,
        },
        target_position: Optional[np.ndarray] = None,
        env_type: EnvType = EnvType.FLAT,
        target_distance_threshold: float = 0.015,
        num_catheter_tracking_points: int = 4,
    ):
        # Pass image shape to the scene creation function
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape

        self.target_distance_threshold = target_distance_threshold
        self.num_catheter_tracking_points = num_catheter_tracking_points

        self.env_type = env_type
        if self.env_type == EnvType.FLAT:
            self.target_position = target_position if target_position is not None else FLAT_CATHETER_DESTINATION_EXIT_POINT
            self.scene_path = FLAT_SCENE_DESCRIPTION_FILE_PATH
        elif self.env_type == EnvType.AORTIC:
            self.target_position = target_position if target_position is not None else AORTIC_CATHETER_DESTINATION_EXIT_POINT
            self.scene_path = AORTIC_SCENE_DESCRIPTION_FILE_PATH

        super().__init__(
            scene_path=self.scene_path,
            time_step=time_step,
            frame_skip=frame_skip,
            render_mode=render_mode,
            render_framework=render_framework,
            create_scene_kwargs=create_scene_kwargs,
        )

        # Observation type of the env observation_space
        self.observation_type = observation_type

        # How many simulation steps to wait before starting the episode
        self._settle_steps = settle_steps

        ###########################
        # Set up observation spaces
        ###########################
        if self.observation_type == ObservationType.STATE:
            # Tip of the instrument: 3 position + 4 quaternion
            # Position of n points on the catheter: num_catheter_tracking_points * 3
            # Magnetic field: 3
            # Target position: 3
            observations_size = 3 + 4 + self.num_catheter_tracking_points * 3 + 3 + 3
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observations_size,), dtype=np.float32)
        elif self.observation_type == ObservationType.RGB:
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (3,), dtype=np.uint8)

        ######################
        # Set up action spaces
        ######################
        action_dimensionality = 3
        self.action_type = action_type
        if action_type == ActionType.CONTINUOUS:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dimensionality,), dtype=np.float32)
        else:
            raise NotImplementedError("Only continuous action space is implemented.")

        #########################
        # Episode specific values
        #########################
        self.reward_info = {}
        self.reward_features = {}

        # All parameters of the reward function that are not passed default to 0.0
        self.reward_amount_dict = defaultdict(float) | reward_amount_dict

    def reset(self, seed: Union[int, np.random.SeedSequence, None] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Union[np.ndarray, None], Dict]:
        super().reset(seed)

        # Reset mcr controller
        self.mcr_controller_sofa.reset()

        # Reset reward info dict
        self.reward_info = {}

        # Fill the first values used as previous_reward_features
        self.reward_features = {}
        self.reward_features["tip_pos_distance_to_dest_pos"] = self._get_distance_tip_to_dest()

        # Animate several timesteps without actions until simulation settles
        for _ in range(self._settle_steps):
            self.sofa_simulation.animate(self._sofa_root_node, self._sofa_root_node.getDt())

        return self._get_observation(image_observation=self._maybe_update_rgb_buffer()), {}

    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        """Step function of the environment that applies the action to the simulation and returns observation, reward, done signal, and info."""

        image_observation = super().step(action)
        observation = self._get_observation(image_observation)
        reward = self._get_reward()
        terminated = self._get_done()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _get_observation(self, image_observation: Union[np.ndarray, None]) -> Union[np.ndarray, dict]:
        """Assembles the correct observation based on the ``ObservationType``."""

        if self.observation_type == ObservationType.RGB:
            return image_observation
        elif self.observation_type == ObservationType.STATE:
            obs = {
                "position-quaternion-catheter-tip": self.mcr_controller_sofa.get_pos_quat_catheter_tip(),
                "position-catheter": self.mcr_controller_sofa.get_pos_catheter(num_points=self.num_catheter_tracking_points),
                "magnetic-field-des": self.mcr_controller_sofa.get_mag_field_des(),
                "target-position": self.target_position,
            }
            return np.concatenate(tuple(obs.values()))
        else:
            return {}

    def _get_reward_features(self, previous_reward_features: dict) -> dict:
        """Get the features that may be used to assemble the reward function

        Features:
            - successful_task (bool): Whether the task is done
            - tip_pos_distance_to_dest_pos (float): Distance between catheter tip and destination point
            - delta_tip_pos_distance_to_dest_pos (float): Change in distance between catheter tip and destination point
            - workspace_constraint_violation (bool): Whether the workspace constraint is violated
        """

        reward_features = {}
        # Check if task is done
        if previous_reward_features["tip_pos_distance_to_dest_pos"] < self.target_distance_threshold:
            reward_features["successful_task"] = True
        else:
            reward_features["successful_task"] = False

        # Distance between catheter tip and destination point
        reward_features["tip_pos_distance_to_dest_pos"] = self._get_distance_tip_to_dest()

        # Change in distances
        reward_features["delta_tip_pos_distance_to_dest_pos"] = reward_features["tip_pos_distance_to_dest_pos"] - previous_reward_features["tip_pos_distance_to_dest_pos"]

        # Tip extraction penalty
        reward_features["workspace_constraint_violation"] = self.mcr_controller_sofa.invalid_action

        return reward_features

    def _get_reward(self) -> float:
        """Retrieve the reward features and scale them with the ``reward_amount_dict``."""

        reward = 0.0
        self.reward_info = {}
        reward_features = self._get_reward_features(previous_reward_features=self.reward_features)
        self.reward_features = reward_features.copy()

        for key, value in reward_features.items():
            value = self.reward_amount_dict[key] * value
            if "distance" in key:
                value = value * self.cartesian_scaling_factor
            self.reward_info[f"reward_{key}"] = value
            reward += self.reward_info[f"reward_{key}"]

        self.reward_info["reward"] = reward
        return float(reward)

    def _get_done(self) -> bool:
        """Look up if the episode is finished."""
        return self.reward_features["successful_task"]

    def _get_info(self) -> dict:
        """Assemble the info dictionary."""
        self.info = {}
        self.episode_info = {}
        return {**self.info, **self.reward_info, **self.episode_info, **self.reward_features}

    def _get_distance_tip_to_dest(self):
        """Calculate the distance between the catheter tip and the destination point."""
        return np.linalg.norm(self.target_position - self.mcr_controller_sofa.get_pos_quat_catheter_tip()[0:3])

    def _do_action(self, action: np.ndarray) -> None:
        """Apply action to the simulation."""
        self.mcr_controller_sofa.rotateZ(action[0])
        self.mcr_controller_sofa.rotateX(action[1])
        self.mcr_controller_sofa.insertRetract(action[2])

    def _init_sim(self):
        "Initialise simulation."
        super()._init_sim()
        self.mcr_controller_sofa: ControllerSofa = self.scene_creation_result["mcr_controller_sofa"]
        self.mcr_environment = self.scene_creation_result["mcr_environment"]
        vessel_positions = self.mcr_environment.get_vessel_tree_positions()
        self.cartesian_scaling_factor = 1 / np.linalg.norm(np.min(vessel_positions, axis=0) - np.max(vessel_positions, axis=0))


if __name__ == "__main__":
    env = MCREnv(env_type=EnvType.AORTIC)
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        if done:
            break
    env.close()
