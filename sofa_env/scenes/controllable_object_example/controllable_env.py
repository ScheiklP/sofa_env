import gymnasium.spaces as spaces
import numpy as np

from pathlib import Path
from sofa_env.base import SofaEnv, RenderMode, RenderFramework
from typing import Optional, Tuple, Union, Any, Dict


class ControllableEnv(SofaEnv):
    def __init__(
        self,
        scene_path: Union[str, Path],
        time_step: float = 0.01,
        frame_skip: int = 1,
        render_mode: RenderMode = RenderMode.HUMAN,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        create_scene_kwargs: Optional[dict] = None,
        maximum_velocity: float = 50.0,
    ) -> None:
        super().__init__(
            scene_path,
            time_step=time_step,
            frame_skip=frame_skip,
            render_mode=render_mode,
            render_framework=render_framework,
            create_scene_kwargs=create_scene_kwargs,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(600, 600, 3), dtype=np.uint8)

        self.maximum_velocity = maximum_velocity

    def _do_action(self, action: np.ndarray) -> None:
        scaled_action = action * self.time_step * self.maximum_velocity

        old_pose = self.scene_creation_result["controllable_sphere"].get_pose()

        new_pose = old_pose + np.append(scaled_action, np.array([0, 0, 0, 1]))

        self.scene_creation_result["controllable_sphere"].set_pose(new_pose)

    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        rgb_observation = super().step(action)
        info = {"sphere_position": self.scene_creation_result["controllable_sphere"].get_pose()[:3]}
        terminated = info["sphere_position"][0] <= -130.0
        reward = 10.0 if done else 0.0
        return rgb_observation, reward, terminated, False, info

    def reset(self, seed: Union[int, np.random.SeedSequence, None] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Union[np.ndarray, None], Dict]:
        """Reset the state of the environment and return the initial observation."""
        # Reset from parent class -> calls the simulation's reset function
        return super().reset(seed)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    scene_description = here / Path("scene_description.py")

    env = ControllableEnv(
        scene_path=scene_description,
        render_mode=RenderMode.HUMAN,
        render_framework=RenderFramework.PYGAME,
    )

    env.reset()

    action = np.array([-1, 0, 0], dtype=np.float32)
    done = False

    try:
        while not done:
            _, _, done, _, _ = env.step(action)
    except KeyboardInterrupt:
        pass

    env.close()
