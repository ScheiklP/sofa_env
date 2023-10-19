import gymnasium as gym
import numpy as np
from typing import Callable


class EpisodicEnvWrapper(gym.Wrapper):
    """Wraps an environment to make it episodic.

    The environment is considered done after a fixed number of steps.

    Args:
        env (gymnasium.Env): The environment to wrap.
        reward_accumulation_func (Callable): Function that takes a list of rewards and returns a single reward.
        episode_length (int): The fixed length of an episode.
        stop_on_terminated (bool): Whether to stop the episode when the environment returns ``terminated=True``.
        stop_on_truncated (bool): Whether to stop the episode when the environment returns ``truncated=True``.
        return_buffers (bool): Whether to return the observation, reward, ... buffers of the episode.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_accumulation_func: Callable,
        episode_length: int,
        stop_on_terminated: bool = False,
        stop_on_truncated: bool = False,
        return_buffers: bool = False,
    ):
        super().__init__(env)
        self.reward_accumulation_func = reward_accumulation_func
        self.dt = self.env.time_step * self.frame_skip
        self.episode_length = episode_length
        self.stop_on_terminated = stop_on_terminated
        self.stop_on_truncated = stop_on_truncated
        self.return_buffers = return_buffers

    def step(self, actions):
        """Applies a list of actions to the environment and accumulates the rewards.

        Args:
            actions (list): List of actions to apply to the environment. Must match the ``episode_length``.

        Returns:
            If ``return_buffers`` is ``False``:
                observation (np.ndarray): The last observation of the environment.
                reward (float): The accumulated reward.
                terminated (bool): Whether the episode is terminated in the last step.
                truncated (bool): Whether the episode is truncated in the last step.
                info (dict): The last info dictionary of the environment.

            If ``return_buffers`` is ``True``:
                observation_buffer (np.ndarray): The observations of the episode.
                reward_buffer (np.ndarray): The rewards of the episode.
                terminated_buffer (np.ndarray): The terminated flags of the episode.
                truncated_buffer (np.ndarray): The truncated flags of the episode.
                info_list (List[dict]): The info dictionaries of the episode.
                reward (float): The accumulated reward.
        """

        if not len(actions) == self.episode_length:
            raise ValueError(f"Number of actions ({len(actions)}) does not match specified episode length ({self.episode_length}).")

        if self.return_buffers:
            observation_buffer = np.zeros((self.episode_length,) + self.observation_space.shape, dtype=self.observation_space.dtype)
            terminated_buffer = np.zeros(self.episode_length, dtype=np.bool_)
            truncated_buffer = np.zeros(self.episode_length, dtype=np.bool_)
            info_list = []

        reward_buffer = np.zeros(self.episode_length, dtype=np.float32)

        for step, action in enumerate(actions):
            obs, rew, terminated, truncated, info = self.env.step(action)

            reward_buffer[step] = rew

            if self.return_buffers:
                observation_buffer[step] = obs
                terminated_buffer[step] = terminated
                truncated_buffer[step] = truncated
                info_list.append(info)

            if self.stop_on_terminated and terminated:
                break

            if self.stop_on_truncated and truncated:
                break

        reward = self.reward_accumulation_func(reward_buffer)

        if self.return_buffers:
            return observation_buffer, reward_buffer, terminated_buffer, truncated_buffer, info_list, reward
        else:
            return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    """Example usage of the EpisodicEnvWrapper."""
    import numpy as np
    from sofa_env.scenes.reach.reach_env import ReachEnv, RenderMode, ObservationType, ActionType

    episode_length = 300
    env = ReachEnv(
        observation_type=ObservationType.STATE,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.CONTINUOUS,
        image_shape=(1024, 1024),
        frame_skip=1,
        time_step=1 / 30,
        observe_target_position=True,
        maximum_robot_velocity=35.0,
    )

    env = EpisodicEnvWrapper(
        env=env,
        reward_accumulation_func=np.sum,
        episode_length=episode_length,
        stop_on_terminated=True,
        stop_on_truncated=True,
        return_buffers=True,
    )

    reset_obs, reset_info = env.reset()

    action_plan = np.ones((episode_length, 3)) * (env.target_position - env.end_effector.get_pose()[:3]) / (35.0 * env.dt * 0.001 * episode_length)

    observations, rewards, terminated, truncated, infos, cumulative_reward = env.step(action_plan)
