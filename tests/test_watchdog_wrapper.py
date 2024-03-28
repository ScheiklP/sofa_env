import time
import numpy as np

import gymnasium as gym
from sofa_env.wrappers.watchdog import WatchdogEnv


class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,))
        self.action_space = gym.spaces.Discrete(2)
        self.life_time = 0

    def step(self, action):

        self.life_time += 1

        if action == 1:
            time.sleep(1.0)

        return np.zeros(3), 0.0, False, False, {}

    def reset(self, seed=None, options=None):
        return np.zeros(3), {}


class TestWatchdogWrapper:
    def test_watchdog_wrapper(self):
        env = WatchdogEnv(
            env_fn=DummyEnv,
            step_timeout_sec=0.5,
            reset_process_on_env_reset=True,
        )

        obs, info = env.reset()

        actions = [0, 1, 0, 0]
        expected_life_times = [1, 0, 1, 2]

        for action, expected_life_time in zip(actions, expected_life_times):
            obs, reward, terminated, truncated, info = env.step(action)
            if action == 1:
                assert not terminated
                assert truncated
            else:
                assert not terminated
                assert not truncated

            life_time = env.get_attr("life_time")
            assert life_time == expected_life_time

    def test_watchdog_wrapper_reset_with_process_reset(self):
        env = WatchdogEnv(
            env_fn=DummyEnv,
            step_timeout_sec=0.5,
            reset_process_on_env_reset=True,
        )

        env.reset()

        for _ in range(3):
            env.step(0)

        life_time = env.get_attr("life_time")
        assert life_time == 3

        env.reset()

        life_time = env.get_attr("life_time")
        assert life_time == 0

        env.close()

    def test_watchdog_wrapper_reset_without_process_reset(self):
        env = WatchdogEnv(
            env_fn=DummyEnv,
            step_timeout_sec=0.5,
            reset_process_on_env_reset=False,
        )

        env.reset()

        for _ in range(3):
            env.step(0)

        life_time = env.get_attr("life_time")
        assert life_time == 3

        env.reset()

        life_time = env.get_attr("life_time")
        assert life_time == 3

        env.close()
