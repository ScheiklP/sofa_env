import gymnasium as gym

from time import perf_counter, sleep


class RealtimeWrapper(gym.Wrapper):
    """Wraps an environment to wait with the next step, until ``time_step*frame_skip`` seconds have passed."""

    def step(self, *args, **kwargs):
        return_values = self.env.step(*args, **kwargs)

        now = perf_counter()
        sleep(max(0, self._delta_t - (now - self._last_t)))
        self._last_t = perf_counter()

        return return_values

    def reset(self, **kwargs):
        return_values = self.env.reset(**kwargs)

        self._delta_t = self.env.time_step * self.env.frame_skip
        self._last_t = perf_counter()

        return return_values
