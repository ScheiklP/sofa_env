import gymnasium as gym
import numpy as np
import multiprocessing as mp

from datetime import datetime
from typing import Callable, Optional
from collections import defaultdict

from typing import Any, Callable, Dict, Optional
import cloudpickle


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

    From https://github.com/DLR-RM/stable-baselines3

    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    """
    From https://github.com/DLR-RM/stable-baselines3
    """

    parent_remote.close()

    env = env_fn_wrapper.var()

    reset_info: Optional[Dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                remote.send((observation, reward, terminated, truncated, info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class WatchdogEnv:
    """
    This Env features a watchdog in its asynchronous step function to reset environments that take longer than
    ``step_timeout_sec`` for one step. This might happen when unstable deformations cause SOFA to hang.

    Resetting a SOFA scene that features topological changes such as removing/cutting tetrahedral elements does not
    restore the initial number of elements in the meshes. Manually removing and adding elements to SOFA's simulation
    tree technically works, but is sometimes quite unreliable and prone to memory leaks. This Env avoids this
    problem by creating a completely new environment and simulation when a reset signal is sent to the environment.

    Notes:
        If an environment is reset by the step watchdog, the returned values for this environment will be:
        ``reset_obs, 0.0, False, True, defaultdict(float) | reset_info``. Meaning it returs the reset observation, a reward of 0.0,
        a false termination signal, a true truncated signal, and a dict with the reset information and defaults returning 0.0,
        if a key is accessed and the reset_info dict. The ``defaultdict`` is used to prevent crashing the ``Monitor`` when accessing the info dict.

    Args:
        env_fn (Callable[[], gymnasium.Env]): Environment constructor.
        step_timeout_sec (Optional[float]): Timeout in seconds for a single step. If a step takes longer than this
            timeout, the environment will be reset. If ``None``, no timeout is used.
        reset_process_on_env_reset (bool): Additionally to a hanging env, close and restart the process of the env at every reset.
    """

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        step_timeout_sec: Optional[float] = None,
        reset_process_on_env_reset: bool = False,
    ) -> None:
        self.waiting = False
        self.closed = False

        # forkserver is way too slow since we need to start a new process on every reset
        ctx = mp.get_context("fork")

        self.remote, self.work_remote = ctx.Pipe()

        args = (self.work_remote, self.remote, CloudpickleWrapper(env_fn))
        # daemon=True: if the main process crashes, we should not cause things to hang
        self.process = ctx.Process(target=_worker, args=args, daemon=True)
        self.process.start()

        self.ctx = ctx
        self.remote.send(("get_spaces", None))
        self.observation_space, self.action_space = self.remote.recv()

        self.env_fn = env_fn
        self.step_timeout = step_timeout_sec
        self.reset_process_on_env_reset = reset_process_on_env_reset

        self.render_mode = self.get_attr("render_mode")

    def step(self, action: np.ndarray):
        self.step_async(action)
        return self.step_wait()

    def step_async(self, action: np.ndarray) -> None:
        self.remote.send(("step", action))
        self.waiting = True

    def step_wait(self):

        success = self.remote.poll(timeout=self.step_timeout)

        if not success:
            print(f"Environment is hanging and will be terminated and restarted " f"({datetime.now().strftime('%H:%M:%S')})")
            self.process.terminate()  # terminate worker
            # clear any data in the pipe
            while self.remote.poll():
                self.remote.recv()
            # start new worker, seed, and reset it
            self._restart_process()

        result = self.remote.recv()
        self.waiting = False

        if not success:
            # if the environments was just reset it will send an extra message that must be consumed.
            # in addition, the observation and done state and reset info must be updated
            reset_obs, reset_info = result
            reset_info = defaultdict(float) | reset_info
            # Result order: obs, reward, terminated, truncated, info
            # Similar to time limit -> truncated = True
            result = (reset_obs, 0.0, False, True, reset_info)

        obs, reward, terminated, truncated, info = result

        # if self.reset_process_on_env_reset:
        #     done = terminated or truncated
        #     if done and success:  # do not double-reset environments that were hanging
        #         self.remote.send(("close", None))  # command worker to stop
        #         self.process.join()  # wait for worker to stop
        #         # start new worker, seed, and reset it
        #         self._restart_process()
        #         obs, info = self.remote.recv()  # collect reset observation

        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        # seeds and options are only used once
        options = options if options else {}
        self._seed = seed
        self._options = options

        if self.reset_process_on_env_reset:
            self.process.terminate()  # terminate worker
            # clear any data in the pipe
            while self.remote.poll():
                self.remote.recv()

            # start new workers, seed, and reset them
            self._restart_process()

            result = self.remote.recv()
            reset_obs, reset_info = result

            return reset_obs, reset_info
        else:
            self.remote.send(("reset", (seed, options)))
            result = self.remote.recv()
            reset_obs, reset_info = result
            return reset_obs, reset_info

    def _restart_process(self) -> None:
        """Restarts the worker process with its original ``env_fn``. The
        original pipe is reused. The new environment is seeded and reset, but
        the reset observation is *not* yet collected from the pipe.
        """
        # start new worker
        args = (self.work_remote, self.remote, CloudpickleWrapper(self.env_fn))
        process = self.ctx.Process(target=_worker, args=args, daemon=True)
        process.start()

        # reseed and reset new env
        self.remote.send(("reset", (self._seed, self._options)))
        self.process = process

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            self.remote.recv()
        self.remote.send(("close", None))
        self.process.join()
        self.closed = True

    def get_attr(self, attr_name: str):
        """Return attribute from the environment (see base class)."""
        self.remote.send(("get_attr", attr_name))
        return self.remote.recv()

    def set_attr(self, attr_name: str, value: Any) -> None:
        """Set attribute inside the environment (see base class)."""
        self.remote.send(("set_attr", (attr_name, value)))
        self.remote.recv()

    def env_method(self, method_name: str, *method_args, **method_kwargs):
        """Call instance methods of the environment."""
        self.remote.send(("env_method", (method_name, method_args, method_kwargs)))
        self.remote.recv()

    def render(self, mode: str = "rgb_array"):
        self.remote.send(("render", mode))
        return self.remote.recv()


if __name__ == "__main__":
    import time
    import numpy

    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,))
            self.action_space = gym.spaces.Discrete(2)

        def step(self, action):

            if np.random.rand() < 0.2:
                print("Sleep!")
                time.sleep(1.0)

            return np.zeros(3), 0.0, False, False, {}

        def reset(self, seed=None, options=None):
            return np.zeros(3), {}

    env = WatchdogEnv(
        env_fn=DummyEnv,
        step_timeout_sec=0.5,
        reset_process_on_env_reset=True,
    )

    obs, info = env.reset()

    for _ in range(10):
        print(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(obs, reward, terminated, truncated, info)
