import gymnasium as gym
import numpy as np
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union, Callable, Any
import json


class TrajectoryRecorder(gym.Wrapper):
    """Wraps an environment to record trajectories.

    Args:
        env: The environment to wrap.
        log_dir: The directory to save the trajectories to.
        before_step_callbacks: A list of callbacks that are executed before the environment's step function is called.
        after_step_callbacks: A list of callbacks that are executed after the environment's step function is called.
        before_reset_callbacks: A list of callbacks that are executed before the environment's reset function is called.
        after_reset_callbacks: A list of callbacks that are executed after the environment's reset function is called.
        write_to_disk_callbacks: A list of callbacks that are executed when the episode is done and the trajectory data is written to the disk. Each callback function is called with the TrajectoryRecorder instance, the path to the trajectory directory, and the trajectory number.
        store_info: If True, the info dictionary of the environment is stored in the trajectory.
        metadata: A dictionary of metadata that is stored in the trajectory.
        save_compressed_keys: A list of keys that are saved as compressed numpy arrays.
        only_record_successful_trajectories: If True, only successful trajectories are recorded, i.e. written to disk.
        check_success_hook: A function that is called to determine whether the episode was successful or not; used when only_record_successful_trajectories is True. The function is called with the TrajectoryRecorder instance, the terminated flag, the truncated flag, and the info dictionary of the environment.

    Notes:
        Each trajectory is stored as a new subdirectory in the log directory.
        The naming is derived from the environment's name, and the highest existing index.
        e.g. if the log directory contains ``ExampleEnv_0`` and ``ExampleEnv_2``,
        the next trajectory will be stored in ``ExampleEnv_3``.
    """

    def __init__(
        self,
        env: gym.Env,
        log_dir: Union[Path, str],
        before_step_callbacks: Optional[List[Callable]] = None,
        after_step_callbacks: Optional[List[Callable]] = None,
        before_reset_callbacks: Optional[List[Callable]] = None,
        after_reset_callbacks: Optional[List[Callable]] = None,
        write_to_disk_callbacks: Optional[List[Callable[["TrajectoryRecorder", Path, int], None]]] = None,
        store_info: bool = False,
        metadata: Optional[dict] = None,
        save_compressed_keys: Optional[List[str]] = None,
        only_record_successful_trajectories: bool = False,
        check_success_hook: Optional[Callable[["TrajectoryRecorder", bool, bool, dict[str, Any]], bool]] = None,
    ):
        super().__init__(env)
        self.trajectory = defaultdict(list)
        self.before_step_callbacks = before_step_callbacks or []
        self.after_step_callbacks = after_step_callbacks or []
        self.before_reset_callbacks = before_reset_callbacks or []
        self.after_reset_callbacks = after_reset_callbacks or []
        self.write_to_disk_callbacks = write_to_disk_callbacks or []
        self.store_info = store_info
        self.log_dir = Path(log_dir)
        self.metadata = metadata

        self.save_compressed_keys = save_compressed_keys or []

        self.only_record_successful_trajectories = only_record_successful_trajectories
        if self.only_record_successful_trajectories:
            if check_success_hook is None:
                raise ValueError("check_success_hook must be provided if only_record_successful_trajectories is True.")
            else:
                self.check_success_hook = check_success_hook

        if not self.log_dir.is_dir():
            self.log_dir.mkdir(parents=True)

    def step(self, action):
        # Run all callbacks that should be executed before the step
        for callback in self.before_step_callbacks:
            callback(self)

        observation, reward, terminated, truncated, info = self.env.step(action)

        # Store action, reward, terminated, truncated, and optionally info of T=t
        self.trajectory["action"].append(deepcopy(action))
        self.trajectory["reward"].append(deepcopy(reward))
        self.trajectory["terminated"].append(deepcopy(terminated))
        self.trajectory["truncated"].append(deepcopy(truncated))
        if self.store_info:
            self.trajectory["info"].append(deepcopy(info))

        # Run all callbacks that should be executed after the step
        for callback in self.after_step_callbacks:
            callback(self)

        # Store observation of T=t+1
        if terminated or truncated:
            self.trajectory["terminal_observation"].append(deepcopy(observation))
            if self.only_record_successful_trajectories:
                if self.check_success_hook(self, terminated, truncated, info):
                    self.write_trajectory_to_disk()
                else:
                    self.trajectory = defaultdict(list)
            else:
                self.write_trajectory_to_disk()
        else:
            self.trajectory["observation"].append(deepcopy(observation))

        return (observation, reward, terminated, truncated, info)

    def reset(self, **kwargs):
        self.trajectory = defaultdict(list)
        for callback in self.before_reset_callbacks:
            callback(self)
        observation, reset_info = self.env.reset(**kwargs)
        # Observation of T=0
        self.trajectory["observation"].append(deepcopy(observation))
        for callback in self.after_reset_callbacks:
            callback(self)
        return observation, reset_info

    def write_trajectory_to_disk(self):
        # Generate the base name of the trajectory based on the environment name
        base_name = type(self.env.unwrapped).__name__

        # Get all existing trajectories with the same base name
        existing_trajectories = [trajectory_name for trajectory_name in self.log_dir.iterdir() if trajectory_name.stem.startswith(base_name)]
        # sorting rule for 1, 2, 3 .... 10 instead of 1, 10, ...
        existing_trajectories.sort(key=lambda x: int(x.stem.split("_")[-1]) if x.stem.split("_")[-1].isdigit() else 0)

        # If there are none, start by appending _0 to the base name
        if len(existing_trajectories) == 0:
            trajectory_number = 0
        else:
            # Otherwise, get the last trajectory and increment the number by 1
            last_trajectory = existing_trajectories[-1]
            trajectory_number = int(last_trajectory.stem.split('_')[-1]) + 1
        trajectory_name = base_name + f"_{trajectory_number}"

        # Create a directory for the trajectory
        trajectory_dir = self.log_dir / trajectory_name
        trajectory_dir.mkdir()

        for callback in self.write_to_disk_callbacks:
            callback(self, trajectory_dir, trajectory_number)

        # Remove the arrays that should be compressed from the trajectory dictionary
        for key in self.save_compressed_keys:
            # Pop from trajectory dictionary
            data = self.trajectory.pop(key)
            file_name = trajectory_dir / f"{key}.npz"
            if file_name.is_file():
                raise FileExistsError(f"File {file_name} already exists.")
            np.savez_compressed(file_name, data)

        # Save the rest uncompressed
        np.save(trajectory_dir / "trajectory_dict.npy", self.trajectory)

        # Write metadate to disk in a separate json file
        if self.metadata is not None:
            with open(trajectory_dir / "metadata.json", "w") as outfile:
                json.dump(self.metadata, outfile)

        # Clear the trajectory dictionary
        self.trajectory = defaultdict(list)
