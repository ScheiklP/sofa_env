from sofa_env.wrappers.trajectory_recorder import TrajectoryRecorder
from sofa_env.wrappers.realtime import RealtimeWrapper
from sofa_env.scenes.magnetic_continuum_robot.mcr_env import MCREnv, EnvType
import cv2
from typing import Dict, Tuple
import numpy as np
import time
from collections import deque
from pathlib import Path
import argparse
from pynput import keyboard
import time
import threading
from typing import List


class KeyboardController:
    """Class to interface a keyboard.

    Args:
        id (int): The id of the controller. Defaults to 0.
    """

    def __init__(self, id: int = 0) -> None:
        self.id = id
        self.r_x = 0.0
        self.r_y = 0.0
        self.retract = 0.0
        self.x = 0.0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, daemon=True)
        print("starting keyboard thread")
        self._monitor_thread.start()

    def read(self) -> List[float]:
        """Reads the current state of the controller."""
        return [self.r_x, self.r_y, self.retract]

    def is_alive(self) -> bool:
        return self._monitor_thread.is_alive()

    def _monitor_controller(self):
        """This function is run in a separate thread and constantly monitors the controller."""
        with keyboard.Events() as events:
            for event in events:
                if type(event) == (keyboard.Events.Press):
                    if event.key == keyboard.Key.esc:
                        self.x = 1.0
                        break
                    if event.key.char == "w":
                        self.retract = 1.0
                    elif event.key.char == "s":
                        self.retract = -1.0

                    if event.key.char == "a":
                        self.r_x = -1.0
                    elif event.key.char == "d":
                        self.r_x = 1.0

                    if event.key.char == "q":
                        self.r_y = 1.0
                    elif event.key.char == "e":
                        self.r_y = -1.0
                elif type(event) == keyboard.Events.Release:
                    if event.key.char == "w" or event.key.char == "s":
                        self.retract = 0.0
                    if event.key.char == "a" or event.key.char == "d":
                        self.r_x = 0.0
                    if event.key.char == "q" or event.key.char == "e":
                        self.r_y = 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup human input behavior.")
    parser.add_argument("-rv", "--record_video", action="store_true", help="Record video of the trajectory.")
    parser.add_argument("-rt", "--record_trajectory", action="store_true", help="Record the full trajectory.")
    parser.add_argument("-i", "--info", action="store", type=str, help="Additional info to store in the metadata.")
    args = parser.parse_args()

    controller = KeyboardController()
    time.sleep(0.1)
    if not controller.is_alive():
        raise RuntimeError("Could not find keyboard controller.")

    image_shape = (1024, 1024)
    image_shape_to_save = (256, 256)

    env = MCREnv(image_shape=image_shape, env_type=EnvType.AORTIC)

    env = RealtimeWrapper(env)

    if args.record_video:
        video_folder = Path("videos")
        video_folder.mkdir(exist_ok=True)
        video_name = time.strftime("%Y%m%d-%H%M%S")
        video_path = video_folder / f"{video_name}.mp4"
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            1 / (env.time_step / env.frame_skip),
            image_shape[::-1],
        )
    else:
        video_writer = None

    if args.record_trajectory:

        def store_rgb_obs(self: TrajectoryRecorder, shape: Tuple[int, int] = image_shape_to_save):
            observation = self.env.render()
            observation = cv2.resize(
                observation,
                shape,
                interpolation=cv2.INTER_AREA,
            )
            self.trajectory["rgb"].append(observation)

        metadata = {
            "frame_skip": env.frame_skip,
            "time_step": env.time_step,
            "observation_type": env.observation_type.name,
            "reward_amount_dict": env.reward_amount_dict,
            "user_info": args.info,
        }

        env = TrajectoryRecorder(
            env,
            log_dir="trajectories",
            metadata=metadata,
            store_info=True,
            save_compressed_keys=["observation", "terminal_observation", "rgb", "info"],
            after_step_callbacks=[store_rgb_obs],
            after_reset_callbacks=[store_rgb_obs],
        )

    reset_obs, reset_info = env.reset()
    if video_writer is not None:
        video_writer.write(env.render()[:, :, ::-1])

    done = False

    fps_list = deque(maxlen=100)

    while not done:
        start = time.perf_counter()

        r_z, r_x, retract = controller.read()
        print(f"r_z: {r_z}    r_x: {r_x}    retract: {retract}")

        sample_action: Dict = env.action_space.sample()

        # rotate z
        sample_action[0] = r_z
        # rotate x
        sample_action[1] = r_x
        # retraction
        sample_action[2] = retract

        obs, reward, terminated, truncated, info = env.step(sample_action)
        done = terminated or truncated
        if video_writer is not None:
            video_writer.write(env.render()[:, :, ::-1])

        if controller.x:
            cv2.imwrite("exit_image.png", env.render()[:, :, ::-1])
            break

        end = time.perf_counter()
        fps = 1 / (end - start)
        fps_list.append(fps)
        print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")

    if video_writer is not None:
        video_writer.release()
