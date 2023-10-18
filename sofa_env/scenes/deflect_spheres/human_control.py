from sofa_env.utils.human_input import XboxController
from sofa_env.wrappers.trajectory_recorder import TrajectoryRecorder
from sofa_env.wrappers.realtime import RealtimeWrapper
from sofa_env.scenes.deflect_spheres.deflect_spheres_env import DeflectSpheresEnv, Mode, ObservationType, ActionType, RenderMode

import cv2
import numpy as np
import time
import argparse

from copy import deepcopy
from collections import deque
from pathlib import Path
from typing import Dict, Tuple


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup human input behavior.")
    parser.add_argument("-rv", "--record_video", action="store_true", help="Record video of the trajectory.")
    parser.add_argument("-rt", "--record_trajectory", action="store_true", help="Record the full trajectory.")
    parser.add_argument("-i", "--info", action="store", type=str, help="Additional info to store in the metadata.")
    args = parser.parse_args()

    controller = XboxController()
    time.sleep(0.1)
    if not controller.is_alive():
        raise RuntimeError("Could not find Xbox controller.")

    image_shape = (1024, 1024)
    image_shape_to_save = (256, 256)

    env = DeflectSpheresEnv(
        observation_type=ObservationType.STATE,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.CONTINUOUS,
        image_shape=image_shape,
        frame_skip=1,
        time_step=1 / 30,
        settle_steps=10,
        single_agent=False,
        individual_agents=True,
        mode=Mode.WITHOUT_REPLACEMENT,
    )

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

        def save_instrument_states(self: TrajectoryRecorder):
            self.trajectory["right_ptsd_state"].append(deepcopy(self.env.right_cauter.get_state()))
            self.trajectory["right_pose"].append(deepcopy(self.env.right_cauter.get_pose()))
            self.trajectory["left_ptsd_state"].append(deepcopy(self.env.left_cauter.get_state()))
            self.trajectory["left_pose"].append(deepcopy(self.env.left_cauter.get_pose()))

        def save_instrument_velocities(self: TrajectoryRecorder):
            if len(self.trajectory["right_ptsd_state"]) == 1:
                right_ptsd_velocity = np.zeros_like(self.trajectory["right_ptsd_state"][0])
                left_ptsd_velocity = np.zeros_like(self.trajectory["right_ptsd_state"])[0]
            else:
                previous_right_ptsd_state = self.trajectory["right_ptsd_state"][-2]
                previous_left_ptsd_state = self.trajectory["left_ptsd_state"][-2]
                right_ptsd_velocity = (self.env.right_cauter.get_state() - previous_right_ptsd_state) / (self.env.time_step * self.env.frame_skip)
                left_ptsd_velocity = (self.env.left_cauter.get_state() - previous_left_ptsd_state) / (self.env.time_step * self.env.frame_skip)

            self.trajectory["right_ptsd_velocity"].append(right_ptsd_velocity)
            self.trajectory["left_ptsd_velocity"].append(left_ptsd_velocity)

        def save_post_states(self: TrajectoryRecorder):
            number_of_spheres = self.env.num_objects
            self.trajectory["sphere_positions"].append(deepcopy(self.trajectory["observation"][-1][: number_of_spheres * 3]))
            self.trajectory["active_sphere_position"].append(deepcopy(self.trajectory["observation"][-1][number_of_spheres * 3 : (number_of_spheres + 1) * 3]))
            self.trajectory["active_agent"].append(deepcopy(self.trajectory["observation"][-1][-12]))

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
            save_compressed_keys=[
                "observation",
                "terminal_observation",
                "rgb",
                "info",
                "right_ptsd_state",
                "right_ptsd_velocity",
                "right_pose",
                "left_ptsd_state",
                "left_ptsd_velocity",
                "left_pose",
                "sphere_positions",
                "active_sphere_position",
            ],
            after_step_callbacks=[
                store_rgb_obs,
                save_instrument_states,
                save_instrument_velocities,
                save_post_states,
            ],
            after_reset_callbacks=[
                store_rgb_obs,
                save_instrument_states,
                save_instrument_velocities,
                save_post_states,
            ],
        )

    reset_obs, reset_info = env.reset()
    if video_writer is not None:
        video_writer.write(env.render()[:, :, ::-1])

    done = False

    instrument = 0
    up = True

    fps_list = deque(maxlen=100)

    while not done:
        start = time.perf_counter()

        lx, ly, rx, ry, lt, rt = controller.read()
        sample_action: Dict = env.action_space.sample()

        sample_action["right_cauter"][:] = 0.0
        sample_action["right_cauter"][0] = rx
        sample_action["right_cauter"][1] = -ry

        sample_action["left_cauter"][:] = 0.0
        sample_action["left_cauter"][0] = lx
        sample_action["left_cauter"][1] = -ly

        if controller.y:
            if up:
                instrument = 0 if instrument == 1 else 1
            up = False
        else:
            up = True

        action = sample_action["right_cauter"] if instrument == 0 else sample_action["left_cauter"]
        action[2] = controller.b - controller.x
        action[3] = rt - lt

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
