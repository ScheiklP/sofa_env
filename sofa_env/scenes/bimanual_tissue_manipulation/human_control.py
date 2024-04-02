import argparse
import cv2
import numpy as np
import time
import pprint

from typing import Tuple
from copy import deepcopy
from collections import deque
from pathlib import Path

from sofa_env.utils.human_input import XboxController
from sofa_env.wrappers.realtime import RealtimeWrapper
from sofa_env.wrappers.trajectory_recorder import TrajectoryRecorder

from sofa_env.scenes.bimanual_tissue_manipulation.bimanual_tissue_manipulation_env import BimanualTissueManipulationEnv, ObservationType, ActionType, RenderMode


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

    env = BimanualTissueManipulationEnv(
        observation_type=ObservationType.RGB,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.CONTINUOUS,
        image_shape=image_shape,
        frame_skip=10,
        time_step=0.01,
        with_collision=False,
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
            self.trajectory["right_pd_state"].append(deepcopy(self.env.right_gripper.get_state()[[0, 3]]))
            self.trajectory["right_pose"].append(deepcopy(self.env.right_gripper.get_pose()))
            self.trajectory["left_pd_state"].append(deepcopy(self.env.left_gripper.get_state()[[0, 3]]))
            self.trajectory["left_pose"].append(deepcopy(self.env.left_gripper.get_pose()))

        def save_instrument_velocities(self: TrajectoryRecorder):
            if len(self.trajectory["right_pd_state"]) == 1:
                right_pd_velocity = np.zeros_like(self.trajectory["right_pd_state"][0])
                left_pd_velocity = np.zeros_like(self.trajectory["right_pd_state"])[0]

            else:
                previous_right_pd_state = self.trajectory["right_pd_state"][-2]
                right_pd_velocity = (self.env.right_gripper.get_state()[[0, 3]] - previous_right_pd_state) / (self.env.time_step * self.env.frame_skip)
                previous_left_pd_state = self.trajectory["left_pd_state"][-2]
                left_pd_velocity = (self.env.left_gripper.get_state()[[0, 3]] - previous_left_pd_state) / (self.env.time_step * self.env.frame_skip)

            self.trajectory["right_pd_velocity"].append(right_pd_velocity)
            self.trajectory["left_pd_velocity"].append(left_pd_velocity)

        def save_marker_states(self: TrajectoryRecorder):
            self.trajectory["marker_positions"].append(deepcopy(self.env.tissue.get_marker_positions()[:, :2]).ravel())
            self.trajectory["marker_positions_in_image"].append(deepcopy(self.env.get_marker_positions_in_image()).ravel())

        def save_target_states(self: TrajectoryRecorder):
            self.trajectory["target_positions"].append(deepcopy(self.env.target_positions[:, :2]).ravel())
            self.trajectory["target_positions_in_image"].append(deepcopy(self.env.get_target_positions_in_image()).ravel())

        def save_time(self: TrajectoryRecorder):
            self.trajectory["time"].append(len(self.trajectory["time"]) * self.env.time_step * self.env.frame_skip)

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
                "right_pd_state",
                "right_pose",
                "left_pd_state",
                "left_pose",
                "right_pd_velocity",
                "left_pd_velocity",
                "marker_positions",
                "marker_positions_in_image",
                "target_positions",
                "target_positions_in_image",
                "time",
            ],
            after_step_callbacks=[
                store_rgb_obs,
                save_instrument_states,
                save_instrument_velocities,
                save_marker_states,
                save_target_states,
                save_time,
            ],
            after_reset_callbacks=[
                store_rgb_obs,
                save_instrument_states,
                save_instrument_velocities,
                save_marker_states,
                save_target_states,
                save_time,
            ],
        )

    reset_obs, reset_info = env.reset()
    if video_writer is not None:
        video_writer.write(env.render()[:, :, ::-1])
    done = False

    pp = pprint.PrettyPrinter()

    fps_list = deque(maxlen=100)

    counter = 0
    trajectories = 0
    while True:
        start = time.perf_counter()
        lx, ly, rx, ry, lt, rt = controller.read()

        action = np.zeros_like(env.action_space.sample())
        action[0] = ly
        action[1] = -lx

        action[2] = -ry
        action[3] = rx

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        counter += 1
        if video_writer is not None:
            video_writer.write(env.render()[:, :, ::-1])

        if controller.start:
            env.reset()
        if controller.x:
            break
        if done:
            trajectories += 1
            print(f"Done after {counter}. Trajectories: {trajectories}")
            env.reset()
            counter = 0

        end = time.perf_counter()
        fps = 1 / (end - start)
        fps_list.append(fps)
        print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")

    if video_writer is not None:
        video_writer.release()
