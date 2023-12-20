from copy import deepcopy
from sofa_env.utils.human_input import XboxController
from sofa_env.wrappers.realtime import RealtimeWrapper
from sofa_env.wrappers.trajectory_recorder import TrajectoryRecorder
from sofa_env.scenes.grasp_lift_touch.grasp_lift_touch_env import GraspLiftTouchEnv, ObservationType, ActionType, RenderMode, Phase
import cv2
from typing import Tuple
import numpy as np
import time
from collections import deque
from pathlib import Path
import argparse

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

    env = GraspLiftTouchEnv(
        render_mode=RenderMode.HUMAN,
        observation_type=ObservationType.STATE,
        action_type=ActionType.CONTINUOUS,
        start_in_phase=Phase.GRASP,
        end_in_phase=Phase.DONE,
        image_shape=image_shape,
        individual_agents=False,
        individual_rewards=False,
        time_step=1 / 30,
        frame_skip=1,
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
            self.trajectory["gripper_tpsda_state"].append(deepcopy(self.env.gripper.articulated_state))
            self.trajectory["cauter_tpsd_state"].append(deepcopy(self.env.cauter.ptsd_state))
            self.trajectory["gripper_pose"].append(deepcopy(self.env.gripper.get_pose()))
            self.trajectory["cauter_pose"].append(deepcopy(self.env.cauter.get_pose()))

        def save_instrument_velocities(self: TrajectoryRecorder):
            if len(self.trajectory["gripper_tpsda_state"]) == 1:
                gripper_tpsda_velocity = np.zeros_like(self.trajectory["gripper_tpsda_state"][0])
                cauter_tpsd_velocity = np.zeros_like(self.trajectory["cauter_tpsd_state"][0])
            else:
                previous_gripper_tpsda_state = self.trajectory["gripper_tpsda_state"][-2]
                gripper_tpsda_velocity = (self.env.gripper.articulated_state - previous_gripper_tpsda_state) / (self.env.time_step * self.env.frame_skip)
                previous_cauter_tpsd_state = self.trajectory["cauter_tpsd_state"][-2]
                cauter_tpsd_velocity = (self.env.cauter.ptsd_state - previous_cauter_tpsd_state) / (self.env.time_step * self.env.frame_skip)

            self.trajectory["gripper_tpsda_velocity"].append(gripper_tpsda_velocity)
            self.trajectory["cauter_tpsd_velocity"].append(cauter_tpsd_velocity)

        def save_gallbladder_state(self: TrajectoryRecorder):
            self.trajectory["gallbladder_state"].append(deepcopy(self.env.gallbladder_surface_mechanical_object.position.array()[self.env.gallbladder_tracking_indices].ravel()))

        def save_phase(self: TrajectoryRecorder):
            self.trajectory["phase"].append([self.env.active_phase.value])  # stored as a single element list to have a shape of [T, 1] instead of [T]

        def save_time(self: TrajectoryRecorder):
            self.trajectory["time"].append(len(self.trajectory["time"]) * self.env.time_step * self.env.frame_skip)

        serializable_reward_amount_dict = {}
        for key, val in env.reward_amount_dict.items():
            serializable_reward_amount_dict[key.name] = val

        metadata = {
            "frame_skip": env.frame_skip,
            "time_step": env.time_step,
            "observation_type": env.observation_type.name,
            "reward_amount_dict": serializable_reward_amount_dict,
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
                "gripper_tpsda_state",
                "cauter_tpsd_state",
                "gripper_tpsda_velocity",
                "cauter_tpsd_velocity",
                "gripper_pose",
                "cauter_pose",
                "gallbladder_state",
                "phase",
                "time",
            ],
            after_step_callbacks=[
                store_rgb_obs,
                save_instrument_states,
                save_instrument_velocities,
                save_gallbladder_state,
                save_phase,
                save_time,
            ],
            after_reset_callbacks=[
                store_rgb_obs,
                save_instrument_states,
                save_instrument_velocities,
                save_gallbladder_state,
                save_phase,
                save_time,
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
        action = np.zeros_like(env.action_space.sample())
        action[0] = ly
        action[1] = lx
        action[5] = rx
        action[6] = -ry
        action[2 + instrument * 5] = controller.right_bumper - controller.left_bumper
        action[3 + instrument * 5] = rt - lt
        action[4 + instrument * 5] = controller.b - controller.a

        if controller.y:
            if up:
                instrument = 0 if instrument == 1 else 1
            up = False
        else:
            up = True

        obs, reward, terminated, truncated, info = env.step(action)
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
