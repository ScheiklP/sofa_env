from typing import Tuple, Dict
from copy import deepcopy
from collections import deque
from pathlib import Path
import time
import argparse

import numpy as np
import cv2

from sofa_env.scenes.rope_threading.rope_threading_env import RopeThreadingEnv, ObservationType, ActionType, RenderMode
from sofa_env.utils.human_input import XboxController
from sofa_env.wrappers.trajectory_recorder import TrajectoryRecorder
from sofa_env.wrappers.realtime import RealtimeWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup human input behavior.")
    parser.add_argument("-rv", "--record_video", action="store_true", help="Record video of the trajectory.")
    parser.add_argument("-rt", "--record_trajectory", action="store_true", help="Record the full trajectory.")
    parser.add_argument("-re", "--randomize_eyes", action="store_true", help="Randomize the eye poses.")
    parser.add_argument("-i", "--info", action="store", type=str, help="Additional info to store in the metadata.")
    args = parser.parse_args()

    controller = XboxController()
    time.sleep(0.1)
    if not controller.is_alive():
        raise RuntimeError("Could not find Xbox controller.")

    image_shape = (1024, 1024)
    image_shape_to_save = (256, 256)
    eye_config = [
        (60, 10, 0, 90),
        # (10, 10, 0, 90),
        # (10, 60, 0, -45),
        # (60, 60, 0, 90),
    ]
    create_scene_kwargs = {
        "eye_config": eye_config,
        "eye_reset_noise": {
            "low": np.array([-20.0, -20.0, 0.0, -35]),
            "high": np.array([20.0, 20.0, 0.0, 35]),
        },
        "randomize_gripper": True,
        "randomize_grasp_index": False,
        "start_grasped": True,
        "gripper_and_rope_same_collision_group": True,
        "cartesian_workspace_limits": {
            "low": (-20.0, -20.0, 0.0),
            "high": (230.0, 180.0, 100.0),
        },
    }

    env = RopeThreadingEnv(
        observation_type=ObservationType.STATE,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.CONTINUOUS,
        image_shape=image_shape,
        create_scene_kwargs=create_scene_kwargs,
        frame_skip=10,
        time_step=0.01,
        settle_steps=10,
        fraction_of_rope_to_pass=0.05,
        only_right_gripper=True,
        individual_agents=True,
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
            ptsda = self.env.right_gripper.get_articulated_state()
            self.trajectory["gripper_tpsda_state"].append(deepcopy(ptsda))
            self.trajectory["gripper_tpsd_state"].append(deepcopy(ptsda[:4]))
            self.trajectory["gripper_pose"].append(deepcopy(self.env.right_gripper.get_pose()))
            self.trajectory["grasp_center_position"].append(deepcopy(self.env.right_gripper.get_grasp_center_position()))

        def save_instrument_velocities(self: TrajectoryRecorder):
            if len(self.trajectory["gripper_tpsda_state"]) == 1:
                gripper_tpsda_velocity = np.zeros_like(self.trajectory["gripper_tpsda_state"][0])
            else:
                previous_gripper_tpsda_state = self.trajectory["gripper_tpsda_state"][-2]
                gripper_tpsda_velocity = (self.env.right_gripper.get_articulated_state() - previous_gripper_tpsda_state) / (self.env.time_step * self.env.frame_skip)
            self.trajectory["gripper_tpsda_velocity"].append(gripper_tpsda_velocity)
            self.trajectory["gripper_tpsd_velocity"].append(gripper_tpsda_velocity[:4].copy())

        def save_eyelet_states(self: TrajectoryRecorder):
            eyelet = self.env.eyes[0]
            pose = np.concatenate([eyelet.center_pose[:3], [eyelet.rotation]], axis=-1)
            self.trajectory["eyelet_center_pose"].append(pose)

        def save_rope_tracking_points(self: TrajectoryRecorder):
            # solely record the rope points up to and including the grasping point
            grasp_point_idx = self.env.right_gripper.grasp_index_pair[1]
            self.trajectory["rope_tracking_points"].append(deepcopy(self.env.rope.get_positions()[: grasp_point_idx + 1].ravel()))

        def save_time(self: TrajectoryRecorder):
            self.trajectory["time"].append(len(self.trajectory["time"]) * self.env.time_step * self.env.frame_skip)

        metadata = {
            "frame_skip": env.frame_skip,
            "time_step": env.time_step,
            "observation_type": env.observation_type.name,
            "reward_amount_dict": env.reward_amount_dict,
            "user_info": args.info,
            "seed": None,
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
                "gripper_tpsda_velocity",
                "gripper_tpsd_state",
                "gripper_tpsd_velocity",
                "gripper_pose",
                "grasp_center_position",
                "eyelet_center_pose",
                "rope_tracking_points",
                "time",
            ],
            after_step_callbacks=[
                store_rgb_obs,
                save_instrument_states,
                save_instrument_velocities,
                save_eyelet_states,
                save_rope_tracking_points,
                save_time,
            ],
            after_reset_callbacks=[
                store_rgb_obs,
                save_instrument_states,
                save_instrument_velocities,
                save_eyelet_states,
                save_rope_tracking_points,
                save_time,
            ],
        )

    seed = np.random.randint(0, 2**31 - 1)
    reset_obs, reset_info = env.reset(seed=seed)
    env.metadata["seed"] = seed
    if video_writer is not None:
        video_writer.write(env.render()[:, :, ::-1])

    done = False

    instrument = 0
    up = True

    fps_list = deque(maxlen=100)

    counter = 0
    trajectories = 0
    while True:
        start = time.perf_counter()
        lx, ly, rx, ry, lt, rt = controller.read()

        # sample_action: Dict = env.action_space.sample()
        # sample_action["left_gripper"][:] = 0.0
        # sample_action["left_gripper"][0] = -lx
        # sample_action["left_gripper"][1] = ly
        # sample_action[:] = 0.0
        # sample_action[0] = -rx
        # sample_action[1] = ry
        # if controller.y:
        #    if up:
        #        instrument = 0 if instrument == 1 else 1
        #    up = False
        # else:
        #    up = True
        # action = sample_action["right_gripper"] if instrument == 0 else sample_action["left_gripper"]
        # action[2] = controller.right_bumper - controller.left_bumper
        # action[3] = rt - lt
        # action[4] = controller.b - controller.a
        # obs, reward, terminated, truncated, info = env.step(action)

        # action ptsda state (delta)
        action = np.zeros(5)
        action[0] = -rx
        action[1] = ry
        action[2] = controller.right_bumper - controller.left_bumper
        action[3] = rt - lt
        action[4] = controller.b - controller.a
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        counter += 1

        if video_writer is not None:
            video_writer.write(env.render()[:, :, ::-1])

        if controller.x:
            print(f"Trajectories: {trajectories}")
            break

        if done:
            trajectories += 1
            print(f"Done after {counter}. Trajectories: {trajectories}")
            seed = np.random.randint(0, 2**31 - 1)
            env.reset(seed=seed)
            env.metadata["seed"] = seed
            counter = 0

        end = time.perf_counter()
        fps = 1 / (end - start)
        fps_list.append(fps)
        print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")

    if video_writer is not None:
        video_writer.release()
