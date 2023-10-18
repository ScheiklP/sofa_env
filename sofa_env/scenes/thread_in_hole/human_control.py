import numpy as np
from sofa_env.utils.human_input import XboxController
from sofa_env.scenes.thread_in_hole.thread_in_hole_env import ThreadInHoleEnv, ObservationType, ActionType, RenderMode
from sofa_env.wrappers.trajectory_recorder import TrajectoryRecorder
from sofa_env.wrappers.realtime import RealtimeWrapper
import time
from collections import deque
from pathlib import Path
import argparse
import cv2
from typing import Tuple

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
    mode = "inverted"

    mechanical_configs = {
        "normal": {
            "thread_config": {"length": 50.0, "radius": 2.0, "total_mass": 1.0, "young_modulus": 4000.0, "poisson_ratio": 0.3, "beam_radius": 5.0, "mechanical_damping": 0.2},
            "hole_config": {"inner_radius": 5.0, "outer_radius": 25.0, "height": 30.0, "young_modulus": 5000.0, "poisson_ratio": 0.3, "total_mass": 10.0},
        },
        "flexible": {
            "thread_config": {"length": 80.0, "radius": 2.0, "total_mass": 1.0, "young_modulus": 1000.0, "poisson_ratio": 0.3, "beam_radius": 3.0, "mechanical_damping": 0.2},
            "hole_config": {"inner_radius": 6.0, "outer_radius": 25.0, "height": 30.0, "young_modulus": 5000.0, "poisson_ratio": 0.3, "total_mass": 10.0},
        },
        "inverted": {
            "thread_config": {"length": 50.0, "radius": 2.0, "total_mass": 10.0, "young_modulus": 1e5, "poisson_ratio": 0.3, "beam_radius": 5.0, "mechanical_damping": 0.2},
            "hole_config": {"inner_radius": 6.0, "outer_radius": 15.0, "height": 60.0, "young_modulus": 5e2, "poisson_ratio": 0.3, "total_mass": 1.0},
        },
    }

    env = ThreadInHoleEnv(
        observation_type=ObservationType.RGB,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.CONTINUOUS,
        image_shape=image_shape,
        frame_skip=1,
        time_step=1 / 30,
        settle_steps=50,
        create_scene_kwargs={
            "randomize_gripper": True,
            "gripper_config": {
                "cartesian_workspace": {
                    "low": np.array([-100.0] * 2 + [0.0]),
                    "high": np.array([100.0] * 2 + [200.0]),
                },
                "state_reset_noise": np.array([15.0, 15.0, 0.0, 20.0]),
                "rcm_reset_noise": np.array([10.0, 10.0, 10.0, 5.0, 5.0, 5.0]),
                "gripper_ptsd_state": np.array([60.0, 0.0, 180.0, 90.0]),
                "gripper_rcm_pose": np.array([100.0, 0.0, 150.0, 0.0, 180.0, 0.0]),
            },
            "camera_config": {
                "placement_kwargs": {
                    "position": [0.0, -135.0, 100.0],
                    "lookAt": [0.0, 0.0, 45.0],
                },
                "vertical_field_of_view": 62.0,
            },
            **mechanical_configs[mode],
        },
        num_thread_tracking_points=4,
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
        lx, ly, rx, ry, lt, rt = controller.read()
        action = np.zeros_like(env.action_space.sample())
        action[0] = -ry
        action[1] = -rx
        action[2] = lx
        action[3] = rt - lt
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
