from sofa_env.scenes.search_for_point.search_for_point_env import ActiveVision, SearchForPointEnv, ObservationType, ActionType, RenderMode
from sofa_env.utils.human_input import XboxController
from sofa_env.wrappers.trajectory_recorder import TrajectoryRecorder
from sofa_env.wrappers.realtime import RealtimeWrapper
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
    parser.add_argument("-av", "--active_vision", action="store_true", help="Use active vision.")
    args = parser.parse_args()

    controller = XboxController()
    time.sleep(0.1)
    if not controller.is_alive():
        raise RuntimeError("Could not find Xbox controller.")

    image_shape = (1024, 1024)
    image_shape_to_save = (256, 256)

    env = SearchForPointEnv(
        observation_type=ObservationType.RGB,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.CONTINUOUS,
        image_shape=image_shape,
        individual_agents=False,
        frame_skip=1,
        time_step=1 / 30,
        active_vision_mode=ActiveVision.CAUTER if args.active_vision else ActiveVision.DEACTIVATED,
        cauter_activation=True,
        check_collision=True if args.active_vision else False,
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
            "active_vision": args.active_vision,
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

    instrument = 0
    up = True

    fps_list = deque(maxlen=100)

    while not done:
        start = time.perf_counter()
        lx, ly, rx, ry, lt, rt = controller.read()

        action = np.zeros_like(env.action_space.sample())
        action[0] = rx
        action[1] = -ry

        if args.active_vision:
            action[4] = lx
            action[5] = -ly
            action[8] = controller.b - controller.a
        else:
            instrument = 0

        action[2 + instrument * 4] = controller.right_bumper - controller.left_bumper
        action[3 + instrument * 4] = rt - lt

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
