from functools import partial
from sofa_env.scenes.precision_cutting.precision_cutting_env import PrecisionCuttingEnv, RenderMode, ObservationType, ActionType
import sofa_env.scenes.precision_cutting.sofa_objects.cloth_cut as cloth_cut
import numpy as np
from sofa_env.scenes.tissue_dissection.human_control import camera_control
from sofa_env.utils.human_input import XboxController
from sofa_env.wrappers.trajectory_recorder import TrajectoryRecorder
from sofa_env.wrappers.realtime import RealtimeWrapper
import time
from collections import deque
import cv2
import argparse
from pathlib import Path
from typing import Tuple, Callable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup human input behavior.")
    parser.add_argument("-rv", "--record_video", action="store_true", help="Record video of the trajectory.")
    parser.add_argument("-rt", "--record_trajectory", action="store_true", help="Record the full trajectory.")
    parser.add_argument("-i", "--info", action="store", type=str, help="Additional info to store in the metadata.")
    parser.add_argument("-c", "--cartesian", action="store_true", help="Use cartesian coordinates to control the scissors.")
    parser.add_argument("-s", "--sine", action="store_true", help="Use a sine wave as cutting path.")
    args = parser.parse_args()

    controller = XboxController()
    time.sleep(0.1)
    if not controller.is_alive():
        raise RuntimeError("Could not find Xbox controller.")

    image_shape = (1024, 1024)
    image_shape_to_save = (256, 256)

    def line_cutting_path_generator(rng: np.random.Generator) -> Callable:
        position = rng.uniform(low=0.3, high=0.7)
        depth = rng.uniform(low=0.3, high=0.7)
        slope = rng.uniform(low=-0.5, high=0.5)
        cutting_path = partial(cloth_cut.linear_cut, slope=slope, position=position, depth=depth)
        return cutting_path

    def sine_cutting_path_generator(rng: np.random.Generator) -> Callable:
        position = rng.uniform(low=0.3, high=0.7)
        depth = rng.uniform(low=0.3, high=0.7)
        frequency = rng.uniform(low=0.5, high=1.5) / 75
        amplitude = rng.uniform(low=10.0, high=20.0)
        cutting_path = partial(cloth_cut.sine_cut, frequency=frequency, amplitude=amplitude, position=position, depth=depth)
        return cutting_path

    cartesian_control = args.cartesian
    dt = 0.025
    control_camera = False
    env = PrecisionCuttingEnv(
        observation_type=ObservationType.STATE,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.CONTINUOUS,
        image_shape=image_shape,
        frame_skip=int(1.0 / (30.0 * dt)),
        time_step=dt,
        settle_steps=50,
        cartesian_control=cartesian_control,
        create_scene_kwargs={
            "debug_rendering": False,
            "show_closest_point_on_path": False,
        },
        cloth_cutting_path_func_generator=sine_cutting_path_generator if args.sine else line_cutting_path_generator,
        ratio_to_cut=0.85,
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
            "cartesian_control": cartesian_control,
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
        action = np.zeros_like(env.action_space.sample())
        if control_camera:
            camera_control(controller, env)
        else:
            if cartesian_control:
                lx, ly, rx, ry, lt, rt = controller.read()
                action[0] = rx
                action[1] = -ry
                action[2] = -ly
                action[3] = controller.right_bumper - controller.left_bumper
                action[4] = rt - lt
                action[5] = lx
            else:
                lx, ly, rx, ry, lt, rt = controller.read()
                action[0] = rx
                action[1] = -ry
                action[2] = lx
                action[3] = rt - lt
                action[4] = controller.a - controller.b
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

        if controller.x:
            break

        print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")

    if video_writer is not None:
        video_writer.release()
