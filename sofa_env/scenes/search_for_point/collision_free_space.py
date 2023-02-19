from sofa_env.utils.io import PointCloudWriter
from sofa_env.scenes.search_for_point.search_for_point_env import ActiveVision, SearchForPointEnv, ObservationType
from sofa_env.base import RenderMode

from typing import Union, Optional
import numpy as np
from datetime import datetime

from tqdm import tqdm
from pathlib import Path


TARGET_FOLDER = Path(__file__).resolve().parent / "collision_point_cloud"


def collision_free_boundary(env: SearchForPointEnv, samples: Union[tuple, int] = (40, 40, 100), initial_depth: Optional[float] = None):
    """calculates the collison_free space from a tool and saves it as a point cloud.

    Args:
        env (SearchForPointEnv): Search for Point Env.
        samples (Union[tuple, int], optional): Number of samples in p, t, d space.

    Returns:
        np.ndarry: Points on the collision free boundary.
    """

    env.reset()

    working_space = env.cauter.state_limits
    previous_position = env.cauter.get_collision_tip_position().copy()
    collision_free_boundary = np.array([previous_position])

    if isinstance(samples, int):
        p_sample = t_sample = d_sample = samples
    else:
        p_sample = samples[0]
        t_sample = samples[1]
        d_sample = samples[2]

    p_space = np.linspace(working_space["low"][0], working_space["high"][0], num=p_sample)
    t_space = np.linspace(working_space["low"][1], working_space["high"][1], num=t_sample)
    d_start = working_space["low"][3] if initial_depth is None else initial_depth
    d_space = np.linspace(d_start, working_space["high"][3], num=d_sample)

    progress = tqdm(range(p_space.size * t_space.size), leave=True)
    for p in p_space:
        for t in t_space:

            for d in d_space:
                # Sets the cauter to a new state
                env.cauter.set_state(np.array([p, t, 0.0, d]))
                env.step(np.zeros(env.action_space.shape))

                position = env.cauter.get_collision_tip_position()

                if env.collision():
                    # Saves the position of the cauter tip position, which has no collision
                    collision_free_boundary = np.append(collision_free_boundary, [previous_position], axis=0)

                    # Sets the position from the poi to the saved position, to visualize it
                    env.poi.set_pose(np.append(previous_position, [0.0, 0.0, 0.0, 1.0]))

                    break

                previous_position = position.copy()
            progress.update(1)

    return collision_free_boundary


def save_as_npy(points: np.ndarray, file_path: Optional[Path] = None):
    """Save points in a npy file.

    The file is then used in the ``Search_for point_env`` to load the points of interest.

    Args:
        points (np.ndarray): _description_
        file_path (Optional[Path], optional): _description_. Defaults to None.
    """
    if file_path is None:
        now = datetime.now()
        now_str = now.strftime("%m-%d-%Y_%H-%M-%S")
        file_path = TARGET_FOLDER / now_str
    np.save(file_path, points)


def save_as_ply(points, target_folder: Optional[Path] = None):
    """Save points in a .ply file.

    Can be used to visualize the collision free points.

    Args:
        points
        target_folder
    """
    if target_folder is None:
        target_folder = TARGET_FOLDER

    writer = PointCloudWriter(target_folder)
    writer.write(points)


if __name__ == "__main__":
    # If we don't visualize the progress we can speed up the calculation
    visualize = True

    if visualize:
        # Sets the scene arguments to visualize the process
        create_scene_kwargs = {
            "transparent_abdominal_wall": True,
            "use_spotlight": True,
            "place_camera_outside": True,
            "without_abdominal_wall": True,
        }
        env = SearchForPointEnv(
            render_mode=RenderMode.HUMAN,
            image_shape=(600, 600),
            frame_skip=4,
            create_scene_kwargs=create_scene_kwargs,
            check_collision=True,
            active_vision_mode=ActiveVision.CAUTER,
        )
    else:
        env = SearchForPointEnv(
            render_mode=RenderMode.NONE,
            observation_type=ObservationType.STATE,
            frame_skip=4,
            create_scene_kwargs={"without_abdominal_wall": True, },
            check_collision=True,
            active_vision_mode=ActiveVision.CAUTER,
        )

    points = collision_free_boundary(env, 40.0)
    save_as_npy(points)
    save_as_ply(points)
