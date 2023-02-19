import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from typing import Callable, Optional, Dict
from collections import deque

from sofa_env.scenes.tissue_retraction.tissue_retraction_env import TissueRetractionEnv, ObservationType
from sofa_env.base import RenderMode
from sofa_env.sofa_templates.visual import set_color

HERE = Path(__file__).resolve().parent


def parametrize_camera_shake_callback(delta_position_limits: Dict, max_delta_angle: float) -> Callable:

    assert "low" in delta_position_limits
    assert "high" in delta_position_limits

    rotation_axis_low = np.array([-1.0] * 3)
    rotation_axis_high = np.array([1.0] * 3)

    def camera_shake_callback(env: TissueRetractionEnv) -> None:

        camera = env.scene_creation_result["camera"]

        position_delta = env.rng.uniform(delta_position_limits["low"], delta_position_limits["high"])
        rotation_axis = env.rng.uniform(rotation_axis_low, rotation_axis_high)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle_delta = env.rng.uniform(0, max_delta_angle)

        rotation_delta = R.from_rotvec(rotation_axis * angle_delta, degrees=True)

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.__mul__.html
        # If a and b are two rotations, then the composition of ‘b followed by a’ is equivalent to a * b. In terms of rotation matrices, the composition can be expressed as a.as_matrix().dot(b.as_matrix()).
        new_orientation = rotation_delta * R.from_quat(camera.initial_pose[3:])
        new_position = camera.initial_pose[:3] + position_delta

        new_pose = np.append(new_position, new_orientation.as_quat(), axis=0)

        camera.set_pose(new_pose)

    return camera_shake_callback


def parametrize_random_light_color_callback(min: Optional[np.ndarray] = None, max: Optional[np.ndarray] = None) -> Callable:
    if min is None:
        min = np.array([0.5, 0.5, 0.5, 1.0])

    if max is None:
        max = np.array([1.0, 1.0, 1.0, 1.0])

    def random_light_color_callback(env: TissueRetractionEnv) -> None:
        with env._sofa_root_node.PositionalLight.color.writeable() as color:
            color[:] = env.rng.uniform(min, max)

    return random_light_color_callback


def parametrize_random_light_position_callback(min: Optional[np.ndarray] = None, max: Optional[np.ndarray] = None) -> Callable:
    if min is None:
        min = np.array([-0.075, 0.04, -0.075])

    if max is None:
        max = np.array([0.075, 0.09, 0.075])

    def random_light_position_callback(env: TissueRetractionEnv) -> None:
        with env._sofa_root_node.PositionalLight.position.writeable() as position:
            position[:] = env.rng.uniform(min, max)

    return random_light_position_callback


def parametrize_random_object_color_callback(min: Optional[np.ndarray] = None, max: Optional[np.ndarray] = None) -> Callable:
    if min is None:
        min = np.array([0.3, 0.3, 0.3])

    if max is None:
        max = np.array([1.0, 1.0, 1.0])

    def random_object_color_callback(env: TissueRetractionEnv) -> None:
        color = env.rng.uniform(min, max)
        object = env._sofa_root_node.scene.grid_board.OglModel
        set_color(object, color)

        color = env.rng.uniform(min, max)
        object = env._sofa_root_node.scene.end_effector.end_effector_gripper.end_effector_gripper_motion_target.visual_open.OglModel
        set_color(object, color)

        color = env.rng.uniform(min, max)
        object = env._sofa_root_node.scene.end_effector.end_effector_gripper.end_effector_gripper_motion_target.visual_closed.OglModel
        set_color(object, color)

        color = env.rng.uniform(min, max)
        object = env._sofa_root_node.scene.end_effector.end_effector_main_link.end_effector_main_link_motion_target.visual.OglModel
        set_color(object, color)

        color = env.rng.uniform(min, max)
        object = env._sofa_root_node.scene.visual_target.OglModel
        set_color(object, color)

    return random_object_color_callback


def setup_random_object_texture_callback() -> Callable:
    possible_textures = [path for path in (HERE / Path("meshes/domain_randomization_textures")).iterdir()]
    num_textures = len(possible_textures)

    def random_object_texture_callback(env: TissueRetractionEnv) -> None:

        texture = possible_textures[env.rng.choice(num_textures)]
        object = env._sofa_root_node.scene.floor.OglModel
        object.findData("texturename").value = str(texture)

        texture = possible_textures[env.rng.choice(num_textures)]
        object = env._sofa_root_node.scene.background.OglModel
        object.findData("texturename").value = str(texture)

        texture = possible_textures[env.rng.choice(num_textures)]
        object = env._sofa_root_node.scene.tissue.visual.OglModel
        object.findData("texturename").value = str(texture)

    return random_object_texture_callback


if __name__ == "__main__":
    import time

    camera_shake_callback = parametrize_camera_shake_callback(
        {
            "low": np.array([-0.003, -0.003, -0.003]),
            "high": np.array([0.003, 0.003, 0.003]),
        },
        10.0,
    )

    random_object_texture_callback = setup_random_object_texture_callback()
    random_object_color_callback = parametrize_random_object_color_callback()

    random_light_color_callback = parametrize_random_light_color_callback()
    random_light_position_callback = parametrize_random_light_position_callback()

    callbacks = [
        camera_shake_callback,
        random_object_color_callback,
        random_object_texture_callback,
        random_light_position_callback,
        random_light_color_callback,
    ]

    env = TissueRetractionEnv(
        observation_type=ObservationType.RGB,
        render_mode=RenderMode.HUMAN,
        image_shape=(600, 600),
        frame_skip=3,
        time_step=0.1,
        maximum_grasp_height=0.0088819,
        create_scene_kwargs={
            "show_floor": True,
            "texture_objects": True,
            "camera_field_of_view_vertical": 42,
        },
        on_reset_callbacks=callbacks,
    )

    env.reset()

    fps_list = deque(maxlen=100)

    counter = 0

    while True:

        start = time.time()
        obs, reward, done, info = env.step(env.action_space.sample())
        end = time.time()
        fps = 1 / (end - start)
        fps_list.append(fps)

        counter += 1

        if counter % 30 == 0:
            env.reset()
            counter = 0

        print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")
