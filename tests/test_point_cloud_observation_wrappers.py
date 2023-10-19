import gymnasium.spaces as spaces
import numpy as np
import open3d as o3d

from pathlib import Path
from typing import Optional, Tuple, Union

import Sofa
import Sofa.Core

from sofa_env.base import SofaEnv, RenderMode

from sofa_env.sofa_templates.camera import Camera, CAMERA_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import add_scene_header, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST
from sofa_env.sofa_templates.mappings import MappingType, MAPPING_PLUGIN_LIST

from sofa_env.wrappers.point_cloud import PointCloudFromDepthImageObservationWrapper, PointCloudObservationWrapper

PLUGIN_LIST = CAMERA_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST + VISUAL_PLUGIN_LIST + MAPPING_PLUGIN_LIST

HERE = Path(__file__).resolve().parent
MESH_DIR = HERE.parent / "assets/meshes/models"
THIS_FILE = Path(__file__)


def createScene(root_node: Sofa.Core.Node, image_shape: Tuple[int, int], z_far: float, z_near: float, camera_placement_kwargs: dict):

    add_scene_header(root_node=root_node, plugin_list=PLUGIN_LIST, dt=0.1)

    root_node.addObject("LightManager")

    root_node.addObject("DirectionalLight", direction=(0, -1, 1), color=(0.8, 0.8, 0.8, 1.0))
    root_node.addObject("DirectionalLight", direction=(-1, 0, 1), color=(0.8, 0.8, 0.8, 1.0))

    camera = Camera(
        root_node=root_node,
        placement_kwargs=camera_placement_kwargs,
        z_near=z_near,
        z_far=z_far,
        width_viewport=image_shape[1],
        height_viewport=image_shape[0],
    )

    sphere_radius = 10.0
    center_sphere_node = root_node.addChild("center_sphere")
    center_sphere_node.addObject(
        "MechanicalObject",
        template="Rigid3d",
        position=[0.0, 0.0, 30.0] + [0.0, 0.0, 0.0, 1.0],
    )
    center_visual_model_node = add_visual_model(
        attached_to=center_sphere_node,
        surface_mesh_file_path=MESH_DIR / "unit_sphere.stl",
        scale=sphere_radius,
        color=(1.0, 0.0, 0.0),
        mapping_type=MappingType.RIGID,
    )

    offcenter_sphere_node = root_node.addChild("offcenter_sphere")
    offcenter_sphere_node.addObject(
        "MechanicalObject",
        template="Rigid3d",
        position=[30.0, 30.0, 10.0] + [0.0, 0.0, 0.0, 1.0],
    )
    offcenter_visual_model_node = add_visual_model(
        attached_to=offcenter_sphere_node,
        surface_mesh_file_path=MESH_DIR / "unit_sphere.stl",
        scale=sphere_radius,
        color=(0.0, 1.0, 0.0),
        mapping_type=MappingType.RIGID,
    )

    sofa_objects = {"root_node": root_node, "camera": camera}

    sofa_objects["pointcloud_objects"] = {}
    sofa_objects["pointcloud_objects"]["position_containers"] = []
    sofa_objects["pointcloud_objects"]["triangle_containers"] = []
    sofa_objects["pointcloud_objects"]["position_containers"].append(center_visual_model_node.OglModel)
    sofa_objects["pointcloud_objects"]["triangle_containers"].append(center_visual_model_node.OglModel)
    sofa_objects["pointcloud_objects"]["position_containers"].append(offcenter_visual_model_node.OglModel)
    sofa_objects["pointcloud_objects"]["triangle_containers"].append(offcenter_visual_model_node.OglModel)

    return sofa_objects


class PointCloudEnv(SofaEnv):
    def __init__(
        self,
        camera_placement_kwargs: dict = {
            "position": [-100.0, -100.0, 50.0],
            "lookAt": [0.0, 0.0, 0.0],
        },
        z_near: float = 0.1,
        z_far: float = 300.0,
        scene_path: Union[str, Path] = THIS_FILE,
        image_shape: Tuple[int, int] = (400, 400),
        time_step: float = 0.01,
        frame_skip: int = 1,
        render_mode: RenderMode = RenderMode.HEADLESS,
        create_scene_kwargs: Optional[dict] = None,
    ) -> None:

        # Pass image shape to the scene creation function
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape
        create_scene_kwargs["camera_placement_kwargs"] = camera_placement_kwargs
        create_scene_kwargs["z_far"] = z_far
        create_scene_kwargs["z_near"] = z_near

        super().__init__(
            scene_path=scene_path,
            time_step=time_step,
            frame_skip=frame_skip,
            render_mode=render_mode,
            create_scene_kwargs=create_scene_kwargs,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (3,), dtype=np.uint8)

    def _do_action(self, action: np.ndarray) -> None:
        pass

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        rgb_observation = super().step(action)
        info = {}
        done = False
        reward = 0.0
        return rgb_observation, reward, done, info

    def reset(self) -> np.ndarray:
        return super().reset()


class TestPointCloudObservationWrappers:
    def test_generate_point_clouds(self) -> None:
        reference_point_cloud = np.asarray(o3d.io.read_point_cloud("reference_point_cloud_a.ply").points)
        env = PointCloudEnv(render_mode=RenderMode.NONE)
        env = PointCloudObservationWrapper(env)
        reset_obs = env.reset()
        env.close()

        assert reset_obs.shape == (5131, 3)
        assert np.allclose(reset_obs, reference_point_cloud)

    def test_generate_point_clouds_from_depth_image(self) -> None:
        reference_point_cloud = np.asarray(o3d.io.read_point_cloud("reference_point_cloud_b.ply").points)
        env = PointCloudEnv(render_mode=RenderMode.HEADLESS)
        env = PointCloudFromDepthImageObservationWrapper(env)
        reset_obs = env.reset()
        env.close()

        assert reset_obs.shape == (5131, 3)
        assert np.allclose(reset_obs, reference_point_cloud)

    def test_transformation_to_world(self) -> None:

        env = PointCloudEnv(render_mode=RenderMode.NONE)
        env = PointCloudObservationWrapper(env)
        reset_obs = env.reset()
        env.close()

        env = PointCloudEnv(render_mode=RenderMode.HEADLESS)
        env = PointCloudFromDepthImageObservationWrapper(env, transform_to_world_coordinates=True)
        reset_obs_from_depth_image = env.reset()
        env.close()

        reset_point_cloud = o3d.geometry.PointCloud()
        reset_point_cloud.points = o3d.utility.Vector3dVector(reset_obs)

        reset_point_cloud_from_depth_image = o3d.geometry.PointCloud()
        reset_point_cloud_from_depth_image.points = o3d.utility.Vector3dVector(reset_obs_from_depth_image)

        evaluation = o3d.pipelines.registration.evaluate_registration(reset_point_cloud, reset_point_cloud_from_depth_image, 0.5, np.identity(4))
        assert len(evaluation.correspondence_set) == 5131
