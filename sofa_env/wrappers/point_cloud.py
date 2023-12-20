import gymnasium as gym
import numpy as np
from typing import List, Optional, Type, Callable
import open3d as o3d

from sofa_env.base import RenderMode

from sofa_env.utils.camera import vertical_to_horizontal_fov, determine_look_at, vertical_to_horizontal_fov, get_focal_length
from sofa_env.utils.math_helper import rotated_y_axis


class PointCloudFromDepthImageObservationWrapper(gym.ObservationWrapper):
    """Point cloud from depth image wrapper for SOFA simulation environments.

    Replaces the observation of the step with a point cloud.

    Args:
        transform_to_world_coordinates: Whether the point cloud coordinates should be expressed in world or camera coordinates.
        depth_array_max_distance (float): Threshold for the background in the depth array if ``point_cloud_from_depth_array=True``. Points with a depth value above this threshold will be ignored. Defaults to ``0.99 * max_depth``.
        post_processing_functions (Optional[List[Callable]]): List of functions to post process the created pointcloud.
    """

    def __init__(
        self,
        env: gym.Env,
        transform_to_world_coordinates: bool = False,
        depth_array_max_distance: Optional[float] = None,
        post_processing_functions: Optional[List[Callable]] = None,
    ) -> None:
        super().__init__(env)

        self.depth_array_max_distance = depth_array_max_distance
        self.post_processing_functions = post_processing_functions
        self.transform_to_world_coordinates = transform_to_world_coordinates

        if self.env.render_mode == RenderMode.NONE:
            raise ValueError("RenderMode of environment cannot be RenderMode.NONE, if point clouds are to be created from OpenGL depth images.")

    def reset(self, **kwargs):
        """Reads the data for the point clouds from the sofa_env after it is resetted."""

        # First reset calls _init_sim to setup the scene
        observation, reset_info = self.env.reset(**kwargs)

        if not isinstance(self.env.scene_creation_result, dict) and "camera" in self.env.scene_creation_result and isinstance(self.env.scene_creation_result["camera"], (self.env.sofa_core.Object, self.env.camera_templates.Camera)):
            raise AttributeError("No camera was found to create a raycasting scene. Please make sure createScene() returns a dictionary with key 'camera' or specify the cameras for point cloud creation in camera_configs.")

        if isinstance(self.env.scene_creation_result["camera"], self.env.camera_templates.Camera):
            self.camera_object = self.env.scene_creation_result["camera"].sofa_object
        else:
            self.camera_object = self.env.scene_creation_result["camera"]

        # Read camera parameters from SOFA camera
        self.width = int(self.camera_object.widthViewport.array())
        self.height = int(self.camera_object.heightViewport.array())

        return self.observation(observation), reset_info

    def create_point_cloud(self) -> np.ndarray:
        """Returns a point cloud calculated from the depth image of the sofa scene"""

        # Set the intrinsic camera parameters
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        fx, fy = get_focal_length(self.camera_object, self.width, self.height)
        intrinsic.set_intrinsics(self.width, self.height, fx, fy, self.width / 2, self.height / 2)

        # Get the depth image from the SOFA scene and remove the background
        depth_array = self.get_depth_from_open_gl()
        if self.depth_array_max_distance is not None:
            background_threshold = self.depth_array_max_distance
        else:
            background_threshold = 0.99 * depth_array.max()
        depth_array_no_background = np.where(depth_array > background_threshold, 0, depth_array)

        # Calculate point cloud
        depth_image = o3d.geometry.Image(depth_array_no_background)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, intrinsic)

        if self.transform_to_world_coordinates:
            # Get the model view matrix of the camera
            model_view_matrix = self.camera_object.getOpenGLModelViewMatrix()
            # Reshape from list into 4x4 matrix
            model_view_matrix = np.asarray(model_view_matrix).reshape((4, 4), order="F")

            # Invert the model_view_matrix
            transformation = np.identity(4)
            transformation[:3, :3] = model_view_matrix[:3, :3].transpose()
            transformation[:3, 3] = -model_view_matrix[:3, :3].transpose() @ model_view_matrix[:3, 3]

            # Negate part of the rotation component to flip around the z axis.
            # This is necessary because the camera looks in the negative z direction in SOFA,
            # but we invert z in get_depth_from_open_gl().
            # TODO: Find a better solution for this.
            transformation[:3, 1:3] *= -1

            pcd = pcd.transform(transformation)

        return np.asarray(pcd.points)

    def observation(self, observation) -> np.ndarray:
        """Replaces the observation of a step in a sofa_env scene with a point cloud."""

        point_cloud = self.create_point_cloud()

        # Apply optional post processing functions to point cloud
        if self.post_processing_functions is not None:
            for function in self.post_processing_functions:
                point_cloud = function(point_cloud)

        return point_cloud


class PointCloudObservationWrapper(gym.ObservationWrapper):
    """Point cloud wrapper for SOFA simulation environments.

    Replaces the observation of the step with a point cloud.

    Notes:
        The createScene function of the SofaEnv should return a dict with all objects that should
        be included in the point cloud. To correctly create the point cloud, we need points, and triangles.

        ``position_containers`` are objects where ``object.position.array()`` is valid (e.g. ``MechanicalObject``).
        ``triangle_containers`` are objects where ``object.triangles.array()`` is valid (e.g. ``TriangleSetTopologyContainer``).

        Example:
            >>> sofa_objects["pointcloud_objects"] = {}
            >>> sofa_objects["pointcloud_objects"]["position_containers"] = []
            >>> sofa_objects["pointcloud_objects"]["triangle_containers"] = []
            >>> for obj in objects:
                  sofa_objects["pointcloud_objects"]["position_containers"].append(objects[obj].visual_model_node.OglModel)
                  sofa_objects["pointcloud_objects"]["triangle_containers"].append(objects[obj].visual_model_node.OglModel)
            >>> return sofa_objects

    Args:
        append_object_id (bool): Whether to extend the point cloud with the object id of each point ``[x, y, z, id]``.
        camera_configs (Optional[List]): List of Dicts ``[{"pose": [x, y, z, a, b, c, w], "hfov": float, "width": int, "height": int}]`` including camera parameters for creating the point cloud. Will use the scene's camera if ``None``.
        post_processing_functions (Optional[List[Callable]]): List of functions to post process the created pointcloud.
        max_num_points (int): Maximum number of points in the pointcloud. Required for the shape of the observation space.
    """

    def __init__(
        self,
        env: gym.Env,
        append_object_id: bool = False,
        camera_configs: Optional[List] = None,
        post_processing_functions: Optional[List[Callable]] = None,
        max_num_points = 10000
    ) -> None:
        super().__init__(env)

        self.camera_configs = camera_configs
        self.post_processing_functions = post_processing_functions
        self.append_object_id = append_object_id

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(max_num_points, 3), dtype=np.float32)

    def reset(self, **kwargs):
        """Reads the data for the point clouds from the sofa_env after it is resetted."""

        # First reset calls _init_sim to setup the scene
        observation, reset_info = self.env.reset(**kwargs)

        self.position_containers = self.env.scene_creation_result["pointcloud_objects"]["position_containers"]
        self.triangle_containers = self.env.scene_creation_result["pointcloud_objects"]["triangle_containers"]

        if self.camera_configs is None:
            if not isinstance(self.env.scene_creation_result, dict) and "camera" in self.env.scene_creation_result and isinstance(self.env.scene_creation_result["camera"], (self.env.sofa_core.Object, self.env.camera_templates.Camera)):
                raise AttributeError("No camera was found to create a raycasting scene. Please make sure createScene() returns a dictionary with key 'camera' or specify the cameras for point cloud creation in camera_configs.")

            if isinstance(self.env.scene_creation_result["camera"], self.env.camera_templates.Camera):
                self.camera_object = self.env.scene_creation_result["camera"].sofa_object
            else:
                self.camera_object = self.env.scene_creation_result["camera"]

            # Read camera parameters from SOFA camera
            self.width = int(self.camera_object.widthViewport.array())
            self.height = int(self.camera_object.heightViewport.array())
            camera_fov = float(self.camera_object.fieldOfView.array())
            self.hfov = vertical_to_horizontal_fov(camera_fov, self.width, self.height)

        else:
            # Check if the config contains all required keys
            for camera in self.camera_configs:
                required_keys = ["pose", "width", "height", "hfov"]
                for key in required_keys:
                    if key not in camera:
                        raise KeyError(f"Could not find key {key} in configuration for camera {camera}.")

        return self.observation(observation), reset_info

    def create_open3d_scene_from_containers(self) -> Type[o3d.t.geometry.PointCloud]:
        """Returns a Open3D raycasting scene from the given positions and triangles containers.

        Note:
            For each mesh added to the scene, an ID is stored that corresponds to the passed object order.
        """

        scene = o3d.t.geometry.RaycastingScene()
        for position_container, triangle_container in zip(self.position_containers, self.triangle_containers):
            mesh = o3d.t.geometry.TriangleMesh()
            mesh.vertex.positions = o3d.core.Tensor(position_container.position.array(), o3d.core.float32)
            mesh.triangle.indices = o3d.core.Tensor(triangle_container.triangles.array(), o3d.core.int32)
            scene.add_triangles(mesh)

        return scene

    def create_point_cloud_from_scene_objects(self) -> np.ndarray:
        """Returns a point cloud including the object ID's calculated through raycasting in a created raycasting scene

        If no camera_configs are passed, the camera parameters are taken from the sofa camera object.
        Otherwise, it will be raycasted from each camera_config.
        To each point in the generated point cloud the object id is added.
        This corresponds to the position of the object in create_scene created dict sofa_object["pointcloud_object"].
        """
        ray_list = []
        if self.camera_configs is None:
            center = self.camera_object.lookAt.array()
            up_direction = rotated_y_axis(self.camera_object.orientation.array())
            eye = self.camera_object.position.array()

            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                fov_deg=self.hfov,  # field of view of camera
                center=center,  # lookAt vector
                eye=eye,  # position of the camera
                up=up_direction,  # up direction of camera
                width_px=self.width,
                height_px=self.height,
            )

            ray_list.append(rays)

        else:
            for camera in self.camera_configs:
                center = determine_look_at(camera["pose"][:3], camera["pose"][3:])
                up_direction = rotated_y_axis(camera["pose"][3:])
                rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                    fov_deg=camera["hfov"],  # field of view of camera
                    center=center,  # lookAt vector
                    eye=camera["pose"][:3],  # position of the camera
                    up=up_direction,  # up direction of camera
                    width_px=camera["width"],
                    height_px=camera["height"],
                )
                ray_list.append(rays)

        scene = self.create_open3d_scene_from_containers()

        point_list = []
        for rays in ray_list:
            ans = scene.cast_rays(rays)
            hit = ans["t_hit"].isfinite()
            points = rays[hit][:, :3] + rays[hit][:, 3:] * ans["t_hit"][hit].reshape((-1, 1))
            if self.append_object_id:
                ids = ans["geometry_ids"][hit].reshape((-1, 1)).to(dtype=o3d.core.Dtype.Float32)
                points = points.append(ids, axis=1)
            point_list.append(points.numpy())

        return np.concatenate(point_list)

    def observation(self, observation) -> np.ndarray:
        """Replaces the observation of a step in a sofa_env scene with a point cloud."""

        point_cloud = self.create_point_cloud_from_scene_objects()

        # Apply optional post processing functions to point cloud
        if self.post_processing_functions is not None:
            for function in self.post_processing_functions:
                point_cloud = function(point_cloud)

        return point_cloud
