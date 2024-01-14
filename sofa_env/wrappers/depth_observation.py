import gymnasium as gym
import numpy as np
from typing import List, Optional, Type
import open3d as o3d


from sofa_env.utils.camera import vertical_to_horizontal_fov, determine_look_at, vertical_to_horizontal_fov
from sofa_env.utils.math_helper import rotated_y_axis


class DepthObservationWrapper(gym.ObservationWrapper):
    """Wrapper for SOFA simulation environments to generate depth observations with Open3D.

    Replaces the observation of the step with a depth image.

    Notes:
        The createScene function of the SofaEnv should return a dict with all objects that should
        be included in the depth image. To correctly create the depth image, we need points, and triangles.

        ``position_containers`` are objects where ``object.position.array()`` is valid (e.g. ``MechanicalObject``).
        ``triangle_containers`` are objects where ``object.triangles.array()`` is valid (e.g. ``TriangleSetTopologyContainer``).

        Example:
            >>> sofa_objects["render_objects"] = {}
            >>> sofa_objects["render_objects"]["position_containers"] = []
            >>> sofa_objects["render_objects"]["triangle_containers"] = []
            >>> for obj in objects:
                  sofa_objects["render_objects"]["position_containers"].append(objects[obj].visual_model_node.OglModel)
                  sofa_objects["render_objects"]["triangle_containers"].append(objects[obj].visual_model_node.OglModel)
            >>> return sofa_objects

    Args:
        camera_configs (Optional[List]): List of Dicts ``[{"pose": [x, y, z, a, b, c, w], "hfov": float, "width": int, "height": int}]`` including camera parameters for creating the depth image. Will use the scene's camera if ``None``.
    """

    def __init__(
        self,
        env: gym.Env,
        camera_configs: Optional[List] = None,
    ) -> None:
        super().__init__(env)

        self.camera_configs = camera_configs

        if camera_configs is None:
            if not hasattr(env, "create_scene_kwargs"):
                raise ValueError("No camera_configs specified and could not find create_scene_kwargs in env.")

            if "image_shape" not in env.create_scene_kwargs:
                raise ValueError("No camera_configs specified and could not find image_shape in create_scene_kwargs.")

            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(env.create_scene_kwargs["image_shape"][0], env.create_scene_kwargs["image_shape"][1]),
                dtype=np.uint8,
            )
        else:
            num_cameras = len(camera_configs)
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(camera_configs[0]["height"], camera_configs[0]["width"], num_cameras),
                dtype=np.uint8,
            )

    def reset(self, **kwargs):
        """Reads the data for the depth image from the sofa_env after it is reset."""

        # First reset calls _init_sim to setup the scene
        observation, reset_info = self.env.reset(**kwargs)

        self.position_containers = self.env.scene_creation_result["render_objects"]["position_containers"]
        self.triangle_containers = self.env.scene_creation_result["render_objects"]["triangle_containers"]

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

    def observation(self, observation) -> np.ndarray:
        """Returns a depth image through raycasting in a created raycasting scene

        If no camera_configs are passed, the camera parameters are taken from the sofa camera object.
        Otherwise, it will be raycasted from each camera_config.
        """
        ray_list = []
        if self.camera_configs is None:
            center = self.camera_object.lookAt.array()
            # TODO: for some reason, the up direction is flipped
            up_direction = -rotated_y_axis(self.camera_object.orientation.array())
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

        depth_images = []
        for rays in ray_list:
            ans = scene.cast_rays(rays)
            depth_images.append(ans["t_hit"].numpy())

        return np.concatenate(depth_images)
