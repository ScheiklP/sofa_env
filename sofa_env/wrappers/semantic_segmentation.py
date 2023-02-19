import gym
import gym.spaces
import numpy as np
from typing import List, Optional, Type
import open3d as o3d

from sofa_env.utils.camera import vertical_to_horizontal_fov, determine_look_at, vertical_to_horizontal_fov
from sofa_env.utils.math_helper import rotated_y_axis


class SemanticSegmentationWrapper(gym.ObservationWrapper):
    """Semantic segmentation wrapper for SOFA simulation environments.

    Replaces the observation of the step a semantically segmented image.

    TODO:
        - add cuda support
        - add support for multiple cameras with different resolutions -> will return a list of arrays, not a single array -> observation_space needs to be changed

    Notes:
        The createScene function of the SofaEnv should return a dict with all objects that should
        be included in the semantically segmented image. To correctly create the semantic segmentation, we need points, triangles, and a color per object.

        ``position_containers`` are objects where ``object.position.array()`` is valid (e.g. ``MechanicalObject``).
        ``triangle_containers`` are objects where ``object.triangles.array()`` is valid (e.g. ``OglModel``).

        Example:
            >>> sofa_objects["semantic_segmentation_objects"] = {}
            >>> sofa_objects["semantic_segmentation_objects"]["position_containers"] = []
            >>> sofa_objects["semantic_segmentation_objects"]["triangle_containers"] = []
            >>> for obj in objects:
                  sofa_objects["semantic_segmentation_objects"]["position_containers"].append(objects[obj].visual_model_node.OglModel)
                  sofa_objects["semantic_segmentation_objects"]["triangle_containers"].append(objects[obj].visual_model_node.OglModel)
            >>> return sofa_objects

    Args:
        env (gym.Env): The environment to wrap.
        camera_configs (Optional[List]): List of Dicts ``[{"pose": [x, y, z, a, b, c, w], "hfov": float, "width": int, "height": int}]`` including camera parameters for creating semantic segmentation images. Will use the scene's camera if ``None``.

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

            self._observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(env.create_scene_kwargs["image_shape"][0], env.create_scene_kwargs["image_shape"][1], 1), dtype=np.uint16)
        else:
            self._observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(camera_configs[0]["height"], camera_configs[0]["width"], len(camera_configs)), dtype=np.uint16)

    def reset(self, **kwargs):
        """Reads the data for the point clouds from the sofa_env after it is resetted."""

        # First reset calls _init_sim to setup the scene
        observation = self.env.reset(**kwargs)

        self.position_containers = self.env.scene_creation_result["semantic_segmentation_objects"]["position_containers"]
        self.triangle_containers = self.env.scene_creation_result["semantic_segmentation_objects"]["triangle_containers"]

        if self.camera_configs is None:
            if not isinstance(self.env.scene_creation_result, dict) and "camera" in self.env.scene_creation_result and isinstance(self.env.scene_creation_result["camera"], (self.env.sofa_core.Object, self.env.camera_templates.Camera)):
                raise AttributeError("No camera was found to create a raycasting scene. Please make sure createScene() returns a dictionary with key 'camera' or specify the cameras for in camera_configs.")

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
            target_width = self.camera_configs[0]["width"]
            target_height = self.camera_configs[0]["height"]
            for camera in self.camera_configs:
                required_keys = ["pose", "width", "height", "hfov"]
                for key in required_keys:
                    if key not in camera:
                        raise KeyError(f"Could not find key {key} in configuration for camera {camera}.")
                if not camera["width"] == target_width or not camera["height"] == target_height:
                    raise ValueError("All cameras need to have the same width and height to return an array of semantically segmented images.")

        return self.observation(observation)

    def create_open3d_scene_from_containers(self) -> Type[o3d.t.geometry.PointCloud]:
        """Returns a Open3D raycasting scene from the given position and triangle containers."""

        scene = o3d.t.geometry.RaycastingScene()
        for position_container, triangle_container in zip(self.position_containers, self.triangle_containers):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(position_container.position.array().copy())
            mesh.triangles = o3d.utility.Vector3iVector(triangle_container.triangles.array().copy())
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

            scene.add_triangles(mesh)

        return scene

    def observation(self, observation) -> np.ndarray:
        """Creates a semantic segmentation image.

        The values of the image correspond to the order of the passed objects in ``semantic_segmentation_objects`` -> object id.
        If no ``camera_configs`` are passed, the camera parameters are taken from the SOFA camera object.
        Otherwise, it will be raycasted from each configuration in ``camera_configs``.

        Note:
            Assumes that ``ans['geometry_ids']`` is the same as the order of the passed objects in ``semantic_segmentation_objects``.

        Returns:
            np.ndarray: Semantic segmentation image with values matching the object ids. Shape: If ``camera_configs=None`` ``(height, width, 1)``, else ``(height, width, len(camera_configs))``.
        """
        ray_list = []

        if self.camera_configs is None:
            center = self.camera_object.lookAt.array()
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
                up_direction = -rotated_y_axis(camera["pose"][3:])
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

        segmentation_frames = []
        for rays in ray_list:
            ans = scene.cast_rays(rays)
            seg = ans["geometry_ids"].numpy()
            segmentation_frames.append(seg[:, :, None])

        return np.concatenate(segmentation_frames, axis=-1)
