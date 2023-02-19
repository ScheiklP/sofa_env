import numpy as np

from stl import mesh
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union, List, Tuple

import Sofa
import Sofa.Core

from sofa_env.sofa_templates.rigid import RigidObject, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, set_color, VISUAL_PLUGIN_LIST

POI_PLUGIN_LIST = RIGID_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST + VISUAL_PLUGIN_LIST


class PointOfInterest(RigidObject):
    """Point of interest model in the search for point scene

    Args:
        parent_node (Sofa.Core.Node): Parent node of the SOFA object.
        name (str): Name of the object.
        randomized_pose (bool): Whether to randomly sample a new poisition on reset.
        initial_position (np.ndarray): Initial position of the point of interest. Only used if ``randomized_pose`` is False.
        visual_mesh_path (Optional[Union[str, Path]]): Path to the visual surface mesh file.
        poi_mesh_path (Optional[Union[str, Path]]): Path to the point of interest mesh file from which the target positions are sampled.
        radius (float): Radius of the point of interest.
        add_visual_model_func (Callable): Function that defines how the visual surface from ``visual_mesh_path`` is added to the object.
        animation_loop_type (AnimationLoopType): The scenes animation loop in order to correctly add constraint correction objects.
        show_object (bool): Whether to render the pose frame of the object.
        show_object_scale (float): Render size of the node for ``show_object=True``.
        non_default_target_positions (Optional[List]): List of possible target positions. If None, the target positions are calculated from the ``poi_mesh_path``.
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        randomized_pose: bool = True,
        initial_position: np.ndarray = np.zeros(3),
        visual_mesh_path: Optional[Union[str, Path]] = None,
        poi_mesh_path: Optional[Union[str, Path]] = None,
        radius: float = 3,
        add_visual_model_func: Callable = partial(add_visual_model, color=(0, 1, 0)),
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        show_object: bool = False,
        show_object_scale: float = 10.0,
        non_default_target_positions: Optional[List] = None,
    ) -> None:
        self.randomized_pose = randomized_pose

        if non_default_target_positions is None:
            if poi_mesh_path is not None:
                if (isinstance(poi_mesh_path, Path) and poi_mesh_path.suffix == ".stl") or (isinstance(poi_mesh_path, str) and poi_mesh_path.endswith(".stl")):
                    self.possible_target_poses = self.stl_to_pose_list(poi_mesh_path)
                elif (isinstance(poi_mesh_path, Path) and poi_mesh_path.suffix == ".npy") or (isinstance(poi_mesh_path, str) and poi_mesh_path.endswith(".npy")):
                    self.possible_target_poses = self.npy_to_pose_list(poi_mesh_path)
                else:
                    raise ValueError(f"Received {poi_mesh_path=}. This is not a valid file. Please pass a valid path to the file.")
        else:
            if not all([len(position) == 3 for position in non_default_target_positions]):
                raise ValueError(f"Please pass target positions as a list of XYZ values. Received {non_default_target_positions}.")
            self.possible_target_poses = [position.extend([0.0, 0.0, 0.0, 1.0]) for position in non_default_target_positions]

        if self.randomized_pose:
            self.pose_index = np.random.randint(0, len(self.possible_target_poses))  # self.rng does not exist yet -> call "old" numpy random
            self.pose = np.array(self.possible_target_poses[self.pose_index])
        else:
            if not isinstance(initial_position, np.ndarray) and not initial_position.shape == (3,):
                raise ValueError(f"Please pass the position of the point of interest as a numpy array with XYZ coordinates. Received {initial_position}.")
            initial_pose = np.zeros(7)
            initial_pose[-1] = 1.0
            initial_pose[:3] = initial_position
            self.pose = initial_pose
            self.pose_index = None

        self.radius = radius
        super().__init__(
            parent_node=parent_node,
            name=name,
            pose=self.pose,
            fixed_position=False,
            fixed_orientation=False,
            visual_mesh_path=visual_mesh_path,
            scale=radius,
            add_visual_model_func=add_visual_model_func,
            animation_loop_type=animation_loop_type,
            show_object=show_object,
            show_object_scale=show_object_scale,
        )

    def stl_to_pose_list(self, mesh_file: Union[str, Path]):
        """Calculates a list of the centroids of a given mesh file.
        Args:
            mesh_file (Union[str, Path]): Path to the mesh file.

        Returns:
            pose_list (list[float]): list with position and orientation of the poses.
        """
        new_mesh = mesh.Mesh.from_file(Path(mesh_file))
        pose_list = new_mesh.centroids.tolist()
        for position in pose_list:
            position.extend([0.0, 0.0, 0.0, 1.0])

        return pose_list

    def npy_to_pose_list(self, file_path: Union[str, Path]):
        """Returns the positions from a npy file as a list of poses.

        Args:
            file_path (Union[str, Path]): Path to the .npy file.

        Returns:
            pose_list (list[float]): List with position and orientation of the poses: [x, y, z, 0.0, 0.0, 0.0, 1.0].
        """
        pose_list = np.load(file_path).tolist()

        for position in pose_list:
            position.extend([0.0, 0.0, 0.0, 1.0])

        return pose_list

    def get_pose(self) -> np.ndarray:
        """Returns the pose from the poi as [x, y, z, a, b, c, w]."""
        return self.pose

    def get_position(self) -> np.ndarray:
        """Returns the position from the poi as [x, y, z]."""
        return self.pose[:3]

    def set_pose(self, pose: np.ndarray):
        """Sets the pose from the poi.

        Args:
            pose (np.ndarray): Pose of the poi
        """
        self.pose = pose
        with self.mechanical_object.position.writeable() as poi_pose:
            poi_pose[0] = self.pose

    def reset(self) -> None:
        """Reset the state of the PointOfInterest.

        Sets the color back to green and samples a new random pose (if ``randomized_pose``).
        """
        self.set_color(color=(0.0, 1.0, 0.0))
        if self.randomized_pose:
            self.pose_index = self.rng.integers(0, len(self.possible_target_poses))
            self.pose = np.array(self.possible_target_poses[self.pose_index])
            with self.mechanical_object.position.writeable() as pose:
                pose[0] = self.pose

    def set_color(self, color: Tuple[float, float, float]) -> None:
        set_color(self.visual_model_node.OglModel, color=color)

    def is_in_point_of_interest(self, position: np.ndarray) -> bool:
        """Checks whether a position is within the sphere that describes the PointOfInterest.
        Note:
            Depends on ``scale`` that is used as a radius.
        """
        return bool(np.linalg.norm(position - self.pose[:3]) < self.radius)

    def seed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Creates a random number generator from a seed."""
        self.rng = np.random.default_rng(seed)
