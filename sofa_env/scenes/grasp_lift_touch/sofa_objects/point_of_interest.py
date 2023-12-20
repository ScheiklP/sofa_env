import numpy as np

from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union, List, Tuple

import Sofa
import Sofa.Core

from sofa_env.sofa_templates.rigid import RigidObject, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, set_color, VISUAL_PLUGIN_LIST

POI_PLUGIN_LIST = RIGID_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST + VISUAL_PLUGIN_LIST

TARGET_POSES = [
    (-50.0, 15.0, -25.0, 0.0, 0.0, 0.0, 1.0),
    (-55.0, 10.0, -20.0, 0.0, 0.0, 0.0, 1.0),
    (-60.0, 8.0, -15.0, 0.0, 0.0, 0.0, 1.0),
]


class PointOfInterest(RigidObject):
    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        randomized_pose: bool = True,
        initial_position: np.ndarray = np.zeros(3),
        visual_mesh_path: Optional[Union[str, Path]] = None,
        scale: float = 3,
        add_visual_model_func: Callable = partial(add_visual_model, color=(1, 1, 1)),
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        show_object: bool = False,
        show_object_scale: float = 7,
        non_default_target_positions: Optional[List] = None,
    ) -> None:
        self.randomized_pose = randomized_pose
        if non_default_target_positions is None:
            self.possible_target_poses = TARGET_POSES
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

        self.scale = scale
        super().__init__(
            parent_node=parent_node,
            name=name,
            pose=self.pose,
            fixed_position=False,
            fixed_orientation=False,
            visual_mesh_path=visual_mesh_path,
            scale=scale,
            add_visual_model_func=add_visual_model_func,
            animation_loop_type=animation_loop_type,
            show_object=show_object,
            show_object_scale=show_object_scale,
        )

        self.activated = False

    def reset(self) -> None:
        """Reset the state of the PointOfInterest.

        Sets the color back to white and samples a new random pose (if ``randomized_pose``).
        """
        self.set_color(color=(1.0, 1.0, 1.0))
        self.activated = False
        if self.randomized_pose:
            self.pose_index = self.rng.integers(0, len(self.possible_target_poses))
        else:
            self.pose_index = 0
        self.pose = np.array(self.possible_target_poses[self.pose_index])
        with self.mechanical_object.position.writeable() as positions:
            positions[0] = self.pose

    def set_color(self, color: Tuple[float, float, float]) -> None:
        set_color(self.visual_model_node.OglModel, color=color)

    def is_in_point_of_interest(self, position: np.ndarray) -> bool:
        """Checks whether a position is within the sphere that describes the PointOfInterest.
        Note:
            Depends on ``scale`` that is used as a radius.
        """
        return bool(np.linalg.norm(position - self.pose[:3]) < self.scale)

    def seed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Creates a random number generator from a seed."""
        self.rng = np.random.default_rng(seed)
