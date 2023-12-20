import numpy as np
import Sofa
import Sofa.Core

from functools import partial
from typing import Tuple, Optional, Dict, Union, List
from pathlib import Path
from enum import Enum, unique

from sofa_env.sofa_templates.rigid import RigidObject, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, set_color, VISUAL_PLUGIN_LIST

from sofa_env.utils.math_helper import point_rotation_by_quaternion, rotated_y_axis

EYE_PLUGIN_LIST = RIGID_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST + VISUAL_PLUGIN_LIST


@unique
class EyeStates(Enum):
    """Possible states of an Eye in the task. Value of the enum is the respective color."""

    NEXT = (0.0, 0.0, 1.0)
    DONE = (0.0, 1.0, 0.0)
    OPEN = (1.0, 0.0, 0.0)
    TRANSITION = (1.0, 1.0, 0.0)


class Eye:
    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        position: Tuple[float, float, float],
        rotation: float,
        name: str,
        index: int,
        animation_loop_type: AnimationLoopType,
        surface_mesh_path: Path,
        show_object: bool = False,
        show_object_scale: float = 5.0,
        position_reset_noise: Optional[Dict[str, np.ndarray]] = None,
        total_mass: Optional[float] = None,
        collision_positions: Optional[List] = None,
        collision_group: Optional[int] = None,
        render_index: bool = True,
    ) -> None:
        # Placement of the peg
        quaternion = np.array([0, 0, np.sin(rotation * np.pi / 360), np.cos(rotation * np.pi / 360)])
        self.peg_pose = np.zeros(7)
        self.peg_pose[:3] = position
        self.peg_pose[3:] = quaternion

        # position and rotation will contain the current pose, initial_position and initial_rotation are used as
        # a base position that the noise is added to when resetting
        self.initial_position = position
        self.position = position
        self.initial_rotation = rotation
        self.rotation = rotation

        self.index = index
        self.state = EyeStates.OPEN

        self.name = name

        self.position_reset_noise = position_reset_noise

        # Rigid object
        visual_eye_model_func = partial(add_visual_model, color=self.state.value)
        self.rigid_object = RigidObject(
            parent_node=parent_node,
            name=name,
            pose=self.peg_pose,
            fixed_position=True,
            fixed_orientation=False,
            visual_mesh_path=surface_mesh_path,
            animation_loop_type=animation_loop_type,
            show_object=show_object,
            show_object_scale=show_object_scale,
            add_visual_model_func=visual_eye_model_func,
            total_mass=total_mass,
        )

        # Center of the eye
        self.center_node = self.rigid_object.node.addChild("center")
        self.center_pose = np.zeros(7, dtype=np.float64)
        self.center_pose[:3] = np.array(position) + point_rotation_by_quaternion(np.array([1.55, 0.0, 18.7]), quaternion)
        self.center_pose[3:] = quaternion
        self.center_node.addObject("MechanicalObject", template="Rigid3d", position=self.center_pose, showObject=show_object, showObjectScale=show_object_scale)
        self.center_node.addObject("RigidMapping", template="Rigid3,Rigid3", globalToLocalCoords=True)

        self.render_index = render_index
        if render_index:
            self.text = self.rigid_object.node.addObject("Visual3DText", text=str(index), position=[position[0] + 5, position[1], 10], scale=8)

        # Sphere collision models
        if collision_positions is None:
            collision_positions = [
                # shaft
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 6.0],
                [0.0, 0.0, 10.0],
                # arc bottom
                [1.6, 0.0, 13.5],
                # left arc
                [-2.0, 0.0, 15.5],
                [-3.3, 0.0, 19.0],
                [-2.0, 0.0, 22.0],
                # arc top
                [1.6, 0.0, 23.5],
                # right arc
                [5.0, 0.0, 22.0],
                [6.3, 0.0, 19.0],
                [5.0, 0.0, 15.5],
            ]

        self.collision_node = self.rigid_object.node.addChild("collision")
        self.collision_node.addObject("MechanicalObject", template="Vec3d", position=collision_positions)
        collision_model_kwargs = {}
        if collision_group is not None:
            collision_model_kwargs["group"] = collision_group
        self.collision_node.addObject("SphereCollisionModel", radius=2.0, **collision_model_kwargs)
        self.collision_node.addObject("RigidMapping")

        self.normal_vector = rotated_y_axis(quaternion)

    def set_color(self, color: Tuple[float, float, float]) -> None:
        """Set the color of the visual model.

        Args:
            color (Tuple[float, float, float]): RGB values of the new color in [0, 1].
        """
        set_color(self.rigid_object.visual_model_node.OglModel, color)

    def set_state(self, state: EyeStates, color_eyes: bool = False) -> None:
        """Set the state of the eye and optionally the color of the visual model.

        Args:
            state (EyeStates): The state of the eye.
            color_eyes (bool): Whether to adapt eye color to current state.
        """
        self.state = state
        if color_eyes:
            self.set_color(state.value)

    def get_state(self) -> EyeStates:
        """Get the state of the eye.

        Returns:
            state (EyeStates): The state of the eye.
        """
        return self.state

    def points_are_right_of_eye(self, points: np.ndarray) -> np.ndarray:
        """Checks if points are right of the eye based on checking the distance to a plane.

        Args:
            points (np.ndarray): an Nx3 array of cartesian points.

        Returns:
            signs (np.ndarray): an Nx1 array of {1, -1} if the nth point is {right, left} of the plane.

        """
        return np.sign(np.dot((self.center_pose[:3] - points), self.normal_vector))

    def reset(self) -> None:
        """If the Eye was initialized with position_reset_noise, add some noise to the initial position and rotation of the Eye."""
        if self.position_reset_noise is not None:
            xyzphi = np.append(self.initial_position, self.initial_rotation) + self.rng.uniform(
                self.position_reset_noise["low"],
                self.position_reset_noise["high"],
            )

            self.set_xyzphi(xyzphi)

    def set_xyzphi(self, xyzphi: np.ndarray) -> None:
        quaternion = np.array([0, 0, np.sin(xyzphi[-1] * np.pi / 360), np.cos(xyzphi[-1] * np.pi / 360)])

        self.peg_pose[:3] = xyzphi[:3]
        self.peg_pose[3:] = quaternion
        self.center_pose[:3] = xyzphi[:3] + point_rotation_by_quaternion(np.array([1.55, 0.0, 18.7]), quaternion)
        self.center_pose[3:] = quaternion
        self.position = xyzphi[:3]
        self.rotation = xyzphi[3]

        new_pose = np.append(xyzphi[:3], quaternion)
        with self.rigid_object.mechanical_object.position.writeable() as pose:
            pose[:] = new_pose

        if self.render_index:
            with self.text.position.writeable() as position:
                position[:] = xyzphi[:3]

    def set_center_xyzphi(self, xyzphi: np.ndarray) -> None:
        quaternion = np.array([0, 0, np.sin(xyzphi[-1] * np.pi / 360), np.cos(xyzphi[-1] * np.pi / 360)])
        d = point_rotation_by_quaternion(np.array([1.55, 0.0, 18.7]), quaternion)
        self.peg_pose[:3] = xyzphi[:3] - d
        self.peg_pose[3:] = quaternion
        self.center_pose[:3] = xyzphi[:3] 
        self.center_pose[3:] = quaternion
        self.position = xyzphi[:3] - d
        self.rotation = xyzphi[3]

        new_pose = np.append(xyzphi[:3] - d, quaternion)
        with self.rigid_object.mechanical_object.position.writeable() as pose:
            pose[:] = new_pose

        if self.render_index:
            with self.text.position.writeable() as position:
                position[:] = xyzphi[:3] - d

    def seed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Creates a random number generator from a seed."""
        self.rng = np.random.default_rng(seed)
