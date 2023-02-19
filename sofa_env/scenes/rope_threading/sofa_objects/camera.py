import Sofa
import Sofa.Core
from sofa_env.sofa_templates.camera import PhysicalCamera, DEFAULT_LIGHT_SOURCE_KWARGS, CAMERA_PLUGIN_LIST

from typing import Optional, Union, Callable, Tuple
from pathlib import Path
import numpy as np

from sofa_env.sofa_templates.rigid import MechanicalBinding, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.solver import add_solver, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST
from sofa_env.sofa_templates.collision import add_collision_model, COLLISION_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.utils.math_helper import euler_to_rotation_matrix, multiply_quaternions, rotation_matrix_to_quaternion

CONTROLLABLE_CAMERA_KWARGS = [] + CAMERA_PLUGIN_LIST + RIGID_PLUGIN_LIST + SOLVER_PLUGIN_LIST + VISUAL_PLUGIN_LIST + COLLISION_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST


class ControllableCamera(Sofa.Core.Controller, PhysicalCamera):
    def __init__(
        self,
        root_node: Sofa.Core.Node,
        placement_kwargs: dict = {
            "position": [0.0, 0.0, 0.0],
            "lookAt": [0.0, 0.0, 0.0],
        },
        vertical_field_of_view: Union[float, int] = 45,
        z_near: Optional[float] = None,
        z_far: Optional[float] = None,
        width_viewport: Optional[int] = None,
        height_viewport: Optional[int] = None,
        show_object: bool = False,
        show_object_scale: float = 1.0,
        total_mass: Optional[float] = None,
        visual_mesh_path: Optional[Union[str, Path]] = None,
        collision_mesh_path: Optional[Union[str, Path]] = None,
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        add_solver_func: Callable = add_solver,
        add_collision_model_func: Callable = add_collision_model,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        mechanical_binding: MechanicalBinding = MechanicalBinding.ATTACH,
        spring_stiffness: Optional[float] = 2e10,
        angular_spring_stiffness: Optional[float] = 2e10,
        with_light_source: bool = False,
        light_source_kwargs: dict = DEFAULT_LIGHT_SOURCE_KWARGS,
    ):
        Sofa.Core.Controller.__init__(self)

        PhysicalCamera.__init__(
            self,
            root_node=root_node,
            placement_kwargs=placement_kwargs,
            vertical_field_of_view=vertical_field_of_view,
            z_near=z_near,
            z_far=z_far,
            width_viewport=width_viewport,
            height_viewport=height_viewport,
            show_object=show_object,
            show_object_scale=show_object_scale,
            total_mass=total_mass,
            visual_mesh_path=visual_mesh_path,
            collision_mesh_path=collision_mesh_path,
            scale=scale,
            add_solver_func=add_solver_func,
            add_collision_model_func=add_collision_model_func,
            add_visual_model_func=add_visual_model_func,
            animation_loop_type=animation_loop_type,
            mechanical_binding=mechanical_binding,
            spring_stiffness=spring_stiffness,
            angular_spring_stiffness=angular_spring_stiffness,
            with_light_source=with_light_source,
            light_source_kwargs=light_source_kwargs,
        )

    def onKeypressedEvent(self, event):
        key = event["key"]
        if ord(key) == 19:  # up
            self.set_pose(self.get_physical_body_pose() + np.array([0, 0, 1] + [0, 0, 0, 0], dtype=np.float32))

        elif ord(key) == 21:  # down
            self.set_pose(self.get_physical_body_pose() + np.array([0, 0, -1] + [0, 0, 0, 0], dtype=np.float32))

        elif ord(key) == 18:  # left
            self.set_pose(self.get_physical_body_pose() + np.array([-1, 0, 0] + [0, 0, 0, 0], dtype=np.float32))

        elif ord(key) == 20:  # right
            self.set_pose(self.get_physical_body_pose() + np.array([1, 0, 0] + [0, 0, 0, 0], dtype=np.float32))

        elif key == "T":  # forward
            self.set_pose(self.get_physical_body_pose() + np.array([0, 1, 0] + [0, 0, 0, 0], dtype=np.float32))

        elif key == "G":  # backward
            self.set_pose(self.get_physical_body_pose() + np.array([0, -1, 0] + [0, 0, 0, 0], dtype=np.float32))

        elif key == "V":  # Rotation X
            old_pose = self.get_physical_body_pose()
            orientation_delta = rotation_matrix_to_quaternion(euler_to_rotation_matrix(np.array([1, 0, 0])))
            new_pose = np.zeros_like(old_pose)
            new_pose[:3] = old_pose[:3]
            new_pose[3:] = multiply_quaternions(orientation_delta, old_pose[3:])
            self.set_pose(new_pose)

        elif key == "D":  # Rotation -X
            old_pose = self.get_physical_body_pose()
            orientation_delta = rotation_matrix_to_quaternion(euler_to_rotation_matrix(np.array([-1, 0, 0])))
            new_pose = np.zeros_like(old_pose)
            new_pose[:3] = old_pose[:3]
            new_pose[3:] = multiply_quaternions(orientation_delta, old_pose[3:])
            self.set_pose(new_pose)

        elif key == "B":  # Rotation Y
            old_pose = self.get_physical_body_pose()
            orientation_delta = rotation_matrix_to_quaternion(euler_to_rotation_matrix(np.array([0, 1, 0])))
            new_pose = np.zeros_like(old_pose)
            new_pose[:3] = old_pose[:3]
            new_pose[3:] = multiply_quaternions(orientation_delta, old_pose[3:])
            self.set_pose(new_pose)

        elif key == "P":  # Rotation -Y
            old_pose = self.get_physical_body_pose()
            orientation_delta = rotation_matrix_to_quaternion(euler_to_rotation_matrix(np.array([0, -1, 0])))
            new_pose = np.zeros_like(old_pose)
            new_pose[:3] = old_pose[:3]
            new_pose[3:] = multiply_quaternions(orientation_delta, old_pose[3:])
            self.set_pose(new_pose)

        elif key == "Y":  # Rotation Z
            old_pose = self.get_physical_body_pose()
            orientation_delta = rotation_matrix_to_quaternion(euler_to_rotation_matrix(np.array([0, 0, 1])))
            new_pose = np.zeros_like(old_pose)
            new_pose[:3] = old_pose[:3]
            new_pose[3:] = multiply_quaternions(orientation_delta, old_pose[3:])
            self.set_pose(new_pose)

        elif key == "U":  # Rotation -Z
            old_pose = self.get_physical_body_pose()
            orientation_delta = rotation_matrix_to_quaternion(euler_to_rotation_matrix(np.array([0, 0, -1])))
            new_pose = np.zeros_like(old_pose)
            new_pose[:3] = old_pose[:3]
            new_pose[3:] = multiply_quaternions(orientation_delta, old_pose[3:])
            self.set_pose(new_pose)

        elif key == "H":
            print(repr(self.get_camera_pose()))
        else:
            pass
