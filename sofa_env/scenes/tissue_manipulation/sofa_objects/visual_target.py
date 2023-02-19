import Sofa.Core
import numpy as np

from pathlib import Path
from typing import Callable, Union, Tuple, Optional

from sofa_env.sofa_templates.rigid import ControllableRigidObject, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.solver import add_solver, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST

VISUAL_TARGET_PLUGIN_LIST = RIGID_PLUGIN_LIST + SOLVER_PLUGIN_LIST + VISUAL_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST


class VisualTarget(ControllableRigidObject):
    """Python object that creates SOFA objects and python logic to represent a Visual Target that can be controlled by updating its pose.

    Notes:
        - Visual Target is also known as "Image Desired Point" (IDP).

    Args:
        parent_node (Sofa.Core.Node): Parent node of the object.
        name (str): Name of the object.
        visual_mesh_path (Optional[Union[str, Path]]): Path to the visual surface mesh.
        pose (Union[Tuple[float, float, float, float, float, float, float], np.ndarray]): Initial 6D pose of the object described as Cartesian position and quaternion.
        scale (float): Scale factor for loading the meshes.
        add_solver_func (Callable): Function that adds the numeric solvers to the object.
        add_visual_model_func (Callable): Function that defines how the visual surface from ``visual_mesh_path`` is added to the object.
        animation_loop_type (AnimationLoopType): The animation loop of the scene. Required to determine if objects for constraint correction should be added.
        show_object (bool): Whether to render the pose frame of the object.
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        pose: Tuple[float, float, float, float, float, float, float],
        total_mass: Optional[float] = None,
        visual_mesh_path: Optional[Union[str, Path]] = None,
        scale: float = 1.0,
        add_solver_func: Callable = add_solver,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        show_object: bool = False,
    ) -> None:
        super().__init__(
            parent_node=parent_node,
            name=name,
            pose=pose,
            total_mass=total_mass,
            visual_mesh_path=visual_mesh_path,
            scale=scale,
            add_solver_func=add_solver_func,
            add_visual_model_func=add_visual_model_func,
            animation_loop_type=animation_loop_type,
            show_object=show_object,
        )

    def reset(self, new_position: np.ndarray) -> None:
        """Resets VisualTarget - IDP. Activates visibility and sets to random position."""
        self.set_visibility(True)
        self.set_pose(np.append(new_position, [0.0, 0.0, 0.0, 1.0]))

    def set_visibility(self, val: bool) -> None:
        """Activate / Deactivate visibility of VisualTarget."""
        self.visual_model_node.OglModel.isEnabled.value = int(val)
