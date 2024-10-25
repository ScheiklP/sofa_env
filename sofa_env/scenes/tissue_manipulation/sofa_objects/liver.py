import Sofa.Core

from typing import Tuple, Optional, Union, Callable
from pathlib import Path

from sofa_env.sofa_templates.rigid import RIGID_PLUGIN_LIST, RigidObject
from sofa_env.sofa_templates.solver import SOLVER_PLUGIN_LIST, add_solver, ConstraintCorrectionType
from sofa_env.sofa_templates.loader import check_file_path, LOADER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST
from sofa_env.sofa_templates.collision import add_collision_model, COLLISION_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.mappings import MappingType, MAPPING_PLUGIN_LIST

LIVER_PLUGIN_LIST = [] + RIGID_PLUGIN_LIST + SOLVER_PLUGIN_LIST + VISUAL_PLUGIN_LIST + COLLISION_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST + MAPPING_PLUGIN_LIST + LOADER_PLUGIN_LIST


class FixedLiver(RigidObject):
    """Python object that creates SOFA objects and python logic to represent a rigid Liver.

    Args:
        parent_node (Sofa.Core.Node): Parent node of the object.
        name (str): Name of the object.
        visual_mesh_path (Optional[Union[str, Path]]): Path to the visual surface mesh of the liver.
        collision_mesh_path (Optional[Union[str, Path]]): Path to the collision mesh of the liver.
        pose (Tuple[float, float, float, float, float, float, float]): 6D pose of the object described as Cartesian position and quaternion.
        total_mass (Optional[float]): Total mass of the object. Uniformly split to nodes.
        scale (float): Scale factor for loading the meshes.
        add_solver_func (Callable): Function that adds the numeric solvers to the object.
        add_collision_model_func (Callable): Function that defines how the collision surface from ``collision_mesh_path`` is added to the object.
        add_visual_model_func (Callable): Function that defines how the visual surface from ``visual_mesh_path`` is added to the object.
        animation_loop_type (AnimationLoopType): The animation loop of the scene. Required to determine if objects for constraint correction should be added.
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        pose: Tuple[float, float, float, float, float, float, float],
        total_mass: Optional[float] = None,
        visual_mesh_path: Optional[Union[str, Path]] = None,
        collision_mesh_path: Optional[Union[str, Path]] = None,
        scale: float = 1.0,
        add_solver_func: Callable = add_solver,
        add_collision_model_func: Callable = add_collision_model,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
    ) -> None:
        self.parent_node = parent_node
        self.node = self.parent_node.addChild(name)

        # add the solvers
        self.time_integration, self.linear_solver = add_solver_func(attached_to=self.node)

        # add the mechanical object that holds the mechanical state of the rigid object
        self.mechanical_object = self.node.addObject("MechanicalObject", template="Rigid3d", position=pose)
        self.mechanical_object = self.node.addObject("FixedProjectiveConstraint", fixAll=True)

        # add mass to the object
        if total_mass is not None:
            self.mass = self.node.addObject("UniformMass", totalMass=total_mass)

        # if using the FreeMotionAnimationLoop add a constraint correction object
        if animation_loop_type == AnimationLoopType.FREEMOTION:
            self.constraint_correction = self.node.addObject(
                ConstraintCorrectionType.UNCOUPLED.value,
                compliance=1.0 / total_mass if total_mass is not None else 1.0,
            )

        # add collision models to the object
        if collision_mesh_path is not None:
            self.collision_model_node = add_collision_model_func(
                attached_to=self.node,
                surface_mesh_file_path=check_file_path(collision_mesh_path),
                scale=scale,
                mapping_type=MappingType.RIGID,
            )

        # add a visual model to the object
        if visual_mesh_path is not None:
            self.visual_model_node = add_visual_model_func(
                attached_to=self.node,
                surface_mesh_file_path=check_file_path(visual_mesh_path),
                scale=scale,
                mapping_type=MappingType.RIGID,
            )
