import numpy as np
from pathlib import Path
from typing import Tuple, Union
from functools import partial
import Sofa.Core
from sofa_env.sofa_templates.collision import CollisionModelType, add_collision_model

from sofa_env.sofa_templates.motion_restriction import add_rest_shape_spring_force_field_in_bounding_box
from sofa_env.sofa_templates.scene_header import AnimationLoopType
from sofa_env.sofa_templates.solver import ConstraintCorrectionType
from sofa_env.sofa_templates.topology import TopologyTypes
from sofa_env.sofa_templates.visual import add_visual_model
from sofa_env.sofa_templates.materials import MATERIALS_PLUGIN_LIST, Material
from sofa_env.sofa_templates.deformable import DeformableObject, DEFORMABLE_PLUGIN_LIST

LIVER_PLUGIN_LIST = [] + DEFORMABLE_PLUGIN_LIST + MATERIALS_PLUGIN_LIST + DEFORMABLE_PLUGIN_LIST


class Liver(DeformableObject):
    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        volume_mesh_path: Union[str, Path],
        visual_mesh_path: Union[str, Path],
        collision_mesh_path: Union[str, Path],
        animation_loop: AnimationLoopType = AnimationLoopType.DEFAULT,
        name: str = "liver",
        total_mass: float = 15,
        scale: float = 1.0,
        rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        volume_mesh_type: TopologyTypes = TopologyTypes.TETRA,
        constraint_correction_type: ConstraintCorrectionType = ConstraintCorrectionType.PRECOMPUTED,
    ) -> None:
        material = Material(poisson_ratio=0.0, young_modulus=3000)

        collision_model_func = partial(
            add_collision_model,
            collision_group=8,
            model_types=[CollisionModelType.TRIANGLE],
            contact_stiffness=1e4,
        )

        visual_model_func = partial(add_visual_model, color=(1, 0, 0))

        super().__init__(
            parent_node=parent_node,
            name=name,
            volume_mesh_path=volume_mesh_path,
            total_mass=total_mass,
            visual_mesh_path=visual_mesh_path,
            collision_mesh_path=collision_mesh_path,
            rotation=rotation,
            translation=translation,
            scale=scale,
            volume_mesh_type=volume_mesh_type,
            material=material,
            constraint_correction_type=constraint_correction_type,
            add_collision_model_func=collision_model_func,
            add_visual_model_func=visual_model_func,
            animation_loop_type=animation_loop,
        )

        self.fix_indices_in_bounding_box(
            min=(-100, -70, 30),
            max=(100, 80, 100),
            fixture_func=add_rest_shape_spring_force_field_in_bounding_box,
            show_bounding_box=False,
            fixture_func_kwargs={
                "show_bounding_box_scale": 2.0,
                "stiffness": 1e04,
                "angular_stiffness": 1e09,
            },
        )

    def get_internal_force_magnitude(self) -> np.ndarray:
        """Get the sum of magnitudes of the internal forces applied to each vertex of the mesh"""
        return np.sum(np.linalg.norm(self.mechanical_object.force.array(), axis=1))
