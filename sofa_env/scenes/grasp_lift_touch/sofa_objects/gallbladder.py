import numpy as np

from pathlib import Path
from typing import Tuple, Union
from functools import partial

import Sofa
import Sofa.Core

from sofa_env.sofa_templates.collision import COLLISION_PLUGIN_LIST, add_collision_model, CollisionModelType
from sofa_env.sofa_templates.deformable import DEFORMABLE_PLUGIN_LIST, DeformableObject
from sofa_env.sofa_templates.loader import add_loader
from sofa_env.sofa_templates.materials import MATERIALS_PLUGIN_LIST, Material
from sofa_env.sofa_templates.motion_restriction import MOTION_RESTRICTION_PLUGIN_LIST, add_bounding_box, add_rest_shape_spring_force_field_to_indices
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST

from sofa_env.sofa_templates.solver import SOLVER_PLUGIN_LIST, ConstraintCorrectionType
from sofa_env.sofa_templates.topology import TOPOLOGY_PLUGIN_LIST, TopologyTypes
from sofa_env.sofa_templates.visual import VISUAL_PLUGIN_LIST, add_visual_model

GALLBLADDER_PLUGIN_LIST = (
    [
        "SofaGeneralEngine",
        "Sofa.Component.MechanicalLoad",
    ]
    + DEFORMABLE_PLUGIN_LIST
    + MATERIALS_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
    + TOPOLOGY_PLUGIN_LIST
    + MOTION_RESTRICTION_PLUGIN_LIST
    + COLLISION_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
)


class Gallbladder(DeformableObject):
    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        volume_mesh_path: Union[str, Path],
        collision_mesh_path: Union[str, Path],
        visual_mesh_path: Union[str, Path],
        liver_mesh_path: Union[str, Path],
        name: str = "gallbladder",
        total_mass: float = 5.0,
        scale: float = 1.0,
        rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        volume_mesh_type: TopologyTypes = TopologyTypes.TETRA,
        constraint_correction_type: ConstraintCorrectionType = ConstraintCorrectionType.PRECOMPUTED,
        animation_loop: AnimationLoopType = AnimationLoopType.DEFAULT,
    ) -> None:
        material = Material(poisson_ratio=0.0, young_modulus=1500)

        collision_model_func = partial(
            add_collision_model,
            model_types=[CollisionModelType.TRIANGLE, CollisionModelType.POINT],
            collision_group=8,
            check_self_collision=True,
            contact_stiffness=1e2,
        )

        visual_model_func = partial(add_visual_model, color=(1, 1, 0))

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
            constraint_correction_type=constraint_correction_type,
            material=material,
            add_collision_model_func=collision_model_func,
            add_visual_model_func=visual_model_func,
            animation_loop_type=animation_loop,
        )

        self.collision_spring_force_field_indices = [
            39,
            24,
            22,
            25,
            29,
            40,
            0,
            163,
            27,
            60,
            164,
            61,
            59,
            179,
            28,
            20,
            26,
            2,
            1,
            74,
            99,
            109,
            82,
            196,
        ]

        self.restSpringsForceField = add_rest_shape_spring_force_field_to_indices(
            attached_to=self.collision_model_node,
            indices=self.collision_spring_force_field_indices,
            stiffness=1e04,
            angular_stiffness=1e04,
        )

        self.node.addObject(
            "UniformVelocityDampingForceField",
            template="Vec3d",
            name="Damping",
            dampingCoefficient=1,
        )

        mesh_roi_node = self.node.addChild("MeshROI")

        add_loader(attached_to=mesh_roi_node, file_path=liver_mesh_path, name="ROIloader", scale=scale)

        self.mesh_roi = mesh_roi_node.addObject(
            "MeshROI",
            computeMeshROI=True,
            doUpdate=True,
            position=self.node.MechanicalObject.position.getLinkPath(),
            ROIposition="@ROIloader.position",
            ROItriangles="@ROIloader.triangles",
        )

        self.graspable_region_box = add_bounding_box(
            attached_to=self.node,
            min=(-30, -30, -30),
            max=(30, 10, 10),
            name="grasp_box",
            extra_kwargs={"position": self.mechanical_object.position.getLinkPath()},
            rotation=(0, 0, -30),
        )

        self.graspable_region_box_collision = add_bounding_box(
            attached_to=self.collision_model_node,
            min=(-30, -30, -30),
            max=(30, 10, 10),
            name="grasp_box",
            extra_kwargs={"position": self.collision_model_node.MechanicalObject.position.getLinkPath()},
            rotation=(0, 0, -30),
        )

    def get_internal_force_magnitude(self) -> np.ndarray:
        """Get the sum of magnitudes of the internal forces applied to each vertex of the mesh"""
        return np.sum(np.linalg.norm(self.mechanical_object.force.array(), axis=1))

    def get_tissue_velocities(self) -> np.ndarray:
        """Get the velocities of the tissue vertices"""
        return self.mechanical_object.velocity.array()
