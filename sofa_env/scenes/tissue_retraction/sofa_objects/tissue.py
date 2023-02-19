import Sofa.Core
import Sofa.SofaDeformable

import numpy as np
from typing import Tuple, Optional, Union, Callable, List
from pathlib import Path

from sofa_env.sofa_templates.deformable import DeformableObject, DEFORMABLE_PLUGIN_LIST
from sofa_env.sofa_templates.loader import add_loader, LOADER_PLUGIN_LIST
from sofa_env.sofa_templates.materials import Material, MATERIALS_PLUGIN_LIST
from sofa_env.sofa_templates.topology import TopologyTypes, TOPOLOGY_PLUGIN_LIST
from sofa_env.sofa_templates.solver import add_solver, ConstraintCorrectionType, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST

TISSUE_PLUGIN_LIST = [] + DEFORMABLE_PLUGIN_LIST + MATERIALS_PLUGIN_LIST + TOPOLOGY_PLUGIN_LIST + SOLVER_PLUGIN_LIST + VISUAL_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST + LOADER_PLUGIN_LIST


class Tissue(Sofa.Core.Controller):
    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        volume_mesh_path: Union[str, Path],
        total_mass: float,
        visual_mesh_path: Union[str, Path],
        name: str = "tissue",
        rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        volume_mesh_type: TopologyTypes = TopologyTypes.TETRA,
        material: Optional[Material] = None,
        add_solver_func: Callable = add_solver,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        constraint_correction_type: ConstraintCorrectionType = ConstraintCorrectionType.PRECOMPUTED,
        show_object: bool = False,
        grasping_points_from: str = "visual",
    ):
        """Graspable tissue for tissue retraction.

        Args:
            parent_node (Sofa.Core.Node): Parent node of the object.
            name (str): Name of the object.
            volume_mesh_path (Union[str, Path]): Path to the volume mesh.
            total_mass (float): Total mass of the deformable object.
            visual_mesh_path (Optional[Union[str, Path]]): Path to the visual surface mesh.
            rotation (Tuple[float, float, float]): RPY rotations in degrees of the collision model in relation to the parent node. Order is X*Y*Z.
            translation (Tuple[float, float, float]): XYZ translation of the collision model in relation to the parent node.
            scale (float): Scale factor for loading the meshes.
            volume_mesh_type (TopologyTypes): Type of the volume mesh (e.g. TopologyTypes.TETRA for Tetraeder).
            material (Optional[Material]): Description of the material behavior.
            add_solver_func (Callable): Function that adds the numeric solvers to the object.
            add_visual_model_func (Callable): Function that defines how the visual surface from visual_mesh_path is added to the object.
            animation_loop_type (AnimationLoopType): The animation loop of the scene. Required to determine if objects for constraint correction should be added.
            constraint_correction_type (ConstraintCorrectionType): Type of constraint correction for the object, when animation_loop_type is AnimationLoopType.FREEMOTION.
            show_object (bool): Whether to render the nodes of the volume mesh.
            grasping_points_from (str): From which of the meshes ('volume' or 'visual') to select points used for describing grasping.
        """

        Sofa.Core.Controller.__init__(self)

        self.name = f"{name}_controller"

        self.deformable_object = DeformableObject(
            parent_node=parent_node,
            volume_mesh_path=volume_mesh_path,
            total_mass=total_mass,
            name=name,
            visual_mesh_path=visual_mesh_path,
            rotation=rotation,
            translation=translation,
            scale=scale,
            volume_mesh_type=volume_mesh_type,
            material=material,
            add_solver_func=add_solver_func,
            add_visual_model_func=add_visual_model_func,
            animation_loop_type=animation_loop_type,
            constraint_correction_type=constraint_correction_type,
            show_object=show_object,
        )

        # Add a graspable part to the tissue from the surface mesh and map it to the FEM part (that has a lower resolution)
        self.grasping_points_node = self.deformable_object.node.addChild("grasping_points_node")
        if grasping_points_from == "visual":
            grasping_points_mesh = visual_mesh_path
        elif grasping_points_from == "volume":
            grasping_points_mesh = volume_mesh_path
        else:
            raise ValueError(f"Unknown option for creating a grasping node <{grasping_points_from=}>. Valid values are 'visual' and 'volume'.")

        assert grasping_points_mesh is not None

        self.grasping_points_loader = add_loader(
            self.grasping_points_node,
            file_path=grasping_points_mesh,
            loader_kwargs={
                "translation": translation,
                "rotation": rotation,
            },
        )
        self.grasping_points_topology = self.grasping_points_node.addObject("MeshTopology", src=self.grasping_points_loader.getLinkPath())
        self.grasping_points_mechanical_object = self.grasping_points_node.addObject("MechanicalObject", template="Vec3d", showObject=show_object, showObjectScale=5.0, showColor=[0, 0, 255, 255])
        self.grasping_points_node.addObject("BarycentricMapping")

    def get_indices_and_distances_in_spherical_roi(self, sphere_center_position: np.ndarray, sphere_radius: float) -> Tuple[List[int], np.ndarray]:
        """Computes the indices of the grasping mechanical object that lie within a specified sphere."""
        distances = np.linalg.norm(self.grasping_points_mechanical_object.position.array() - sphere_center_position, axis=1)
        # Note: For some reason returns a tuple?!
        indices_in_sphere = np.asarray(distances <= sphere_radius).nonzero()[0]
        return indices_in_sphere.tolist(), distances
