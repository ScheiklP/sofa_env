from pathlib import Path
from typing import Tuple, Optional, Union, Callable, List
import numpy as np

import Sofa.Core

from sofa_env.sofa_templates.collision import COLLISION_PLUGIN_LIST, add_collision_model
from sofa_env.sofa_templates.deformable import DEFORMABLE_PLUGIN_LIST, DeformableObject, check_file_path
from sofa_env.sofa_templates.materials import MATERIALS_PLUGIN_LIST, Material
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.solver import SOLVER_PLUGIN_LIST, add_solver, ConstraintCorrectionType
from sofa_env.sofa_templates.topology import TOPOLOGY_PLUGIN_LIST, TopologyTypes
from sofa_env.sofa_templates.visual import VISUAL_PLUGIN_LIST, add_visual_model
from sofa_env.sofa_templates.mappings import MappingType, MAPPING_PLUGIN_LIST
from sofa_env.sofa_templates.motion_restriction import MOTION_RESTRICTION_PLUGIN_LIST

TISSUE_PLUGIN_LIST = (
    [
        "Sofa.Component.SolidMechanics.FEM.HyperElastic",  # <- [TetrahedronHyperelasticityFEMForceField]
    ]
    + DEFORMABLE_PLUGIN_LIST
    + MATERIALS_PLUGIN_LIST
    + TOPOLOGY_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
    + COLLISION_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
    + MAPPING_PLUGIN_LIST
    + MOTION_RESTRICTION_PLUGIN_LIST
)


class Tissue(Sofa.Core.Controller, DeformableObject):
    """Python object that creates SOFA objects and python logic to represent a deformable tissue which can be rigidified.

     Notes:
        - Manipulation Target is also known as "Tissue Target Point" (TTP).

    Args:
        parent_node (Sofa.Core.Node): Parent node of the object.
        name (str): Name of the object.
        volume_mesh_path (Union[str, Path]): Path to the volume mesh.
        visual_mesh_path (Optional[Union[str, Path]]): Path to the visual surface mesh.
        collision_mesh_path (Optional[Union[str, Path]]): Path to the collision surface mesh.
        total_mass (float): Total mass of the deformable object.
        rotation (Tuple[float, float, float]): RPY rotations in degrees of the collision model in relation to the parent node. Order is X*Y*Z.
        translation (Tuple[float, float, float]): XYZ translation of the collision model in relation to the parent node.
        scale (float): Scale factor for loading the meshes.
        volume_mesh_type (TopologyTypes): Type of the volume mesh (e.g. ``TopologyTypes.TETRA`` for Tetraeder).
        material (Optional[Material]): Description of the material behavior.
        add_solver_func (Callable): Function that adds the numeric solvers to the object.
        add_collision_model_func (Callable): Function that defines how the collision surface from ``collision_mesh_path`` is added to the object.
        add_visual_model_func (Callable): Function that defines how the visual surface from ``visual_mesh_path`` is added to the object.
        animation_loop_type (AnimationLoopType): The animation loop of the scene. Required to determine if objects for constraint correction should be added.
        constraint_correction_type (ConstraintCorrectionType): Type of constraint correction for the object, when ``animation_loop_type`` is ``AnimationLoopType.FREEMOTION``.
        with_manipulation_target (bool): Adds Manipulation Target == Tissue Target Point (TTP) to the tissue.
        manipulation_target_mesh_path (Optional[Union[str, Path]]): Path to the visual surface mesh of the manipulation target.
        manipulation_target_color (Optional[Tuple[float, float, float]]): Color for the manipulation target / TTP.
        randomize_manipulation_target (bool): Specify whether the position of the TTP on the tissue is randomly sampled.
        show_manipulation_target_candidates (bool): Specify whether the TTP candidates should be shown.
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        volume_mesh_path: Union[str, Path],
        total_mass: float,
        name: str = "tissue",
        visual_mesh_path: Optional[Union[str, Path]] = None,
        collision_mesh_path: Optional[Union[str, Path]] = None,
        rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        volume_mesh_type: TopologyTypes = TopologyTypes.TETRA,
        material: Optional[Material] = None,
        add_solver_func: Callable = add_solver,
        add_collision_model_func: Callable = add_collision_model,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        constraint_correction_type: ConstraintCorrectionType = ConstraintCorrectionType.PRECOMPUTED,
        with_manipulation_target: bool = True,
        manipulation_target_mesh_path: Optional[Union[str, Path]] = None,
        manipulation_target_color: Optional[Tuple[float, float, float]] = None,
        randomize_manipulation_target: bool = True,
        show_manipulation_target_candidates: bool = False,
    ):
        Sofa.Core.Controller.__init__(self)
        DeformableObject.__init__(
            self,
            parent_node=parent_node,
            volume_mesh_path=volume_mesh_path,
            total_mass=total_mass,
            name=name + "_deformable",
            rotation=rotation,
            translation=translation,
            scale=scale,
            volume_mesh_type=volume_mesh_type,
            material=material,
            add_solver_func=add_solver_func,
            animation_loop_type=animation_loop_type,
            constraint_correction_type=constraint_correction_type,
            visual_mesh_path=None,
            collision_mesh_path=None,
        )

        self.name = name
        self.rotation = rotation
        self.translation = translation

        self.fix_target_idx = 17

        self.target_idx = 17
        self.target_idx_id = 2

        self.min_displacement = -1
        self.max_displacement = 1e-3 * 2.5  # 5 mm per step
        self.previous_mechanical_state = None
        self.displacement_norm: float
        self.is_stable = True

        self.with_manipulation_target = with_manipulation_target
        self.randomize_manipulation_target = randomize_manipulation_target

        # Add manipulation target - Tissue Target Point (TTP)
        if with_manipulation_target:
            self._add_tissue_target_point(mesh=manipulation_target_mesh_path, color=manipulation_target_color)

        # add collision models to the object
        if collision_mesh_path is not None:
            self.collision_model_node = add_collision_model_func(attached_to=self.node, surface_mesh_file_path=check_file_path(collision_mesh_path), translation=translation, rotation=rotation)

        # add a visual model to the object
        if visual_mesh_path is not None:
            self.visual_model_node = add_visual_model_func(attached_to=self.node, surface_mesh_file_path=check_file_path(visual_mesh_path), translation=translation, rotation=rotation)

        self.manipulation_target_indices = np.array([3, 4, 17, 21, 22, 5, 34, 20, 26])
        self.show_manipulation_target_candidates = show_manipulation_target_candidates
        if show_manipulation_target_candidates:
            self.visual_ttp_node = self.node.addChild("ManipulationCandidates")
            self.visual_ttp_node.addObject("MechanicalObject", template="Vec3d", showObject=True, showObjectScale=5.0, showColor=[1.0, 0.0, 1.0])
            self.visual_ttp_node.addObject("SubsetMapping", indices=self.manipulation_target_indices, input=self.collision_model_node.MechanicalObject.getLinkPath())

    def onSimulationInitDoneEvent(self, _) -> None:
        self.previous_mechanical_state = self.mechanical_object.position.array()

    def onAnimateEndEvent(self, _) -> None:
        # Check for simulation instability at the end of each time step
        current_mechanical_state = self.mechanical_object.position.array()
        displacement = current_mechanical_state - self.previous_mechanical_state

        self.is_stable, self.displacement_norm = self.displacements_are_valid(displacement)
        self.previous_mechanical_state = current_mechanical_state

        if self.with_manipulation_target:
            target_pose = np.zeros(7)
            target_pose[:3] = self.collision_model_node.MechanicalObject.position.array()[self.target_idx]
            target_pose[-1] = 1.0
            self._set_tissue_target_point_pose(target_pose)

    def displacements_are_valid(self, displacement: np.ndarray) -> Tuple[bool, float]:
        """Computes if an array of displacements is inside defined bounds.

        Notes:
            False, if:
                    * there is at least one displacement with NaN value
                    * there is at least one displacement whose norm is <= min_displacement
                    * there is at least one displacement whose norm is >= max_displacement
            otherwise, returns True.

        Args:
            displacement (np.ndarray): A Nx3 array with x,y,z displacements of N points.

        Returns:
            valid (bool): Valid value for the largest displacement.
        """

        displacement_norm = np.linalg.norm(displacement, axis=1)
        maximum_displacement_norm = np.amax(displacement_norm)
        ret = True

        if np.isnan(self.max_displacement):
            if np.isnan(maximum_displacement_norm):
                ret = False
        elif maximum_displacement_norm >= self.max_displacement:
            ret = False
        elif maximum_displacement_norm <= self.min_displacement:
            ret = False

        return ret, maximum_displacement_norm

    def get_displacement_norm(self) -> float:
        """Returns norm of displacements"""
        return self.displacement_norm

    def get_indices_in_spherical_roi(self, sphere_center_position: np.ndarray, sphere_radius: float) -> List[int]:
        """Computes the indices of the mechanical object that lie within a specified sphere."""
        distances = np.linalg.norm(self.mechanical_object.position.array() - sphere_center_position, axis=1)
        # for some reason returns a tuple?!
        indices_in_sphere = np.asarray(distances <= sphere_radius).nonzero()[0]
        return indices_in_sphere.tolist()

    def get_manipulation_target_pose(self) -> np.ndarray:
        """Reads the Rigid3d pose from the Manipulation Target MO as [x, y, z, a, b, c, w]."""
        return self.manipulation_target_mechanical_object.position.array()[0]

    def reset(self) -> None:
        """Resets rigidified Tissue. Resets Tissue Target Point (TTP; ManipulationTarget) on Tissue."""
        self.target_idx = self.fix_target_idx
        self.previous_mechanical_state = self.mechanical_object.position.array()

        if self.with_manipulation_target and self.randomize_manipulation_target:
            self.target_idx_id = self.rng.choice(list(range(len(self.manipulation_target_indices))))
            self.target_idx = self.manipulation_target_indices[self.target_idx_id]

    def _add_tissue_target_point(
        self,
        mesh: Union[str, Path],
        color: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """
        Adds TTP to tissue.
        """
        self.target_idx = self.fix_target_idx
        self.manipulation_target = self.node.addChild("manipulation_target")
        self.manipulation_target_mechanical_object = self.manipulation_target.addObject(
            "MechanicalObject",
            template="Rigid3d",
            position=[0.0] * 6 + [1.0],
        )

        self.visual_model_node = add_visual_model(
            attached_to=self.manipulation_target,
            surface_mesh_file_path=mesh,
            mapping_type=MappingType.RIGID,
            color=color,
        )

    def _set_tissue_target_point_pose(self, pose: np.ndarray) -> None:
        """Writes the Rigid3d pose from the controllable sofa node as [x, y, z, a, b, c, w].

        Notes:
            - pose values are written into the sofa array without assigning the pose array to the sofa array.
              Changes in the pose array after that will not be propagated to sofa.
        """
        with self.manipulation_target_mechanical_object.position.writeable() as state:
            state[:] = pose

    def seed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Creates a random number generator from a seed."""
        self.rng = np.random.default_rng(seed)
