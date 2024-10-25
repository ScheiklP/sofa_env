from pathlib import Path
from typing import Callable, Optional, Tuple, Union, List
from enum import Enum, unique

import Sofa
import Sofa.Core

import numpy as np

from sofa_env.sofa_templates.scene_header import SCENE_HEADER_PLUGIN_LIST, AnimationLoopType
from sofa_env.sofa_templates.solver import SOLVER_PLUGIN_LIST, ConstraintCorrectionType, LinearSolverType, add_solver
from sofa_env.sofa_templates.motion_restriction import MOTION_RESTRICTION_PLUGIN_LIST, add_bounding_box
from sofa_env.utils.math_helper import euler_angles_to_quaternion

TISSUE_PLUGIN_LIST = (
    [
        "Sofa.Component.Visual",  # <- [LineAxis]
        "Sofa.Component.Engine.Transform",  # <- [TransformEngine]
        "Sofa.Component.Topology.Container.Grid",  # <- [RegularGridTopology]
        "Sofa.Component.Topology.Container.Dynamic",  # <- [TriangleSetGeometryAlgorithms, TriangleSetTopologyContainer, TriangleSetTopologyModifier]
        "Sofa.Component.Topology.Mapping",  # <- [SubsetTopologicalMapping]
        "Sofa.Component.SolidMechanics.FEM.Elastic",  # <- [TriangularFEMForceFieldOptim]
    ]
    + SOLVER_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
    + MOTION_RESTRICTION_PLUGIN_LIST
)


@unique
class Color(Enum):
    BLUE = (0.75, 0.25)
    RED = (0.25, 0.75)
    GREEN = (0.75, 0.75)


class Tissue:
    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        size: Tuple[float, float] = (8.0, 10.0),
        grid_resolution: int = 50,
        visual_resolution: Optional[int] = 200,
        total_mass: float = 0.03,
        stiffness: float = 90.0,
        damping: float = 0.3,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        add_solver_func: Callable = add_solver,
        texture_file_path: Optional[Union[str, Path]] = None,
        show_debug_objects: bool = False,
        collision_group: int = 0,
        check_self_collision: bool = True,
    ) -> None:
        assert grid_resolution > 0, f"grid_resolution greater than 0 expected, got: {grid_resolution=}"
        if visual_resolution is not None:
            assert visual_resolution > 0, f"visual grid_resolution greater than 0 expected, got: {visual_resolution=}"
        else:
            visual_resolution = grid_resolution
        assert np.all(np.array(size) > 0), f"size greater than 0 expected, got: {size=}"

        self.grid_size = np.array(size)
        self.grid_resolution = grid_resolution
        self.visual_resolution = visual_resolution
        self.constraint_correction_type = ConstraintCorrectionType.LINEAR
        self.animation_loop_type = animation_loop_type
        self.parent_node = parent_node
        self.name = name
        self.show_debug_objects = show_debug_objects

        # Tissue node
        self.node = self.parent_node.addChild(name)

        # Debug axis
        if self.show_debug_objects:
            self.node.addObject("LineAxis", axis="xyz", size=300)

        # Grid topology
        self.topology_node = self.node.addChild("grid_toplogy")

        self.size = np.array(size)
        self.grid_min = np.array([-0.5 * size[0], -0.5 * size[1], 0.0])
        self.grid_max = np.array([0.5 * size[0], 0.5 * size[1], 0.0])

        self.grid_topology = self.topology_node.addObject(
            "RegularGridTopology",
            n=[grid_resolution, grid_resolution, 1],
            min=self.grid_min,
            max=self.grid_max,
        )
        visual_grid_topology = self.topology_node.addObject(
            "RegularGridTopology",
            name="visual_grid",
            n=[visual_resolution, visual_resolution, 1],
            min=self.grid_min,
            max=self.grid_max,
        )
        self.topology_node.init()

        # Add the solvers
        if animation_loop_type == AnimationLoopType.FREEMOTION:
            self.time_integration, self.linear_solver = add_solver_func(attached_to=self.node, linear_solver_type=LinearSolverType.SPARSELDL)
        else:
            self.time_integration, self.linear_solver = add_solver_func(attached_to=self.node)

        self.tissue_topology = self.node.addObject(
            "TriangleSetTopologyContainer",
            position=f"{self.grid_topology.getLinkPath()}.position",
            triangles=self.grid_topology.triangles.array().copy(),
        )

        self.mechanical_object = self.node.addObject("MechanicalObject", template="Vec3d")
        self.mass_node = self.node.addObject("UniformMass", totalMass=total_mass)

        self.tissue_spring_force_field_node = self.node.addObject(
            "MeshSpringForceField",
            trianglesStiffness=stiffness,
            trianglesDamping=damping,
        )

        if animation_loop_type == AnimationLoopType.FREEMOTION:
            self.constraint_correction = self.node.addObject(self.constraint_correction_type.value)

        if check_self_collision:
            self.collision_model = self.node.addObject(
                "TriangleCollisionModel",
                selfCollision=True,
                group=collision_group,
            )

        # Visual model
        self.visual_node = self.node.addChild("visual")

        self.visual_topology = self.visual_node.addObject(
            "TriangleSetTopologyContainer",
            position=f"{visual_grid_topology.getLinkPath()}.position",
            triangles=visual_grid_topology.triangles.array().copy(),
        )

        self.node.init()

        angle = np.rad2deg(np.arctan2(self.grid_size[0], self.grid_size[1]))

        # Bounding box for right gripper
        bb_top_right_corner = add_bounding_box(
            attached_to=self.node,
            min=(-2.5, -15.0, -1.5),
            max=(2.5, 5.0, 1.5),
            translation=(0.5 * self.grid_size[0], 0.5 * self.grid_size[1], 0.0),
            rotation=(0.0, 0.0, -angle),
            show_bounding_box=self.show_debug_objects,
            name="bb_top_right_corner",
        )
        bb_top_right_corner.init()
        self.top_right_corner_indices = [item for sublist in bb_top_right_corner.indices.toList() for item in sublist]

        # Bounding box for left gripper
        bb_top_left_corner = add_bounding_box(
            attached_to=self.node,
            min=(-2.5, -15.0, -1.5),
            max=(2.5, 5.0, 1.5),
            translation=(-0.5 * self.grid_size[0], 0.5 * self.grid_size[1], 0.0),
            rotation=(0.0, 0.0, angle),
            show_bounding_box=self.show_debug_objects,
            name="bb_top_left_corner",
        )
        bb_top_left_corner.init()
        self.top_left_corner_indices = [item for sublist in bb_top_left_corner.indices.toList() for item in sublist]
        if self.show_debug_objects:
            print(f"Top left indices: {self.top_left_corner_indices}")

        # Rigidify the tissue corners for grasping
        self.rigidify([self.top_left_corner_indices, self.top_right_corner_indices])
        self.rigidified_tissue_node.init()

        # Add ogl model with visual markers
        self.texture_file_path = texture_file_path
        texcoords = self.get_texture_coordinates()
        self.ogl_model = self.visual_node.addObject(
            "OglModel",
            texcoords=texcoords.tolist(),
            texturename=str(texture_file_path),
        )

        # Map the visual model
        self.visual_node.addObject("BarycentricMapping")

    def set_marker_positions(self, marker_positions: List[Tuple[float, float]], marker_colors: Union[List[Color], Color, None] = Color.RED, marker_radii: Union[List[float], float] = 3.0) -> None:
        """Sets the marker positions relative to the tissue grid.

        Args:
            marker_positions (List[Tuple[float, float]]): List of marker positions relative to the tissue grid [0, 1].
            marker_colors (Union[List[Color], Color, None], optional): List of marker colors. Defaults to Color.RED. If None, the markers will be invisible.
            marker_radii (Union[List[float], float], optional): List of marker radii. Defaults to 3.0.
        """

        if isinstance(marker_radii, float):
            marker_radii = [marker_radii] * len(marker_positions)
        else:
            assert len(marker_radii) == len(marker_positions)

        if isinstance(marker_colors, Color):
            marker_colors = [marker_colors] * len(marker_positions)
        elif isinstance(marker_colors, list):
            assert len(marker_colors) == len(marker_positions)

        relative_marker_positions = np.array(marker_positions)
        assert np.all(relative_marker_positions >= 0.0) and np.all(relative_marker_positions <= 1.0)

        # Transform the relative coordinates into absolute coordinates
        self._marker_positions = self.relative_to_absolute_position(relative_marker_positions)
        self._relative_marker_positions = relative_marker_positions

        # Indices of closest points on the visual grid
        array = self.visual_topology.position.array()
        self.marker_indices = [np.argmin(np.linalg.norm(array[:, :2] - pos, axis=1)) for pos in self._marker_positions]

        coloring_indices = [np.where(np.linalg.norm(array[:, :2] - pos, axis=1) <= marker_radii[index])[0] for index, pos in enumerate(self._marker_positions)]

        # Rectangular grid
        row_indices, column_indices = np.meshgrid(np.arange(self.visual_resolution), np.arange(self.visual_resolution), indexing="ij")

        # Texture coordinates for tissue
        tex_coords = np.column_stack([row_indices.ravel(), column_indices.ravel()]) / [2 * self.visual_resolution - 1, 2 * self.visual_resolution - 1]

        if marker_colors is not None:
            for num, index in enumerate(self.marker_indices):
                tex_coords[index] = marker_colors[num].value
            for num, indices in enumerate(coloring_indices):
                tex_coords[indices] = marker_colors[num].value

        self.ogl_model.texcoords = tex_coords

    def relative_to_absolute_position(self, relative_position: np.ndarray) -> np.ndarray:
        """Converts a relative position on the tissue to an absolute position in the world frame.

        Args:
            relative_position (np.ndarray): Relative position on the tissue.

        Returns:
            np.ndarray: Absolute position in the world frame.
        """

        min = -self.grid_size * 0.5
        max = self.grid_size * 0.5

        return relative_position * (max - min) + min

    def get_texture_coordinates(self) -> np.ndarray:
        """Returns texture coordinates.

        Returns:
            tex_coords (np.ndarray): Texture coordinates.
        """
        # Rectangular grid
        row_indices, column_indiecs = np.meshgrid(np.arange(self.visual_resolution), np.arange(self.visual_resolution), indexing="ij")

        # Texture coordinates for tissue
        tex_coords = np.column_stack([row_indices.ravel(), column_indiecs.ravel()]) / [2 * self.visual_resolution - 1, 2 * self.visual_resolution - 1]

        return tex_coords

    def get_marker_positions(self) -> Optional[np.ndarray]:
        """Returns the marker positions"""
        if hasattr(self, "marker_indices"):
            return self.ogl_model.position.array()[self.marker_indices]
        else:
            return None

    def rigidify(self, rigidification_indices: Union[List[int], List[List[int]]]) -> None:
        """Rididify parts of the tissue.
        Adapted from sofa_env/sofa_templates/deformable/rigidify.

        Args:
            rigidification_indices (Union[List[int], List[List[int]]]): A list of indices on the volume mesh of the tissue that should be rigidified. Can also be split into multiple subsets by passing a list of lists of indices.

        Returns:
            rigidified_tissue_node (Sofa.Core.Node): The sofa node that contains the sofa node of the new mixed material node.
        """
        self.rigidified_tissue_node = self.parent_node.addChild(f"rigidified_{self.name}")

        volume_mesh_positions = np.array([list(position) for position in self.grid_topology.position.array()], dtype=np.float64)
        volume_mesh_indices = np.array(list(range(self.grid_resolution * self.grid_resolution)), dtype=np.int64)
        if self.show_debug_objects:
            print(f"{volume_mesh_indices=}")

        # Make sure the indices are a list of lists
        if not isinstance(rigidification_indices[0], list):
            rigidification_indices = [rigidification_indices]

        subset_reference_poses = []
        flat_rigidification_indices = []
        subset_map = []

        angle = np.rad2deg(np.arctan2(self.grid_size[0], self.grid_size[1]))
        subset_orientations = [
            euler_angles_to_quaternion(np.array([-90.0, -angle, 0.0])),
            euler_angles_to_quaternion(np.array([-90.0, angle, 0.0])),
        ]
        # For each list (subset) of indices
        for subset_number in range(len(rigidification_indices)):
            subset_indices = rigidification_indices[subset_number]
            if not len(subset_indices) > 0:
                print(f"[WARNING]: Got an empty list of indices to rigidify for subset {subset_number}. Will skip this subset.")
                continue

            subset_positions = [volume_mesh_positions[index] for index in subset_indices]
            subset_reference_orientation_list = subset_orientations[subset_number].tolist()
            subset_reference_position_list = np.mean(subset_positions, axis=0).tolist()

            subset_reference_poses.append(subset_reference_position_list + subset_reference_orientation_list)
            flat_rigidification_indices.extend(subset_indices)
            subset_map.extend([subset_number] * len(subset_indices))

        deformable_indices = list(filter(lambda x: x not in flat_rigidification_indices, volume_mesh_indices))

        kd_tree = {index: [] for index in volume_mesh_indices}

        # Map the indices of the original volume mesh to [object 0 (deformable), index in object 0 (locally)]
        kd_tree.update({global_index: [0, local_index] for local_index, global_index in enumerate(deformable_indices)})

        # Map the indices of the original volume mesh to [object 1 (rigidified), index in object 1 (locally)]
        kd_tree.update({global_index: [1, local_index] for local_index, global_index in enumerate(flat_rigidification_indices)})

        # Flatten out the list to [object, local_index, object, local_index, ...] so that the index of the list corresponds to the global index in the volume mesh
        flat_index_pairs = [value for pair in kd_tree.values() for value in pair]

        deformable_parts = self.rigidified_tissue_node.addChild("deformable")
        deformable_mechanical_object = deformable_parts.addObject("MechanicalObject", template="Vec3d", position=[list(volume_mesh_positions[index] for index in deformable_indices)])

        rigid_parts = self.rigidified_tissue_node.addChild("rigid")
        rigid_parts.addObject("MechanicalObject", template="Rigid3d", position=subset_reference_poses)

        rigid_subsets = rigid_parts.addChild("rigid_subsets")
        rigid_mechanical_object = rigid_subsets.addObject("MechanicalObject", template="Vec3d", position=[list(volume_mesh_positions[index] for index in flat_rigidification_indices)])
        rigid_subsets.addObject("RigidMapping", globalToLocalCoords=True, rigidIndexPerPoint=subset_map)

        self.node.addObject(
            "SubsetMultiMapping",
            template="Vec3d,Vec3d",
            input=[deformable_mechanical_object.getLinkPath(), rigid_mechanical_object.getLinkPath()],
            output=self.mechanical_object.getLinkPath(),
            indexPairs=flat_index_pairs,
        )

        # Moves solvers and constraint correction from the original node to the new mixedmaterial node
        self.node.removeObject(self.time_integration)
        self.node.removeObject(self.linear_solver)
        self.rigidified_tissue_node.addObject(self.time_integration)
        self.rigidified_tissue_node.addObject(self.linear_solver)

        if self.animation_loop_type == AnimationLoopType.FREEMOTION:
            self.node.removeObject(self.constraint_correction)
            # A new precomputed constraint correction for the deformable parts
            deformable_parts.addObject(self.constraint_correction_type.value)
            # Uncoupled constraint correction for the rigidified parts
            rigid_parts.addObject(ConstraintCorrectionType.UNCOUPLED.value)

        # Add the original node as children to both deformable and rigid nodes
        rigid_subsets.addChild(self.node)
        deformable_parts.addChild(self.node)

        return self.rigidified_tissue_node

    def set_tissue_parameters(self, mass: float, stiffness: float, damping: float) -> None:
        self.mass_node.totalMass = mass
        self.tissue_spring_force_field_node.stiffness = stiffness
        self.tissue_spring_force_field_node.damping = damping

    def get_internal_force_magnitude(self) -> np.ndarray:
        """Get the sum of magnitudes of the internal forces applied to each vertex of the mesh"""
        return np.sum(np.linalg.norm(self.mechanical_object.force.array(), axis=1))

    def get_tissue_velocities(self) -> np.ndarray:
        """Get the velocities of the tissue vertices"""
        return self.mechanical_object.velocity.array()

    def seed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Creates a random number generator from a seed."""
        self.rng = np.random.default_rng(seed)
