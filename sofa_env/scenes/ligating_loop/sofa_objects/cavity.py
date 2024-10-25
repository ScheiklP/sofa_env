import numpy as np

from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict

import Sofa
import Sofa.Core

from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.solver import add_solver, OdeSolverType, LinearSolverType, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.topology import hollow_cylinder_hexahedral_topology_data
from sofa_env.utils.math_helper import euler_to_rotation_matrix, is_inside_mesh, point_rotation_by_quaternion, rotation_matrix_to_quaternion


CAVITY_PLUGIN_LIST = SOLVER_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST


class Cavity:
    """Cavity model of LigatingLoopEnv and ThreadInHoleEnv.

    Args:
        parent_node (Sofa.Core.Node): Parent node of the SOFA object.
        name (str): Name of the object.
        total_mass (float): Total mass of the cavity.
        young_modulus (float): Young modulus of the cavity material.
        poisson_ratio (float): Poisson ratio of the cavity material.
        inner_radius (float): Inner radius of the cavity.
        outer_radius (float): Outer radius of the cavity.
        height (float): Height of the cavity cylinder.
        discretization_radius (int): How many points are used to discretize the cavity along the radius?
        discretization_angle (int): How many points are used to discretize the circumference of the cavity?
        discretization_height (int): How many points are used to discretize the cavity along it's axis?
        ode_solver_rayleigh_mass (float): Rayleigh mass of the ode solver.
        ode_solver_rayleigh_stiffness (float): Rayleigh stiffness of the ode solver.
        translation (Optional[np.ndarray]): XYZ offset of the cavity.
        fixed_lenght (Optional[float]): Optionally adding a fixed constraint to ``fixed_lenght`` of the cavity.
        animation_loop_type (AnimationLoopType): The scenes animation loop in order to correctly add constraint correction objects.
        texture_path (Optional[Union[str, Path]]): Optional texture file to add to the ``OglModel`` of the cavity.
        color (Optional[Tuple[float, float, float]]): Optional RGB color of the cavity.
        show_object (bool): Whether to render the points of the cavity.
        show_object_scale (float): Parameter to change the render size, if ``show_object=True``.
        collision_group (Optional[int]): The group for which collisions with this object should be ignored.
        show_shell (bool): Whether to render the hull of the cavity.
        create_shell (bool): Whether to create a mechanical object that holds the positions of the hull of the cavity.
        check_self_collision (bool): Whether the collision pipeline should consider collisions of the cavity with itself.
        position_reset_noise (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Limits to uniformly sample noise that is added to the cavities' initial position.
        rotation_reset_noise (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Limits to uniformly sample noise that is added to the cavities' initial orientation in XYZ Euler angles.
        band_width (float): Width of the marking in millimeters.
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        total_mass: float,
        young_modulus: float,
        poisson_ratio: float,
        inner_radius: float,
        outer_radius: float,
        height: float,
        discretization_radius: int,
        discretization_angle: int,
        discretization_height: int,
        ode_solver_rayleigh_mass: float = 0.1,
        ode_solver_rayleigh_stiffness: float = 0.1,
        translation: Optional[np.ndarray] = None,
        fixed_lenght: Optional[float] = None,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        texture_path: Optional[Union[str, Path]] = None,
        color: Optional[Tuple[float, float, float]] = None,
        show_object: bool = False,
        show_object_scale: float = 5.0,
        collision_group: Optional[int] = None,
        show_shell: bool = False,
        create_shell: bool = True,
        check_self_collision: bool = True,
        position_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        rotation_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        band_width: float = 4.0,
    ) -> None:
        self.node = parent_node.addChild(name)
        self.height = height
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.translation = translation
        self.discretization_radius = discretization_radius
        self.discretization_angle = discretization_angle
        self.discretization_height = discretization_height

        self.position_reset_noise = position_reset_noise
        self.rotation_reset_noise = rotation_reset_noise

        self.colored_collision_model_indices = []
        self.colored_fem_indices = []
        self.color_band_start = 30.0
        self.color_band_end = self.color_band_start + band_width

        # Create the general topology of the cylinder with hexahedra
        points, hexahedra = hollow_cylinder_hexahedral_topology_data(
            radius_inner=inner_radius,
            radius_outer=outer_radius,
            height=height,
            num_radius=discretization_radius,
            num_phi=discretization_angle,
            num_z=discretization_height,
            translation=translation,
        )
        self.initial_positions = points

        # Quick computation to figure out which points on the model belong to the outside of the cavity
        if translation is None:
            translation = np.zeros(3)
        i = 0
        # Points on the outside surface of the cylinder
        self.outer_indices = []
        # The outer ring of the cavity at 0 height
        self.start_ring_indices = []
        # The outher ring of the cavity at height==height
        self.end_ring_indices = []
        for point_height in np.linspace(0, height, discretization_height):
            for radius in np.linspace(inner_radius, outer_radius, discretization_radius):
                for _ in np.linspace(0, 2 * np.pi, discretization_angle + 1)[:-1]:
                    if radius == outer_radius:
                        self.outer_indices.append(i)
                        if point_height == translation[2]:
                            self.start_ring_indices.append(i)
                        elif point_height == translation[2] + height:
                            self.end_ring_indices.append(i)
                    i += 1

        # Create Hexahedra
        topology_creation_node = parent_node.addChild("topology")
        hexahedra_node = topology_creation_node.addChild("hexa")
        hexahedra_node.addObject("HexahedronSetTopologyContainer", position=points, hexahedra=hexahedra)
        hexahedra_node.addObject("HexahedronSetTopologyModifier")

        # Subdivide them into Tetrahedra for FEM
        tetrahedra_node = hexahedra_node.addChild("tetra")
        tetrahedra_node.addObject("TetrahedronSetTopologyContainer")
        tetrahedra_node.addObject("TetrahedronSetTopologyModifier")
        tetrahedra_node.addObject("Hexa2TetraTopologicalMapping")

        # Determine the surface triangles for collision
        triangle_node = tetrahedra_node.addChild("triangle")
        triangle_node.addObject("TriangleSetTopologyContainer")
        triangle_node.addObject("TriangleSetTopologyModifier")
        triangle_node.addObject("Tetra2TriangleTopologicalMapping")

        topology_creation_node.init()

        # Create a set of triangles that encloses the hollow cavity as a cylinder
        # This set has two additional points, in the middle of the openings at start and end of the cavity
        shell_triangles = []
        for triangle in triangle_node.TriangleSetTopologyContainer.triangles.array():
            if np.all(np.in1d(triangle, self.outer_indices)):
                shell_triangles.append(triangle)

        shell_points = points.copy()
        center_point_start = np.mean(points[self.start_ring_indices], axis=0)
        center_point_end = np.mean(points[self.end_ring_indices], axis=0)
        shell_points = np.concatenate((shell_points, center_point_start[None, :], center_point_end[None, :]))
        start_point_index = len(points)
        end_point_index = len(points) + 1

        start_triangles = []
        for index in self.start_ring_indices[:-1]:
            triangle = [start_point_index, index + 1, index]
            start_triangles.append(triangle)
        start_triangles.append([start_point_index, self.start_ring_indices[0], self.start_ring_indices[-1]])

        end_triangles = []
        for index in self.end_ring_indices[:-1]:
            triangle = [index, index + 1, end_point_index]
            end_triangles.append(triangle)
        end_triangles.append([self.end_ring_indices[-1], self.end_ring_indices[0], end_point_index])

        shell_triangles.extend(start_triangles)
        shell_triangles.extend(end_triangles)

        self.shell_triangles = shell_triangles

        # Add solvers to the node
        self.time_integration, self.linear_solver = add_solver(
            self.node,
            ode_solver_type=OdeSolverType.IMPLICITEULER,
            linear_solver_type=LinearSolverType.SPARSELDL if animation_loop_type == AnimationLoopType.FREEMOTION else LinearSolverType.CG,
            linear_solver_kwargs={"template": "CompressedRowSparseMatrixd"} if animation_loop_type == AnimationLoopType.FREEMOTION else None,
            ode_solver_rayleigh_mass=ode_solver_rayleigh_mass,
            ode_solver_rayleigh_stiffness=ode_solver_rayleigh_stiffness,
        )

        # FEM model on Tetrahedra
        self.node.addObject("TetrahedronSetTopologyContainer", tetrahedra=tetrahedra_node.TetrahedronSetTopologyContainer.tetrahedra.array(), position=points)
        self.node.addObject("TetrahedronSetTopologyModifier")
        self.mechanical_object = self.node.addObject("MechanicalObject", showObject=show_object, showObjectScale=show_object_scale)
        # TODO: Currently unstable
        # self.node.addObject("FastTetrahedralCorotationalForceField", youngModulus=young_modulus, poissonRatio=poisson_ratio)
        self.node.addObject("TetrahedralCorotationalFEMForceField", youngModulus=young_modulus, poissonRatio=poisson_ratio)
        self.node.addObject("UniformMass", totalMass=total_mass)

        # Fixed constraint at the end
        if fixed_lenght is not None:
            box_limits = np.array([[-outer_radius * 1.1, -outer_radius * 1.1, -1], [outer_radius * 1.1, outer_radius * 1.1, fixed_lenght]])
            if translation is not None:
                box_limits += translation

            bounding_box = self.node.addObject("BoxROI", box=box_limits.ravel())
            self.node.addObject("FixedProjectiveConstraint", indices=f"{bounding_box.getLinkPath()}.indices")

        # Constraint correction
        if animation_loop_type == AnimationLoopType.FREEMOTION:
            self.node.addObject("LinearSolverConstraintCorrection")

        # Retrieve the surface of the cylinder
        surface_node = self.node.addChild("surface")
        self.triangle_collision_topology = surface_node.addObject("TriangleSetTopologyContainer", position=points)
        surface_node.addObject("TriangleSetTopologyModifier")
        surface_node.addObject("TriangleSetGeometryAlgorithms")
        surface_node.addObject("Tetra2TriangleTopologicalMapping")

        # Collision model
        if collision_group is not None:
            collision_model_kwargs = {"group": collision_group}
        else:
            collision_model_kwargs = {}
        self.triangle_collision_model = surface_node.addObject("TriangleCollisionModel", selfCollision=check_self_collision, **collision_model_kwargs)

        # Visual model
        visual_node = surface_node.addChild("ogl")
        ogl_model_kwargs = {"texturename": str(texture_path)} if texture_path is not None else ({"color": (0.0, 1.0, 1.0)} if color is not None else {})
        self.ogl_model = visual_node.addObject("OglModel", **ogl_model_kwargs)
        visual_node.addObject("IdentityMapping")

        self.create_shell = create_shell
        if self.create_shell:
            # Create a node that holds the cylinder points + the points that are in the middle of start end end opening -> mapped barycentrically to the FEM positions
            # By indexing shell_mechanical_object.position.array()[shell_triangles] we get a set of triangles, that encloses the hollow cylinder as a cylinder shell.
            shell_triangle_node = triangle_node.addChild("shell")
            self.shell_mechanical_object = shell_triangle_node.addObject("MechanicalObject", template="Vec3d", position=shell_points, showObject=show_shell, showObjectScale=show_object_scale, showColor=[1.0, 0.0, 1.0])
            shell_triangle_node.addObject("BarycentricMapping", input=self.node.MechanicalObject.getLinkPath(), output=shell_triangle_node.MechanicalObject.getLinkPath())

            # Optional visualization of the shell
            if show_shell:
                visual_node = shell_triangle_node.addChild("visual")
                visual_node.addObject(
                    "OglModel",
                    triangles=shell_triangles,
                    position=shell_points,
                    alphaBlend=False,
                    # Name of the material, Diffuse Enabled R G B Alpha, Ambient Enabled R G B Alpha, ...
                    material="Transparent Diffuse 1 1 0 1 0.75 Ambient 0 1 1 1 1 Specular 1 0 0 1 1 Emissive 0 1 0 0 1 Shininess 1 100",
                )
                visual_node.addObject("IdentityMapping")

    def get_state(self) -> np.ndarray:
        return self.mechanical_object.position.array()

    def get_center_of_opening_position(self) -> np.ndarray:
        if self.create_shell:
            return self.shell_mechanical_object.position.array()[-1]
        else:
            return np.mean(self.mechanical_object.position.array()[self.end_ring_indices][:, :3], axis=0)

    def color_band(self, start: float, end: float, texture_coords_band: Tuple[float, float] = (0.75, 0.75), texture_coords_rest: Tuple[float, float] = (0.25, 0.25)) -> None:
        point_indices_to_color = [i for i, point in enumerate(self.initial_positions) if point[2] > start and point[2] < end]

        # Filter the indices, that belong to the outside of the model
        selected_outer_indices = [index for index in point_indices_to_color if index in self.outer_indices]

        # Through the topology of the collision model, look up the indices of the collision triangles that are colored
        # These will be the triangles that should be in collision, when the loop is closed around the cavity
        self.colored_collision_model_indices = []
        self.colored_fem_indices = selected_outer_indices
        for triangle_index, triangle in enumerate(self.triangle_collision_topology.triangles.array()):
            if triangle[0] in selected_outer_indices:
                self.colored_collision_model_indices.append(triangle_index)

        # Color the selected point indices
        with self.ogl_model.texcoords.writeable() as texcoords:
            texcoords[:] = np.array(texture_coords_rest)
            texcoords[selected_outer_indices] = np.array(texture_coords_band)

    def get_subsampled_indices(self, discretization_radius: int, discretization_angle: int, discretization_height: int) -> List[int]:
        if not self._valid_subsampling(discretization_height=discretization_height, discretization_angle=discretization_angle, discretization_radius=discretization_radius):
            raise ValueError(f"Cannot subsample the cavity with the given discretization value. Radius {discretization_radius} of {self.discretization_radius}\nHeight {discretization_height} of {self.discretization_height}\nAngle {discretization_angle} of {self.discretization_angle}")

        subsampled_points, _ = hollow_cylinder_hexahedral_topology_data(
            radius_inner=self.inner_radius,
            radius_outer=self.outer_radius,
            height=self.height,
            num_radius=discretization_radius,
            num_phi=discretization_angle,
            num_z=discretization_height,
            translation=self.translation,
        )

        indices = []
        # TODO Is there a numpy way to do that?
        # np.argmin(np.linalg.norm(np.repeat(b[:, :, None], 3, axis=2) - a, axis=2), axis=1)
        for point in subsampled_points:
            index_closest_point = np.argmin(np.linalg.norm(self.initial_positions - point, axis=1))
            indices.append(index_closest_point)

        return indices

    def get_subsampled_indices_on_band(self, discretization_radius: int, discretization_angle: int, discretization_height: int) -> List[int]:
        if not self._valid_subsampling(discretization_height=discretization_height, discretization_angle=discretization_angle, discretization_radius=discretization_radius):
            raise ValueError(f"Cannot subsample the cavity with the given discretization value. Radius {discretization_radius} of {self.discretization_radius}\nHeight {discretization_height} of {self.discretization_height}\nAngle {discretization_angle} of {self.discretization_angle}")

        subsampled_points, _ = hollow_cylinder_hexahedral_topology_data(
            radius_inner=self.inner_radius,
            radius_outer=self.outer_radius,
            height=self.color_band_end - self.color_band_start,
            num_radius=discretization_radius,
            num_phi=discretization_angle,
            num_z=discretization_height,
            translation=self.translation + np.array([0, 0, self.color_band_start]),
        )

        original_positions = self.initial_positions[self.colored_fem_indices]

        indices = []
        for point in subsampled_points:
            index_closest_point = self.colored_fem_indices[np.argmin(np.linalg.norm(original_positions - point, axis=1))]
            indices.append(index_closest_point)

        return indices

    def _valid_subsampling(self, discretization_radius: int, discretization_height: int, discretization_angle: int) -> bool:
        valid = True
        if discretization_angle < 1 or discretization_height < 1 or discretization_radius < 1:
            valid = False
        if discretization_radius > self.discretization_radius or discretization_angle > self.discretization_angle or discretization_height > self.discretization_height:
            valid = False

        return valid

    def is_in_cavity(self, position: np.ndarray) -> bool:
        if self.create_shell:
            triangles = self.shell_mechanical_object.position.array()[self.shell_triangles]
            return bool(is_inside_mesh(triangles, position[None, :]))
        else:
            raise ValueError(f"Cannot compute if no shell was created. {self.create_shell=}")

    def are_in_cavity(self, positions: np.ndarray) -> np.ndarray:
        if self.create_shell:
            triangles = self.shell_mechanical_object.position.array()[self.shell_triangles]
            return is_inside_mesh(triangles, positions)
        else:
            raise ValueError(f"Cannot compute if no shell was created. {self.create_shell=}")

    def set_state(self, state: np.ndarray) -> None:
        with self.mechanical_object.position.writeable() as sofa_state:
            sofa_state[:] = state

    def reset_cavity(self) -> None:
        if self.rotation_reset_noise is not None:
            if isinstance(self.rotation_reset_noise, np.ndarray):
                # Uniformly sample from -noise to +noise and rotate the initial points
                rotation_quaternion = rotation_matrix_to_quaternion(euler_to_rotation_matrix(self.rng.uniform(-self.rotation_reset_noise, self.rotation_reset_noise)))
            elif isinstance(self.rotation_reset_noise, dict):
                # Uniformly sample from low to high and and rotate the initial points
                rotation_quaternion = rotation_matrix_to_quaternion(euler_to_rotation_matrix(self.rng.uniform(self.rotation_reset_noise["low"], self.rotation_reset_noise["high"])))
            else:
                raise TypeError("Please pass the rotation_reset_noise as a numpy array or a dictionary with 'low' and 'high' keys of Euler Angles around XYZ.")
        else:
            rotation_quaternion = np.array((0.0, 0.0, 0.0, 1.0))

        if self.position_reset_noise is not None:
            if isinstance(self.position_reset_noise, np.ndarray):
                # Uniformly sample from -noise to +noise and rotate the initial points
                translation = self.rng.uniform(-self.position_reset_noise, self.position_reset_noise)
            elif isinstance(self.position_reset_noise, dict):
                # Uniformly sample from low to high and and rotate the initial points
                translation = self.rng.uniform(self.position_reset_noise["low"], self.position_reset_noise["high"])
            else:
                raise TypeError("Please pass the position_reset_noise as a numpy array or a dictionary with 'low' and 'high' keys for translations of XYZ.")
        else:
            translation = np.zeros(3)

        new_positions = self.initial_positions.copy()
        for index, point in enumerate(self.initial_positions):
            new_positions[index] = point_rotation_by_quaternion(point, rotation_quaternion) + translation

        self.set_state(new_positions)

    def seed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Creates a random number generator from a seed."""
        self.rng = np.random.default_rng(seed)

    def get_tissue_velocities(self) -> np.ndarray:
        """Get the velocities of the tissue vertices"""
        return self.mechanical_object.velocity.array()
