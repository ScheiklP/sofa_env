import numpy as np
from enum import Enum
from typing import Tuple, Optional, Union

import Sofa.Core

from sofa_env.utils.math_helper import euler_to_rotation_matrix, point_rotation_by_quaternion, rotation_matrix_to_quaternion

TOPOLOGY_PLUGIN_LIST = [
    "Sofa.Component.Topology.Container.Dynamic",  # [HexahedronSetTopologyContainer, TetrahedronSetTopologyContainer, HexahedronSetTopologyModifier, TetrahedronSetTopologyModifier]
    "Sofa.Component.Topology.Container.Grid",  # [RegularGridTopology]
    "Sofa.Component.Topology.Mapping",  # [Hexa2TetraTopologicalMapping]
]


class TopologyTypes(Enum):
    TETRA = "Tetrahedron"
    HEXA = "Hexahedron"


def add_tetrahedral_topology(attached_to: Sofa.Core.Node, volume_mesh_loader: Sofa.Core.Object) -> Sofa.Core.Object:
    """Adds the relevant sofa objects for a topology based on tetrahedra."""

    topology_container = attached_to.addObject("TetrahedronSetTopologyContainer", name="Topology", src=volume_mesh_loader.getLinkPath())
    attached_to.addObject("TetrahedronSetTopologyModifier", name="TopologyModifier")
    attached_to.addObject("TetrahedronSetTopologyAlgorithms", name="TopologyAlgorithms", template="Vec3d")
    attached_to.addObject("TetrahedronSetGeometryAlgorithms", name="GeometryAlgorithms", template="Vec3d")
    return topology_container


def add_hexahedral_topology(attached_to: Sofa.Core.Node, volume_mesh_loader: Sofa.Core.Object) -> Sofa.Core.Object:
    """Adds the relevant sofa objects for a topology based on hexahedra."""

    topology_container = attached_to.addObject("HexahedronSetTopologyContainer", name="Topology", src=volume_mesh_loader.getLinkPath())
    attached_to.addObject("HexahedronSetTopologyModifier", name="TopologyModifier")
    attached_to.addObject("HexahedronSetTopologyAlgorithms", name="TopologyAlgorithms", template="Vec3d")
    attached_to.addObject("HexahedronSetGeometryAlgorithms", name="GeometryAlgorithms", template="Vec3d")
    return topology_container


def hollow_cylinder_hexahedral_topology_data(radius_inner: float, radius_outer: float, height: float, num_radius: int, num_phi: int, num_z: int, translation: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a hexahedral topology for a hollow cylinder with inner and outer radii.

    Args:
        radius_inner (float): The inner radius of the hollow cylinder.
        radius_outer (float): The outer radius of the hollow cylinder.
        height (float): The height of the hollow cylinder.
        num_radius (int): Number of points along the radius -> n-1 hexahedra along the radius.
        num_phi (int): Number of points along angle -> n hexahedra around the angle.
        num_z (int): Number of points along the height -> n-1 hexahedra.
        translation (Optional[np.ndarray]): Translation of the hollow cylinder.

    Returns:
        points (List): A list of [x, y, z] coordinates of points.
        hexahedra (List): The list of hexahedra described with 8 indices each corresponding to the points.
    """

    radii = np.linspace(radius_inner, radius_outer, num_radius)
    phis = np.linspace(0, 2 * np.pi, num_phi + 1)[:-1]
    zs = np.linspace(0, height, num_z)

    index_array = np.empty((num_radius, num_phi, num_z), dtype=np.uint64)

    points = []
    i = 0
    for index_z, z in enumerate(zs):
        for index_radius, radius in enumerate(radii):
            for index_phi, phi in enumerate(phis):
                points.append(np.asarray([radius * np.cos(phi), radius * np.sin(phi), z]))
                index_array[index_radius, index_phi, index_z] = i
                i += 1

    points = np.asarray(points)

    hexahedra = []
    for z in range(num_z - 1):
        for r in range(num_radius - 1):
            for phi in range(num_phi):
                phi_upper = (phi + 1) % num_phi
                hexahedron = (
                    index_array[r, phi, z],
                    index_array[r, phi_upper, z],
                    index_array[r, phi_upper, z + 1],
                    index_array[r, phi, z + 1],
                    index_array[r + 1, phi, z],
                    index_array[r + 1, phi_upper, z],
                    index_array[r + 1, phi_upper, z + 1],
                    index_array[r + 1, phi, z + 1],
                )
                hexahedra.append(hexahedron)

    hexahedra = np.asarray(hexahedra)

    if translation is not None:
        points += translation

    return points, hexahedra


def cylinder_shell_triangle_topology_data(
    radius: float,
    height: float,
    num_phi: int,
    num_z: int,
    start_position: Optional[Union[np.ndarray, Tuple]] = None,
    euler_angle_rotation: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a triangle topology for the shell of a cylinder without top and bottom.

    Args:
        radius (float): The radius of the cylinder.
        height (float): The height of the cylinder.
        num_phi (int): Number of points along angle.
        num_z (int): Number of points along the height.
        start_position (Optional[Union[np.ndarray, Tuple]]): Cartesian position of the center of the cylinder's bottom.
        euler_angle_rotation (Optional[np.ndarray]): Optional XYZ euler angles to rotate the shell.

    Returns:
        points (np.ndarray): A Nx3 array of [x, y, z] coordinates of points.
        triangles (np.ndarray): A Nx3 array of triangles described with 3 indices each corresponding to the points.
    """

    phis = np.linspace(0, 2 * np.pi, num_phi + 1)[:-1]
    zs = np.linspace(0, height, num_z)

    index_array = np.empty((num_phi, num_z), dtype=np.uint64)

    points = []
    i = 0
    for index_z, z in enumerate(zs):
        for index_phi, phi in enumerate(phis):
            points.append([radius * np.cos(phi), radius * np.sin(phi), z])
            index_array[index_phi, index_z] = i
            i += 1

    points = np.asarray(points)

    triangles = []
    for z in range(num_z - 1):
        for phi in range(num_phi):
            phi_upper = (phi + 1) % num_phi
            forward_triangle = (
                index_array[phi, z],
                index_array[phi, z + 1],
                index_array[phi_upper, z],
            )
            backward_triangle = (
                index_array[phi, z],
                index_array[phi - 1, z + 1],
                index_array[phi, z + 1],
            )
            triangles.append(forward_triangle)
            triangles.append(backward_triangle)

    triangles = np.asarray(triangles)

    if euler_angle_rotation is not None:
        if not len(euler_angle_rotation) == 3:
            raise ValueError(f"Expected 3 euler angles for XYZ euler rotation. Received {len(euler_angle_rotation)} as {euler_angle_rotation=}")
        transformation_quaternion = rotation_matrix_to_quaternion(euler_to_rotation_matrix(euler_angle_rotation))
        points = np.asarray([point_rotation_by_quaternion(point, transformation_quaternion) for point in points])

    if start_position is not None:
        points += start_position

    return points, triangles


def create_initialized_grid(
    attached_to: Sofa.Core.Node,
    name: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    num_x: int,
    num_y: int,
    num_z: int,
) -> Tuple[Sofa.Core.Object, Sofa.Core.Object, Sofa.Core.Object]:
    """Creates a grid and returns topology containers for points, hexahedra and tetrahedra.

    Args:
        attached_to (Sofa.Core.Node): The node to attach the grid to.
        name (str): The name of the grid.
        xmin (float): The minimum x coordinate.
        xmax (float): The maximum x coordinate.
        ymin (float): The minimum y coordinate.
        ymax (float): The maximum y coordinate.
        zmin (float): The minimum z coordinate.
        zmax (float): The maximum z coordinate.
        num_x (int): The number of points in x direction.
        num_y (int): The number of points in y direction.
        num_z (int): The number of points in z direction.

    Returns:
        grid_topology (Sofa.Core.Object): The grid container.
        hexahedra_topology (Sofa.Core.Object): The hexahedra container.
        tetrahedra_topology (Sofa.Core.Object): The tetrahedra container.
    """

    topology_node = attached_to.addChild(name)
    grid_node = topology_node.addChild("grid")
    grid_topology = grid_node.addObject(
        "RegularGridTopology",
        nx=num_x,
        ny=num_y,
        nz=num_z,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
    )
    grid_node.init()

    hexahedra_node = topology_node.addChild("hexahedra")
    grid_positions = grid_topology.position.array()
    grid_hexahedra = grid_topology.hexahedra.array()
    hexahedron_topology = hexahedra_node.addObject("HexahedronSetTopologyContainer", hexahedra=grid_hexahedra.copy(), position=grid_positions.copy())
    hexahedra_node.addObject("HexahedronSetTopologyModifier")

    tetrahedra_node = hexahedra_node.addChild("tetrahedra")
    tetrahedron_topology = tetrahedra_node.addObject("TetrahedronSetTopologyContainer")
    tetrahedra_node.addObject("TetrahedronSetTopologyModifier")
    tetrahedra_node.addObject("Hexa2TetraTopologicalMapping")

    topology_node.init()

    return grid_topology, hexahedron_topology, tetrahedron_topology
