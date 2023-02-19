from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

import Sofa
import Sofa.Core

import sofa_env.utils.math_helper as mh

from sofa_env.sofa_templates.motion_restriction import MOTION_RESTRICTION_PLUGIN_LIST, add_fixed_constraint_in_bounding_box, add_fixed_constraint_to_indices
from sofa_env.sofa_templates.scene_header import SCENE_HEADER_PLUGIN_LIST, AnimationLoopType
from sofa_env.sofa_templates.solver import SOLVER_PLUGIN_LIST, ConstraintCorrectionType, LinearSolverType, add_solver

from sofa_env.scenes.precision_cutting.sofa_objects.cloth_cut import linear_cut
from sofa_env.scenes.precision_cutting.sofa_objects.grid_path_projection import GridPathProjection, RunningAxis

CLOTH_PLUGIN_LIST = (
    [
        "Sofa.Component.Visual",  # <- [LineAxis]
        "Sofa.Component.Engine.Transform",  # <- [TransformEngine]
        "Sofa.Component.Topology.Container.Grid",  # <- [RegularGridTopology]
        "Sofa.Component.Topology.Container.Dynamic",  # <- [TriangleSetGeometryAlgorithms, TriangleSetTopologyContainer, TriangleSetTopologyModifier]
        "Sofa.Component.Topology.Mapping",  # <- [SubsetTopologicalMapping]
    ]
    + SOLVER_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
    + MOTION_RESTRICTION_PLUGIN_LIST
)


class Cloth(Sofa.Core.Controller):
    """Cloth

    This class adds a deformable/cuttable cloth based on a ``"RegularGridTopology"`` to the scene.

    Args:
        parent_node (Sofa.Core.Node): The SOFA node to which the cloth subtree is added.
        name (str): node name.
        position (Tuple[float, float, float]): .
        orientation (Tuple[float, float, float]): Euler XYZ.
        size (Tuple[float, float]): grid size along (x, y) axes.
        resolution (int): grid resolution, i.e. grid points per unit.
        visual_resolution (int): grid resolution, i.e. grid points per unit.
        total_mass (float): .
        damping (float): .
        stiffness (float): .
        animation_loop_type (AnimationLoopType): .
        add_solver_func (Callable): .
        texture_file_path (Optional[Union[str, Path]]): .
        cutting_path (GridPathProjection): .
        cut_stroke_radius (float): .
        show_debug_objects (bool): .
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        position: Tuple[float, float, float] = (0, 0, 0),
        orientation: Tuple[float, float, float] = (0, 0, 0),  # euler XYZ
        size: Tuple[float, float] = (150.0, 150.0),
        grid_resolution: int = 100,
        visual_resolution: Optional[int] = 100,
        total_mass: float = 0.03,
        stiffness: float = 90.0,
        damping: float = 0.3,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        add_solver_func: Callable = add_solver,
        texture_file_path: Optional[Union[str, Path]] = None,
        cutting_path: GridPathProjection = linear_cut(),
        cut_stroke_radius: float = 1.0,
        show_debug_objects: bool = False,
    ) -> None:
        visual_grid_resolution = visual_resolution if visual_resolution is not None else grid_resolution
        assert grid_resolution > 0
        assert visual_grid_resolution > 0
        assert np.all(np.array(size) > 0)

        # FIXME: Mapping between RegularGridTopologies of different resolutions does not work with cutting/CarvingTool
        # hence we require the resolution of the visual grid to be identical to that of the collision grid
        assert visual_grid_resolution == grid_resolution

        Sofa.Core.Controller.__init__(self)
        self.name = f"{name}_controller"

        self.grid_size = np.array(size)
        self.grid_resolution = grid_resolution

        # cloth/grid initial pose
        # TODO: trafo = mh.euler_angles_to_homogeneous_transform(np.array(orientation), np.array(position))
        trafo = np.row_stack(
            [
                np.column_stack([mh.euler_to_rotation_matrix(np.array(orientation)), np.array(position)]),
                np.identity(4)[-1],
            ]
        )
        cloth_to_global = lambda v: trafo[:3, -1] + (trafo[:3, :3] @ v.reshape(-1, 3, 1)).reshape(-1, 3)
        self.pose = mh.homogeneous_transform_to_pose(trafo)
        self.pose.flags.writeable = False
        cloth_pos, cloth_quat = self.pose[:3], self.pose[3:]

        self.constraint_correction_type = ConstraintCorrectionType.LINEAR
        self.animation_loop_type = animation_loop_type

        self.parent_node = parent_node
        self.node = self.parent_node.addChild(name)

        # debug
        self.show_debug_objects = show_debug_objects
        if self.show_debug_objects:
            self.node.addObject("LineAxis", axis="xyz", size=300)

        # grids
        topology_node = self.node.addChild("grid_toplogy")
        # define grid bounds
        grid_bbox_min = np.array([-0.5 * size[0], -0.5 * size[1], 0.0])
        grid_bbox_max = np.array([0.5 * size[0], 0.5 * size[1], 0.0])

        grid_topology = topology_node.addObject(
            "RegularGridTopology",
            n=[grid_resolution, grid_resolution, 1],
            min=grid_bbox_min,
            max=grid_bbox_max,
        )

        # Sofa.Core.BaseMeshTopology
        visual_grid_topology = topology_node.addObject(
            "RegularGridTopology",
            name="visual_grid",
            n=[visual_grid_resolution, visual_grid_resolution, 1],
            min=grid_bbox_min,
            max=grid_bbox_max,
        )

        topology_node.init()
        self.initial_cloth_triangles = grid_topology.triangles.array().copy()
        self._cutting_path = cutting_path
        self.initial_triangles_on_path, self.centerline_indices = cutting_path.get_triangles_on_path(
            grid_size=size,
            grid_resolution=(grid_resolution, grid_resolution),
            num_samples=2 * grid_resolution,
            stroke_radius=cut_stroke_radius,
            grid_vertices=grid_topology.position.array(),
            grid_triangles=grid_topology.triangles.array(),
        )

        # Add the solvers
        if animation_loop_type == AnimationLoopType.FREEMOTION:
            self.time_integration, self.linear_solver = add_solver_func(attached_to=self.node, linear_solver_type=LinearSolverType.SPARSELDL)
        else:
            self.time_integration, self.linear_solver = add_solver_func(attached_to=self.node)

        # Cloth topology
        cloth_transform = self.node.addObject(
            "TransformEngine",
            name="cloth_transform",
            template="Vec3d",
            translation=cloth_pos,
            quaternion=cloth_quat,
            input_position=f"{grid_topology.getLinkPath()}.position",
        )
        self.cloth_topology = self.node.addObject(
            "TriangleSetTopologyContainer",
            position=f"{cloth_transform.getLinkPath()}.output_position",
            triangles=grid_topology.triangles.array().copy(),
        )
        self.node.addObject("TriangleSetTopologyModifier")
        self.node.addObject("TriangleSetGeometryAlgorithms", template="Vec3d")

        self.mechanical_object = self.node.addObject("MechanicalObject")
        self.node.addObject("UniformMass", totalMass=total_mass)
        self.node.addObject(
            "MeshSpringForceField",
            trianglesStiffness=stiffness,
            trianglesDamping=damping,
        )
        # If using the FreeMotionAnimationLoop add a constraint correction object
        if animation_loop_type == AnimationLoopType.FREEMOTION:
            self.constraint_correction = self.node.addObject(self.constraint_correction_type.value)
        self.node.addObject("PointCollisionModel", tags="CarvingSurface")

        # Add the visual model of the cloth
        self._add_visual_model(
            translation=cloth_pos,
            quaternion=cloth_quat,
            grid_resolution=grid_resolution,
            visual_grid_resolution=visual_resolution,
            visual_grid_topology=visual_grid_topology,
            texture_file_path=texture_file_path,
        )

        # Simulate clamps that hold the corners of the cloth through springs
        self._add_clamps_to_cloth(
            grid_resolution=grid_resolution,
        )

        # fix the upper edge of the complete cloth
        self._fix_cloth_edge(grid_size=size, orientation=orientation, cloth_to_global=cloth_to_global)

        #  keep track of the triangles on the cutting path
        self._add_cloth_subset_topology_on_path(cloth_transform=cloth_transform)

        #  keep track of the triangles off the cutting path
        self._add_cloth_subset_topology_off_path(cloth_transform=cloth_transform)

        self._topology_buffer = {}
        self.update_topology_buffer()

    def _add_cloth_subset_topology_on_path(self, cloth_transform: Sofa.Core.Object):
        """In order to keep track of the triangles on the cutting path we store the respective vertices in a separate TopologyContainer linked to the cloth via a SubsetTopologicalMapping

        Args:
            cloth_transform (Sofa.Core.Object -> TransformEngine): the transform engine that transforms the grid to the cloth pose
        """

        triangles_on_path = self.initial_triangles_on_path.copy()

        vertex_indices_on_path = np.unique(self.initial_triangles_on_path)

        vertex_positions_on_path = cloth_transform.output_position.array()[vertex_indices_on_path].copy()

        self.cloth_subset_on_path, self.cloth_subset_topology_on_path = self._add_triangle_subset_topology(
            attached_to=self.node,
            source_topology=self.cloth_topology,
            node_name="cloth_subset_on_path",
            subset_topology_name="cloth_subset_topology_on_path",
            subset_indices=vertex_indices_on_path,
            subset_positions=vertex_positions_on_path,
            subset_triangles=triangles_on_path,
        )

        subset_indices_of_centerline = np.where(np.in1d(vertex_indices_on_path, self.centerline_indices))[0]
        subset_positions_of_centerline = cloth_transform.output_position.array()[self.centerline_indices].copy()

        self.on_path_subset_centerline, self.on_path_subset_topology_centerline = self._add_point_subset_topology(
            attached_to=self.cloth_subset_on_path,
            node_name="on_path_subset_centerline",
            subset_topology_name="on_path_subset_topology_centerline",
            subset_indices=subset_indices_of_centerline,
            subset_positions=subset_positions_of_centerline,
        )
        self.centerline_mechanical_object = self.on_path_subset_centerline.MechanicalObject

        # For each point on the centerline of the path, count how often the point occurs in triangles of the path.
        self.initial_triangle_counts_per_centerline_index = np.sum(self.centerline_indices[:, None] == triangles_on_path.ravel(), axis=1)
        # The first and last index will have 3 triangles, instead of 6, but we neglect them, because this count is just for normalization
        # of a reward feature.
        self.initial_uncut_centerline_points = np.count_nonzero(self.initial_triangle_counts_per_centerline_index == 6)

        if self.show_debug_objects:
            self.centerline_mechanical_object.showObject = True
            self.centerline_mechanical_object.showObjectScale = 5.0
            self.centerline_mechanical_object.showColor = [0, 1, 0]

    def _add_cloth_subset_topology_off_path(self, cloth_transform: Sofa.Core.Object):
        """In order to keep track of the triangles off the cutting path we store the respective vertices in a separate TopologyContainer linked to the cloth via a SubsetTopologicalMapping.

        Args:
            cloth_transform (Sofa.Core.Object -> TransformEngine): the transform engine that transforms the grid to the cloth pose
        """

        vertex_indices_on_path = np.unique(self.initial_triangles_on_path)
        triangles_off_path = self.initial_cloth_triangles[np.invert(np.any(np.isin(self.initial_cloth_triangles, vertex_indices_on_path), axis=1))].copy()
        self.initial_triangles_off_path = triangles_off_path.copy()

        vertex_indices_off_path = np.unique(triangles_off_path)

        vertex_positions_off_path = cloth_transform.output_position.array()[vertex_indices_off_path].copy()

        self.cloth_subset_off_path, self.cloth_subset_topology_off_path = self._add_triangle_subset_topology(
            attached_to=self.node,
            source_topology=self.cloth_topology,
            node_name="cloth_subset_off_path",
            subset_topology_name="cloth_subset_topology_off_path",
            subset_indices=vertex_indices_off_path,
            subset_positions=vertex_positions_off_path,
            subset_triangles=triangles_off_path,
        )

        self.cloth_subset_off_path_mechanical_object = self.cloth_subset_off_path.MechanicalObject

    def _add_triangle_subset_topology(
        self,
        attached_to: Sofa.Core.Node,
        source_topology: Sofa.Core.Object,
        node_name: str,
        subset_topology_name: str,
        subset_indices: np.ndarray,
        subset_positions: np.ndarray,
        subset_triangles: np.ndarray,
        same_points: bool = False,
    ) -> Tuple[Sofa.Core.Node, Sofa.Core.Object]:
        """
        Args:
            same_points (bool): False indicates that the number of vertices in the subset is less than the number of vertices of the source (super)set.
        """

        if not same_points:
            # In order to pass the triangle list to the subset topology we need to remap the respective vertex indices
            # That is because the subset topology contains fewer vertices than the source (super)set
            source_idx_to_subset_idx = {subset_indices[idx]: idx for idx in range(subset_indices.shape[0])}
            subset_triangles = np.vectorize(source_idx_to_subset_idx.get)(subset_triangles)

        subset_topology_node = attached_to.addChild(node_name)
        subset_topology = subset_topology_node.addObject(
            "TriangleSetTopologyContainer",
            name=subset_topology_name,
            position=subset_positions,
            triangles=subset_triangles,
        )
        subset_topology_node.addObject("TriangleSetTopologyModifier")
        subset_topology_node.addObject("TriangleSetGeometryAlgorithms", template="Vec3d")
        subset_topology_node.addObject(
            "SubsetTopologicalMapping",
            input=source_topology.getLinkPath(),
            output=subset_topology.getLinkPath(),
            samePoints=same_points,
            handleTriangles=True,
        )
        subset_topology_node.addObject("MechanicalObject")
        # subset_topology_node.addObject("IdentityMapping")
        subset_topology_node.addObject(
            "SubsetMapping",
            indices=subset_indices,
            handleTopologyChange=False,
            # ignoreNotFound=True,
            # resizeToModel=True,
        )
        return subset_topology_node, subset_topology

    def _add_point_subset_topology(
        self,
        attached_to: Sofa.Core.Node,
        node_name: str,
        subset_topology_name: str,
        subset_indices: np.ndarray,
        subset_positions: np.ndarray,
    ) -> Tuple[Sofa.Core.Node, Sofa.Core.Object]:

        subset_topology_node = attached_to.addChild(node_name)
        subset_topology = subset_topology_node.addObject(
            "PointSetTopologyContainer",
            name=subset_topology_name,
            position=subset_positions,
        )
        subset_topology_node.addObject("PointSetTopologyModifier")
        subset_topology_node.addObject("PointSetGeometryAlgorithms", template="Vec3d")
        subset_topology_node.addObject("MechanicalObject")
        subset_topology_node.addObject(
            "SubsetMapping",
            indices=subset_indices,
            handleTopologyChange=True,
        )

        return subset_topology_node, subset_topology

    def _add_visual_model(
        self,
        translation,
        quaternion,
        grid_resolution,
        visual_grid_resolution,
        visual_grid_topology,
        texture_file_path,
    ):
        self.visual_node = self.node.addChild("visual")
        if visual_grid_resolution != grid_resolution:
            visual_cloth_transform = self.visual_node.addObject(
                "TransformEngine",
                name="visual_cloth_transform",
                template="Vec3d",
                translation=translation,
                quaternion=quaternion,
                input_position=f"{visual_grid_topology.getLinkPath()}.position",
            )
            self.visual_node.addObject(
                "TriangleSetTopologyContainer",
                position=f"{visual_cloth_transform.getLinkPath()}.output_position",
                triangles=visual_grid_topology.triangles.array().copy(),
            )
            self.visual_node.addObject("TriangleSetTopologyModifier")
        # texture_coordinates
        vertex_indices_on_path = np.unique(self.initial_triangles_on_path)
        n_visu = visual_grid_resolution
        ti, tj = np.meshgrid(np.arange(n_visu), np.arange(n_visu), indexing="ij")
        tex_coords = np.column_stack([ti.ravel(), tj.ravel()]) / [2 * n_visu - 1, n_visu - 1]
        tex_coords[vertex_indices_on_path] = [0.75, 0.75]
        if texture_file_path is not None:
            self.ogl_model = self.visual_node.addObject(
                "OglModel",
                texcoords=tex_coords,
                texturename=str(texture_file_path),
            )
        else:
            self.ogl_model = self.visual_node.addObject("OglModel")
        if visual_grid_resolution == grid_resolution:
            self.visual_node.addObject("IdentityMapping")
        else:
            self.visual_node.addObject("BarycentricMapping")

    def _add_clamps_to_cloth(self, grid_resolution):

        # Which indices to map in the first step
        self.corner_indices = [0, grid_resolution - 1]
        add_fixed_constraint_to_indices(
            self.node,
            self.corner_indices,
        )

    def _fix_cloth_edge(self, grid_size, orientation, cloth_to_global):
        box_pos_global = cloth_to_global(np.array([0, 0.5 * grid_size[1], 0])).flatten()
        add_fixed_constraint_in_bounding_box(
            attached_to=self.node,
            min=(-0.5 * grid_size[0], -3.0, -1.0),
            max=(0.5 * grid_size[0], 3.0, 1.0),
            translation=box_pos_global,
            rotation=orientation,
            show_bounding_box=self.show_debug_objects,
        )

    def get_pose(self) -> np.ndarray:
        """Returns the pose of the cloth as XYZ positions and a quaternion."""
        return self.pose

    def get_cutting_stats(self) -> Dict[str, int]:
        """Returns the cutting stats of the cloth."""
        return self._topology_buffer["cutting_stats"]

    def get_points(self) -> np.ndarray:
        """Return all points of the cloth (regardless of lying on or off the cutting path)"""
        return self._topology_buffer["cloth_points"]

    def get_points_on_cutting_path_centerline(self) -> np.ndarray:
        """Return all points of the cloth lying on the cutting path's centerline"""
        return self._topology_buffer["points_on_cutting_path_centerline"]

    def get_points_off_cutting_path(self) -> np.ndarray:
        """Return all points of the cloth not lying on the cutting path"""
        return self._topology_buffer["points_off_cutting_path"]

    def onAnimateEndEvent(self, _) -> None:
        """Called at the end of each animation step.
        Save the number of topology changes in a buffer, which can be queried by the user.
        """
        # buffer access to sofa state
        self.update_topology_buffer()

    def update_topology_buffer(self):
        """
        Notes:
            - Topology changes should be correctly registered since https://github.com/sofa-framework/SofaPython3/pull/316
            - ``.array()`` always returns a read-only array.
        """
        self._topology_buffer["cloth_points"] = self.mechanical_object.position.array()
        self._topology_buffer["points_on_cutting_path_centerline"] = self.centerline_mechanical_object.position.array()
        self._topology_buffer["points_off_cutting_path"] = self.cloth_subset_off_path_mechanical_object.position.array()

        cs = {
            "all_triangles_cloth": self.initial_cloth_triangles.shape[0],
            "all_triangles_on_path": self.initial_triangles_on_path.shape[0],
            "all_triangles_off_path": self.initial_triangles_off_path.shape[0],
            # uncut
            "uncut_triangles_cloth": self.cloth_topology.triangles.shape[0],
            "uncut_triangles_on_path": self.cloth_subset_topology_on_path.triangles.shape[0],
            "uncut_triangles_off_path": self.cloth_subset_topology_off_path.triangles.shape[0],
        }
        cs = cs | {f"cut_triangles_{id}": cs[f"all_triangles_{id}"] - cs[f"uncut_triangles_{id}"] for id in ["cloth", "on_path", "off_path"]}
        self._topology_buffer["cutting_stats"] = cs

    def get_plane_normal(self) -> np.ndarray:
        """The normalized direction perpendicular to the cloth"""
        v = mh.rotated_z_axis(self.get_pose()[3:])
        return v / np.linalg.norm(v)

    def get_cutting_direction(self) -> np.ndarray:
        """The normalized direction of the cut, i.e. the path's projected x axis."""
        # TODO: consider rotation of cutting path as well?
        running_axis = self._cutting_path.running_axis
        if running_axis == RunningAxis.Y:
            v = mh.rotated_y_axis(self.get_pose()[3:])
        else:
            v = mh.rotated_x_axis(self.get_pose()[3:])
        return v / np.linalg.norm(v)

    def onKeypressedEvent(self, event):
        """Keyboard callback for debugging."""
        from pprint import pprint
        from Sofa.constants import Key

        key = event["key"]

        if key == Key.X:
            cutting_stats = self.get_cutting_stats()
            pprint(cutting_stats)
