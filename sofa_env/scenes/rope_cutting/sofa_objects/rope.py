from typing import Optional, Tuple
import Sofa
import Sofa.Core
import Sofa.SofaDeformable
import numpy as np
from sofa_env.sofa_templates.rope import poses_for_rope_between_points
from sofa_env.sofa_templates.solver import LinearSolverType, add_solver, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import set_color, VISUAL_PLUGIN_LIST

ROPE_PLUGIN_LIST = (
    [
        "Sofa.Component.Topology.Container.Dynamic",  # [EdgeSetTopologyContainer, EdgeSetTopologyModifier]
        "Sofa.Component.MechanicalLoad",  # [PlaneForceField]
        "Sofa.Component.Topology.Mapping",  # [Edge2QuadTopologicalMapping]
    ]
    + SOLVER_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
)


class CuttableRope(Sofa.Core.Controller):
    """Cuttable rope that keeps track of topological changes.

    This rope is implemented as a set of vertices and edges that can be cut with
    the SofaCarving plugin. Each simulation step checks the current size of the
    topology, and saves changes in ``topology_change_buffer`` which can be consumed
    and reset to 0 with ``consume_topology_change_buffer()``.
    Collisions are modeled as lines on the edges of the rope.

    Args:
        attached_to (Sofa.Core.Node): Parent node in the simulation tree.
        start_position (np.ndarray): Cartesian position where the start of the rope is attached.
        end_position (np.ndarray): Cartesian position where the end of the rope is attached.
        name (str): Name of the object.
        number_of_points (int): Number of points that are added to the rope.
        color (Tuple[int, int, int]): RGB values for the visual model of the rope [0, 255].
        show_object (bool): Whether to visualize the points on the rope.
        show_object_scale (float): Scaling factor for point visualization.
        stiffness (float): Stiffness of the spring force field of the rope.
        total_mass (float): Total mass of the rope.
        plane_height (Optional[float]): If set, add a ``PlaneForceField`` that acts on the rope points, if they fall below the ``plane_height``.
        Acts as a floor.
        contraction_ratio (float): Parameterize the initial tension of the rope. If set to ``0.0``, the segments of the rope are as long as the rope self at rest.
        At ``1.0``, the springs that model the behavior of the rope are set to ``0.0`` lenght.
        visual_resolution (int): Number of points per circle for the visual model of the rope.
        radius (float): Visual radius of the rope.
    """

    def __init__(
        self,
        attached_to: Sofa.Core.Node,
        start_position: np.ndarray,
        end_position: np.ndarray,
        name: str,
        number_of_points: int = 10,
        color: Tuple[int, int, int] = (255, 0, 0),
        show_object: bool = False,
        show_object_scale: float = 10.0,
        stiffness: float = 1e3,
        total_mass: float = 0.01,
        plane_height: Optional[float] = None,
        contraction_ratio: float = 0.8,
        visual_resolution: int = 10,
        radius: float = 1.0,
    ) -> None:

        super().__init__()
        self.name: Sofa.Core.DataString = f"{name}_controller"

        self.removed_edges = []
        self.parent_node = attached_to
        self.node = attached_to.addChild(name)
        self.ode_solver, self.linear_solver = add_solver(
            self.node,
            linear_solver_type=LinearSolverType.SPARSELDL,
        )

        if number_of_points < 2:
            raise ValueError(f"Minimum number of points to simulate a CuttableRope is 2. Received {number_of_points}.")

        rope_poses = poses_for_rope_between_points(start_position, end_position, num_points=number_of_points)
        edges = [[x, x + 1] for x in range(len(rope_poses) - 1)]
        self.number_of_points = number_of_points

        self.topology = self.node.addObject("EdgeSetTopologyContainer", edges=edges)
        self.node.addObject("EdgeSetTopologyModifier")
        self.mechanical_object = self.node.addObject("MechanicalObject", template="Rigid3d", position=np.array(rope_poses).tolist(), showObject=show_object, showObjectScale=show_object_scale)
        self.node.addObject("UniformMass", totalMass=total_mass)

        internal_spring_length = float(np.linalg.norm(rope_poses[0][:3] - rope_poses[1][:3]))
        self.internal_springs = []
        for edge_start, edge_end in edges:
            self.internal_springs.append([edge_start, edge_end, stiffness, 0.0, internal_spring_length * (1 - contraction_ratio)])
        self.spring_force_field = self.node.addObject(
            "StiffSpringForceField",
            showArrowSize=0.1,
            drawMode=1,
            spring=self.internal_springs,
        )
        if plane_height is not None:
            self.node.addObject(
                "PlaneForceField",
                normal=[0, 0, 1],
                d=float(plane_height),
                stiffness=str(stiffness),
                damping=1.0,
                showPlane=True,
                showPlaneSize=100,
            )

        self.node.addObject("FixedProjectiveConstraint", indices=[0, len(rope_poses) - 1])
        self.node.addObject("LinearSolverConstraintCorrection")

        collision_node = self.node.addChild("collision")
        collision_node.addObject("MechanicalObject", template="Vec3d")
        collision_node.addObject("IdentityMapping")
        collision_node.addObject("LineCollisionModel", tags="CarvingSurface")

        # Tube for visual model
        tube_node = self.node.addChild("tube")
        tube_mechanical_object = tube_node.addObject("MechanicalObject")
        tube_topology = tube_node.addObject("QuadSetTopologyContainer")
        tube_node.addObject("QuadSetTopologyModifier")
        tube_node.addObject(
            "Edge2QuadTopologicalMapping",
            nbPointsOnEachCircle=visual_resolution,
            radius=radius,
            input=self.topology.getLinkPath(),
            output=tube_topology.getLinkPath(),
        )
        tube_node.addObject(
            "TubularMapping",
            nbPointsOnEachCircle=visual_resolution,
            radius=radius,
            input=self.mechanical_object.getLinkPath(),
            output=tube_mechanical_object.getLinkPath(),
        )

        # OGL visual model
        visual_node = tube_node.addChild("visual")
        self.ogl_model = visual_node.addObject("OglModel", color=color)
        visual_node.addObject("IdentityMapping", input=tube_mechanical_object.getLinkPath(), output=self.ogl_model.getLinkPath())

        self.topology_change_buffer = 0
        self.parent_node.addObject(self)

    def onSimulationInitDoneEvent(self, _) -> None:
        """Called when the simulation is initialized."""
        self.initial_size = self.topology.nbPoints.value
        self.initial_num_edges = len(self.topology.edges)
        self.last_size = self.initial_size
        self.last_num_edges = self.initial_num_edges

    def onAnimateEndEvent(self, _) -> None:
        """Called at the end of each animation step.

        Handle topological changes and propagate them to the spring force field.
        Save the number of topology changes in a buffer, which can be queried by the user.
        """

        current_size = self.topology.nbPoints.value
        current_num_edges = len(self.topology.edges)

        self.topology_change_buffer += self.last_size - current_size
        self.topology_change_buffer += self.last_num_edges - current_num_edges

        self.last_size = current_size
        self.last_num_edges = current_num_edges

    def set_color(self, color: Tuple[int, int, int]) -> None:
        """Sets the color of the rope."""
        set_color(self.ogl_model, color=color)

    def get_positions(self) -> np.ndarray:
        """Returns the positions of the rope."""
        return self.mechanical_object.position.array()[:, :3]

    def set_positions(self, positions: np.ndarray, indices: Optional[np.ndarray] = None) -> None:
        with self.mechanical_object.position.writeable() as array:
            if indices is None:
                array[:, :3] = positions
            else:
                array[indices, :3] = positions

    def consume_topology_change_buffer(self) -> int:
        """Returns the number of topology changes since the last call to this function."""
        topology_change = self.topology_change_buffer
        self.topology_change_buffer = 0
        return topology_change
