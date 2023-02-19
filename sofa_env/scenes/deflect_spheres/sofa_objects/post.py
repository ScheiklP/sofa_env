import Sofa
import Sofa.Core
import numpy as np
from enum import Enum, unique
from typing import Optional, Dict
from pathlib import Path

from sofa_env.sofa_templates.mappings import MappingType, MAPPING_PLUGIN_LIST
from sofa_env.sofa_templates.solver import LinearSolverType, add_solver, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, set_color, VISUAL_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST


POST_PLUGIN_LIST = (
    [
        "Sofa.Component.Topology.Container.Dynamic",  # [EdgeSetTopologyContainer]
    ]
    + MAPPING_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
)


@unique
class State(Enum):
    IDLE = 0
    ACTIVE_RIGHT = 1
    ACTIVE_LEFT = 2
    DONE = 3


HERE = Path(__file__).resolve().parent
MODEL_MESH_DIR = HERE.parent.parent.parent.parent / "assets/meshes/models"


class Post:
    """Object to simulate spheres fixed to a board that can be deflected through collision.

    Args:
        parent_node (Sofa.Core.Node): Parent node of the SOFA object.
        name (str): Name of the object.
        position (np.ndarray): XY position of the sphere.
        height (float): Height of the sphere center.
        total_mass (Optional[float]): Mass of the sphere.
        animation_loop_type (AnimationLoopType): The scenes animation loop in order to correctly add constraint correction objects.
        sphere_radius (float): Radius of the sphere.
        colors (Dict[State, np.ndarray]): A mapping between the state of the sphere and its respective RGB colors.
        stiffness (float): Value of the spring stiffness that holds the spheres in place.
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        position: np.ndarray,
        height: float = 50.0,
        total_mass: Optional[float] = 0.1,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        sphere_radius: float = 6.0,
        colors: Dict[State, np.ndarray] = {
            State.IDLE: np.array([0.7, 0.7, 0.7]),
            State.ACTIVE_RIGHT: np.array([0.0, 0.0, 1.0]),
            State.ACTIVE_LEFT: np.array([1.0, 0.0, 0.0]),
            State.DONE: np.array([0.0, 1.0, 0.0]),
        },
        stiffness: float = 1e3,
    ) -> None:

        self.parent_node = parent_node
        self.name = name
        self.total_mass = total_mass
        self.node = self.parent_node.addChild(self.name)
        self.sphere_radius = sphere_radius

        self.colors = colors
        self.state = State.IDLE

        self.position = position
        self.height = height

        self.ode_solver, self.linear_solver = add_solver(self.node, linear_solver_type=LinearSolverType.SPARSELDL)

        pose_start = np.concatenate((position, np.array([0.0, 0.0, 0.0, 0.0, 1.0])))
        pose_end = np.concatenate((position, np.array([height, 0.0, 0.0, 0.0, 1.0])))
        poses = [pose_start, pose_end]
        edges = np.array([0, 1])

        self.node.addObject("EdgeSetTopologyContainer", edges=edges)
        self.node.addObject("MechanicalObject", template="Rigid3d", position=poses)

        if total_mass is not None:
            self.node.addObject("UniformMass", totalMass=total_mass)

        self.node.addObject("RestShapeSpringsForceField", name="anchor", stiffness=stiffness, angularStiffness=stiffness, points=[0, 1])

        if animation_loop_type == AnimationLoopType.FREEMOTION:
            self.node.addObject("LinearSolverConstraintCorrection")

        self.beam_visual = self.node.addChild("beam_visual")
        self.beam_visual.addObject("OglModel", edges=self.node.EdgeSetTopologyContainer.edges.getLinkPath(), lineWidth=4.0, color=self.colors[State.IDLE])
        self.beam_visual.addObject("IdentityMapping")

        self.collision_node = self.node.addChild("collision")
        self.collision_node.addObject("MechanicalObject", template="Rigid3d")
        self.collision_node.addObject("SubsetMapping", indices=[len(poses) - 1])
        self.sphere_collision_model = self.collision_node.addObject("SphereCollisionModel", radius=sphere_radius)

        self.visual_node = add_visual_model(
            self.collision_node,
            surface_mesh_file_path=MODEL_MESH_DIR / "unit_sphere.stl",
            scale=sphere_radius,
            mapping_type=MappingType.RIGID,
            color=tuple(self.colors[self.state]),
        )

    def set_state(self, state: State) -> None:
        """Sets the state of the post and changes the sphere's color accordingly."""
        self.state = state
        set_color(self.visual_node.OglModel, self.colors[self.state])

    def set_position(self, position: np.ndarray, height: Optional[float] = None) -> None:
        """Changes the spheres position."""
        if height is not None:
            self.height = height
        with self.node.MechanicalObject.rest_position.writeable() as state:
            state[0, :2] = position
            state[1, :2] = position
            state[0, 2] = 0.0
            state[1, 2] = self.height

        with self.node.MechanicalObject.position.writeable() as state:
            state[0, :2] = position
            state[1, :2] = position
            state[0, 2] = 0.0
            state[1, 2] = self.height

    def get_sphere_position(self) -> np.ndarray:
        """Reads the spheres current position."""
        return self.node.MechanicalObject.position.array()[1, :3]

    def get_deflection(self) -> float:
        """Reads the difference between the sphere's rest state and its current position."""
        return float(np.linalg.norm(self.node.MechanicalObject.position.array()[1, :3] - self.node.MechanicalObject.rest_position.array()[1, :3]))
