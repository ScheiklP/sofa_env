import numpy as np

from pathlib import Path
from typing import Tuple, Optional, Union, Callable, List, Dict

import Sofa.Core
import Sofa.SofaDeformable
from Sofa.constants import Key

from sofa_env.sofa_templates.rigid import MechanicalBinding, RIGID_PLUGIN_LIST, PivotizedArticulatedInstrument
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.solver import add_solver, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST


SCISSORS_PLUGIN_LIST = (
    [
        "SofaGeneralRigid",
        "SofaMiscMapping",
        "Sofa.Component.Topology.Container.Constant",
        "SofaCarving",
    ]
    + RIGID_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
)


def jaw_collision_model_func(attached_to: Sofa.Core.Node, collision_group: int, self: PivotizedArticulatedInstrument) -> Sofa.Core.Node:
    self.jaw_collision_carving_nodes = {}
    self.jaw_collision_no_carving_nodes = {}
    # positions of non-carvable collision spheres on jaw 0
    start = np.array([0.3, -1.7, 0.0])
    stop = np.array([0.3, -1.5, 11.0])
    jaw_0_collision_no_carving = np.linspace(start, stop, int(np.floor(np.linalg.norm(stop - start) / 0.6))).tolist()
    start = np.array([0.3, -1.5, 11.0])
    stop = np.array([0.8, -0.7, 14.0])
    jaw_0_collision_no_carving.extend(np.linspace(start, stop, int(np.floor(np.linalg.norm(stop - start) / 0.6))).tolist())

    # positions of carvable collision spheres on jaw 0
    start = np.array([0.0, 1.0, 3.0])
    stop = np.array([0.0, 1.0, 10.0])
    jaw_0_collision_carving = np.linspace(start, stop, int(np.floor(np.linalg.norm(stop - start) / 0.6))).tolist()
    start = np.array([0.0, 1.0, 10.0])
    stop = np.array([0.8, 1.0, 14.0])
    jaw_0_collision_carving.extend(np.linspace(start, stop, int(np.floor(np.linalg.norm(stop - start) / 0.6))).tolist())

    # positions of non-carvable collision spheres on jaw 1
    start = np.array([-0.3, 1.7, 0.0])
    stop = np.array([-0.3, 1.5, 11.0])
    jaw_1_collision_no_carving = np.linspace(start, stop, int(np.floor(np.linalg.norm(stop - start) / 0.6))).tolist()
    start = np.array([-0.3, 1.5, 11.0])
    stop = np.array([0.2, 0.9, 14.0])
    jaw_1_collision_no_carving.extend(np.linspace(start, stop, int(np.floor(np.linalg.norm(stop - start) / 0.6))).tolist())

    # positions of carvable collision spheres on jaw 1
    start = np.array([-0.4, -1.2, 3.0])
    stop = np.array([-0.4, -1.0, 8.0])
    jaw_1_collision_carving = np.linspace(start, stop, int(np.floor(np.linalg.norm(stop - start) / 0.6))).tolist()
    start = np.array([-0.4, -1.0, 8.0])
    stop = np.array([0.3, -1.0, 14.0])
    jaw_1_collision_carving.extend(np.linspace(start, stop, int(np.floor(np.linalg.norm(stop - start) / 0.6))).tolist())

    for num, jaw in enumerate(("jaw_0", "jaw_1")):
        # add cuttable collision model to jaw
        self.jaw_collision_carving_nodes[jaw] = self.physical_jaw_node.addChild(f"{jaw}_collision_carving")
        self.jaw_collision_carving_nodes[jaw].addObject(
            "MechanicalObject",
            template="Vec3d",
            position=jaw_0_collision_carving if num == 0 else jaw_1_collision_carving,
        )
        self.jaw_collision_carving_nodes[jaw].addObject(
            "RigidMapping",
            input=self.joint_mechanical_object.getLinkPath(),
            index=num + 1,
            # globalToLocalCoords=True,
        )
        # Init, so that we can pass the positions to the cutting node
        self.jaw_collision_carving_nodes[jaw].init()

        # add non-cuttable collision model to jaw
        self.jaw_collision_no_carving_nodes[jaw] = self.physical_jaw_node.addChild(f"{jaw}_collision_no_carving")
        self.jaw_collision_no_carving_nodes[jaw].addObject(
            "MechanicalObject",
            template="Vec3d",
            position=jaw_0_collision_no_carving if num == 0 else jaw_1_collision_no_carving,
        )
        self.jaw_collision_no_carving_nodes[jaw].addObject(
            "SphereCollisionModel",
            radius=0.8,
            group=collision_group,
            color=(1, 0.5, 1, 1),
        )
        self.jaw_collision_no_carving_nodes[jaw].addObject(
            "RigidMapping",
            input=self.joint_mechanical_object.getLinkPath(),
            index=num + 1,
        )

    self.jaws_cutting_node = self.physical_jaw_node.addChild("cutting")
    self.jaws_cutting_node.addObject(
        "MechanicalObject",
        template="Vec3d",
        position=[self.jaw_collision_carving_nodes[jaw].MechanicalObject.position.array() for jaw in ("jaw_0", "jaw_1")],
    )
    self.jaws_cutting_node.addObject(
        "SphereCollisionModel",
        radius=0.8,
        group=collision_group,
        tags="CarvingTool",
    )
    # Create a list that tells the SubsetMultiMapping how to map the indices of the MechanicalObject to the MechanicalObjects of the jaws.
    # [[jaw number, index in the jaw], [jaw number, index in the jaw]] -> each entry corresponds to the index of the MechanicalObject -> flattened out for SOFA
    n = {i: len(self.jaw_collision_carving_nodes[f"jaw_{i}"].MechanicalObject.position) for i in range(2)}
    make_pair = lambda i: np.column_stack((np.full(n[i], i), np.arange(n[i]))).ravel().astype(int)
    self.jaws_cutting_node.addObject(
        "SubsetMultiMapping",
        template="Vec3d,Vec3d",
        input=[self.jaw_collision_carving_nodes[jaw].MechanicalObject.getLinkPath() for jaw in ("jaw_0", "jaw_1")],
        output=self.jaws_cutting_node.MechanicalObject.getLinkPath(),
        indexPairs=np.concatenate([make_pair(0), make_pair(1)]).tolist(),
    )


class ArticulatedScissors(Sofa.Core.Controller, PivotizedArticulatedInstrument):
    """
    Note:
        The CarvingManager can handle ONLY ONE ``CollisionModel`` as ``CarvingTool``. If there are multiple ``CollisionModel``s with ``CarvingTool`` tag, only the first one will be used. The jaws, however, should behave independently based on the articulation of ``ArticulatedInstrument``. Therefore, we
        1. create a ``MechanicalObject``, that holds as many points, as there are combined points on the jaw,
        2. add a ``SubsetMultiMapping`` that maps the ``MechanicalObjects`` of the individual jaws onto the new ``MechanicalObject``
        3. create a ``CollisionModel`` on the new ``MechanicalObject``
        Now, when the jaws move the points in the ``MechanicalObject`` will be updated accordingly
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        visual_mesh_path_shaft: Union[str, Path],
        visual_mesh_paths_jaws: List[Union[str, Path]],
        ptsd_state: np.ndarray = np.zeros(4),
        rcm_pose: np.ndarray = np.zeros(6),
        angle: float = 0.0,
        angle_limits: Tuple[float, float] = (-0.0, 60.0),
        total_mass: Optional[float] = None,
        rotation_axis: Tuple[int, int, int] = (1, 0, 0),
        scale: float = 1.0,
        add_solver_func: Callable = add_solver,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        show_object: bool = False,
        show_object_scale: float = 1.0,
        mechanical_binding: MechanicalBinding = MechanicalBinding.ATTACH,
        spring_stiffness: Optional[float] = 1e12,
        angular_spring_stiffness: Optional[float] = 1e12,
        articulation_spring_stiffness: Optional[float] = 1e12,
        cartesian_workspace: Dict = {
            "low": np.array([-np.inf, -np.inf, -np.inf]),
            "high": np.array([np.inf, np.inf, np.inf]),
        },
        ptsd_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        rcm_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        angle_reset_noise: Optional[Union[float, Dict[str, float]]] = None,
        state_limits: Dict = {
            "low": np.array([-90, -95, -np.inf, 0]),
            "high": np.array([90, 95, np.inf, 200]),
        },
        show_remote_center_of_motion: bool = False,
        collision_group: int = 0,
    ) -> None:
        Sofa.Core.Controller.__init__(self)
        self.name = f"{name}_controller"

        PivotizedArticulatedInstrument.__init__(
            self,
            parent_node=parent_node,
            name=f"{name}_instrument",
            visual_mesh_path_shaft=visual_mesh_path_shaft,
            visual_mesh_paths_jaws=visual_mesh_paths_jaws,
            angle=angle,
            angle_limits=angle_limits,
            total_mass=total_mass,
            two_jaws=True,
            rotation_axis=rotation_axis,
            scale=scale,
            add_solver_func=add_solver_func,
            add_visual_model_func=add_visual_model_func,
            animation_loop_type=animation_loop_type,
            show_object=show_object,
            show_object_scale=show_object_scale,
            mechanical_binding=mechanical_binding,
            spring_stiffness=spring_stiffness,
            angular_spring_stiffness=angular_spring_stiffness,
            collision_group=collision_group,
            add_jaws_collision_model_func=[jaw_collision_model_func, lambda attached_to: None],
            ptsd_state=ptsd_state,
            rcm_pose=rcm_pose,
            articulation_spring_stiffness=articulation_spring_stiffness,
            cartesian_workspace=cartesian_workspace,
            ptsd_reset_noise=ptsd_reset_noise,
            rcm_reset_noise=rcm_reset_noise,
            angle_reset_noise=angle_reset_noise,
            state_limits=state_limits,
            show_remote_center_of_motion=show_remote_center_of_motion,
        )

        self._add_shaft_collision_model(collision_group)

    def _add_shaft_collision_model(self, collision_group: int):
        shaft_collision_model_positions = [[0.0, 0.0, z] for z in range(-260, 0, 3)]
        shaft_collision_model_radii = [3 for _ in range(len(shaft_collision_model_positions))]
        self.shaft_collision_model_node = self.physical_shaft_node.addChild("collision_shaft")
        self.shaft_collision_model_node.addObject(
            "MechanicalObject",
            template="Vec3d",
            position=shaft_collision_model_positions,
        )
        self.shaft_collision_model_node.addObject(
            "SphereCollisionModel",
            group=collision_group,
            radius=shaft_collision_model_radii,
            color=(1, 0.5, 1, 1),
        )
        self.shaft_collision_model_node.addObject("RigidMapping")

    @staticmethod
    def get_action_dims() -> int:
        """Return the action dimensionality.
        [pan, tilt, spin, depth, jaw_angle]
        """
        return 5

    def get_cutting_center_position(self) -> np.ndarray:
        """Return the center position where the jaws are attached to the shaft"""
        return self.get_pose()[:3]

    def incremental_action(self, pan: float = 0, tilt: float = 0, spin: float = 0, depth: float = 0, jaw_angle: float = 0):
        ptsd_state = self.ptsd_state + np.array([pan, tilt, spin, depth])
        angle = self.get_angle() + jaw_angle
        self.set_articulated_state(np.concatenate([ptsd_state, angle]))

    def onKeypressedEvent(self, event):
        key = event["key"]
        # increment depth
        if key == Key.uparrow:
            self.incremental_action(depth=1)
        # decrement depth
        elif key == Key.downarrow:
            self.incremental_action(depth=-1)
        # increment tilt
        elif key == Key.F:
            self.incremental_action(tilt=1)
        # decrement tilt
        elif key == Key.D:
            self.incremental_action(tilt=-1)
        # increment pan
        elif key == Key.rightarrow:
            self.incremental_action(pan=1)
        # decrement pan
        elif key == Key.leftarrow:
            self.incremental_action(pan=-1)
        # increment spin
        elif key == Key.T:
            self.incremental_action(spin=1)
        # decrement spin
        elif key == Key.G:
            self.incremental_action(spin=-1)
        # increment jaw angle
        elif key == Key.P:
            self.incremental_action(jaw_angle=1)
        # decrement jaw angle
        elif key == Key.B:
            self.incremental_action(jaw_angle=-1)
        # print state
        elif key == Key.H:
            print(repr(self.ptsd_state), self.get_angle())
            print(f"{self.get_pose() = }")
            print(f"{self.get_physical_pose() = }")
            print(f"{self.pivot_transform(self.get_state()) = }")
        else:
            pass
