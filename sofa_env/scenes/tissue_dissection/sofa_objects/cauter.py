import numpy as np

from pathlib import Path
from typing import Optional, Union, Callable, Dict, Tuple

import Sofa.Core
import Sofa.SofaDeformable

from sofa_env.sofa_templates.rigid import MechanicalBinding, RIGID_PLUGIN_LIST, PivotizedRigidObject
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.solver import add_solver, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST


CAUTER_PLUGIN_LIST = RIGID_PLUGIN_LIST + SOLVER_PLUGIN_LIST + VISUAL_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST + SOLVER_PLUGIN_LIST


def cauter_collision_model_func(attached_to: Sofa.Core.Node, self: PivotizedRigidObject, collision_group: Optional[int]) -> Tuple[Sofa.Core.Node, Sofa.Core.Node]:
    """Create a collision model for the cauter.

    Consists of two parts.
    The part on the tip has a tag that tells SOFA this collision model acts as a cutting object.
    The second part does not have this tag and models the collision objects on the shaft of the cauter.
    """

    if collision_group is None:
        collision_group = 1

    cutting_collision_model_positions = [
        [0.0, -1.5, 14.4],
        [0.0, -1.2, 14.4],
        [0.0, -0.9, 14.4],
        [0.0, -0.6, 14.4],
        [0.0, -0.3, 14.4],
        [0.0, 0.0, 14.4],
        [0.0, 0.4, 14.4],
        [0.0, 0.7, 14.4],
        [0.0, 1.0, 14.4],
        [0.0, 1.3, 14.4],
        [0.0, 1.6, 14.4],
        [0.0, 1.6, 14.4],
        [0.0, 1.5, 13.4],
        [0.0, 1.4, 12.4],
        [0.0, 1.2, 11.4],
        [0.0, 1.0, 10.4],
        [0.0, 0.6, 9.4],
        [0.0, 0.3, 8.4],
        [0.0, 0.0, 7.4],
    ]
    cutting_collision_model_radii = [0.8 for _ in range(len(cutting_collision_model_positions))]
    cutting_collision_node = attached_to.addChild("cutting_collision")
    self.cutting_mechanical_object = cutting_collision_node.addObject("MechanicalObject", template="Vec3d", position=cutting_collision_model_positions)
    self.cutting_center_index = 5
    self.cutting_tip_index = 0
    self.cutting_sphere_collision_model = cutting_collision_node.addObject("SphereCollisionModel", group=collision_group, radius=cutting_collision_model_radii, tags="CarvingTool")
    cutting_collision_node.addObject("RigidMapping")

    shaft_collision_model_positions = [[0.0, 0.0, z] for z in range(-150, 6, 2)]
    shaft_collision_model_radii = [1.5 for _ in range(len(shaft_collision_model_positions))]

    shaft_collision_node = attached_to.addChild("shaft_collision")
    shaft_collision_node.addObject("MechanicalObject", template="Vec3d", position=shaft_collision_model_positions)
    shaft_collision_node.addObject("SphereCollisionModel", group=collision_group, radius=shaft_collision_model_radii)
    shaft_collision_node.addObject("RigidMapping")

    return cutting_collision_node, shaft_collision_node


class PivotizedCauter(Sofa.Core.Controller, PivotizedRigidObject):
    """Pivotized cauter instrument

    Extends the pivotized rigid object by a collision model split into an active part that is tagged
    as a cutting object, and an inactive part along the shaft of the instrument that is not.
    The instrument can receive a CarvingManager through the set_carving_manager method to control the state (On/Off) of the electrode.
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        total_mass: Optional[float] = None,
        visual_mesh_path: Optional[Union[str, Path]] = None,
        scale: float = 1.0,
        add_solver_func: Callable = add_solver,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        show_object: bool = False,
        show_object_scale: float = 1.0,
        mechanical_binding: MechanicalBinding = MechanicalBinding.SPRING,
        spring_stiffness: Optional[float] = 2e10,
        angular_spring_stiffness: Optional[float] = 2e10,
        add_collision_model_func: Callable = cauter_collision_model_func,
        collision_group: Optional[int] = None,
        ptsd_state: np.ndarray = np.zeros(4),
        rcm_pose: np.ndarray = np.zeros(6),
        cartesian_workspace: Dict = {
            "low": np.array([-np.inf, -np.inf, -np.inf]),
            "high": np.array([np.inf, np.inf, np.inf]),
        },
        ptsd_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        rcm_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        state_limits: Dict = {
            "low": np.array([-90, -90, -90, 0]),
            "high": np.array([90, 90, 90, 100]),
        },
    ) -> None:

        Sofa.Core.Controller.__init__(self)
        self.name: Sofa.Core.DataString = f"{name}_controller"

        PivotizedRigidObject.__init__(
            self,
            parent_node=parent_node,
            name=name,
            total_mass=total_mass,
            visual_mesh_path=visual_mesh_path,
            scale=scale,
            add_solver_func=add_solver_func,
            add_collision_model_func=add_collision_model_func,
            add_visual_model_func=add_visual_model_func,
            animation_loop_type=animation_loop_type,
            show_object=show_object,
            show_object_scale=show_object_scale,
            mechanical_binding=mechanical_binding,
            spring_stiffness=spring_stiffness,
            angular_spring_stiffness=angular_spring_stiffness,
            collision_group=collision_group,
            ptsd_state=ptsd_state,
            rcm_pose=rcm_pose,
            cartesian_workspace=cartesian_workspace,
            ptsd_reset_noise=ptsd_reset_noise,
            rcm_reset_noise=rcm_reset_noise,
            state_limits=state_limits,
            show_remote_center_of_motion=show_object,
            show_workspace=show_object,
        )

        self.carving_manager = None
        self.active = False

    def onKeypressedEvent(self, event) -> None:
        key = event["key"]
        if ord(key) == 19:  # up
            state = self.ptsd_state + np.array([0, -1, 0, 0])
            self.set_state(state)

        elif ord(key) == 21:  # down
            state = self.ptsd_state + np.array([0, 1, 0, 0])
            self.set_state(state)

        elif ord(key) == 18:  # left
            state = self.ptsd_state + np.array([1, 0, 0, 0])
            self.set_state(state)

        elif ord(key) == 20:  # right
            state = self.ptsd_state + np.array([-1, 0, 0, 0])
            self.set_state(state)

        elif key == "T":
            state = self.ptsd_state + np.array([0, 0, 1, 0])
            self.set_state(state)

        elif key == "G":
            state = self.ptsd_state + np.array([0, 0, -1, 0])
            self.set_state(state)

        elif key == "V":
            state = self.ptsd_state + np.array([0, 0, 0, 1])
            self.set_state(state)

        elif key == "D":
            state = self.ptsd_state + np.array([0, 0, 0, -1])
            self.set_state(state)

        elif key == "B":
            self.toggle_active()
            if self.active:
                print("Bzzzt")
            else:
                print("zZzZz")

        elif key == "H":
            print(repr(self.ptsd_state))
        else:
            pass

    def reset_cauter(self) -> None:
        """Reset the cauter to its initial state and optionally apply noise."""
        self.reset_state()
        self.active = False
        if self.carving_manager is not None:
            self.carving_manager.active.value = self.active

    def toggle_active(self) -> None:
        """Toggle the active state of the cauter."""
        self.active = not self.active
        if self.carving_manager is None:
            raise KeyError("To activate the cauter instrument, please pass a CarvingManager through the set_carving_manager method.")
        self.carving_manager.active.value = self.active

    def set_activation(self, active: bool) -> None:
        """Set the active state of the cauter."""
        self.active = active
        if self.carving_manager is None:
            raise KeyError("To activate the cauter instrument, please pass a CarvingManager through the set_carving_manager method.")
        self.carving_manager.active.value = self.active

    def set_carving_manager(self, carving_manager: Sofa.Core.Object) -> None:
        """Set the carving manager to use for this cauter instrument."""
        self.carving_manager = carving_manager
        self.carving_manager.active.value = self.active

    def get_cutting_center_position(self) -> np.ndarray:
        """Get the position of the cutting center (the middle of the active tip) in world coordinates."""
        return self.cutting_mechanical_object.position.array()[self.cutting_center_index]

    def get_cutting_tip_position(self) -> np.ndarray:
        """Get the position of the cutting tip (the round end of the active tip) in world coordinates."""
        return self.cutting_mechanical_object.position.array()[self.cutting_tip_index]
