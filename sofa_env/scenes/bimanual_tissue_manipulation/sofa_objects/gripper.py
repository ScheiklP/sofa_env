import numpy as np
import Sofa.Core
from Sofa.constants import Key

from typing import Tuple, Optional, Callable, Dict
from enum import Enum, unique
from functools import partial
from pathlib import Path

from sofa_env.sofa_templates.rigid import PivotizedArticulatedInstrument, MechanicalBinding, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.solver import add_solver, ConstraintCorrectionType, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST

GRIPPER_PLUGIN_LIST = RIGID_PLUGIN_LIST + SOLVER_PLUGIN_LIST + VISUAL_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST + SOLVER_PLUGIN_LIST


@unique
class Side(Enum):
    LEFT = 0
    RIGHT = 1


class PivotizedGripper(Sofa.Core.Controller, PivotizedArticulatedInstrument):
    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        side: Side,
        name: str,
        visual_mesh_path: Path,
        visual_mesh_path_jaw: Path,
        ptsd_state: np.ndarray = np.zeros(4, dtype=np.float32),
        rcm_pose: np.ndarray = np.zeros(6, dtype=np.float32),
        state_limits: Dict = {
            "low": np.array([-75.0, -50.0, -180.0, -200.0]),
            "high": np.array([75.0, 50.0, 180.0, 0.0]),
        },
        spring_stiffness: float = 1e8,
        angular_spring_stiffness: float = 1e8,
        total_mass: Optional[float] = 1.0,
        rotation_axis: Tuple[int, int, int] = (1, 0, 0),
        scale: float = 50.0,
        add_solver_func: Callable = add_solver,
        add_visual_model_func: Callable = partial(add_visual_model, color=(0, 0, 1)),
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        show_object: bool = False,
        show_object_scale: float = 1.0,
        mechanical_binding: MechanicalBinding = MechanicalBinding.SPRING,
        articulation_spring_stiffness: float = 1e15,
        collision_group: int = 0,
    ) -> None:
        Sofa.Core.Controller.__init__(self)

        PivotizedArticulatedInstrument.__init__(
            self,
            parent_node=parent_node,
            name=f"{name}_instrument",
            ptsd_state=ptsd_state,
            rcm_pose=rcm_pose,
            state_limits=state_limits,
            visual_mesh_path_shaft=visual_mesh_path,
            visual_mesh_paths_jaws=[visual_mesh_path_jaw],
            two_jaws=False,
            total_mass=total_mass,
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
            articulation_spring_stiffness=articulation_spring_stiffness,
            show_remote_center_of_motion=show_object,
            collision_group=collision_group,
        )

        self.controller_name: Sofa.Core.DataString = f"{name}_controller"
        self.gripper_name = name
        self.attached = False
        self.side_index = side.value
        self.show_object = show_object

        self._add_shaft_collision_model(collision_group)

    def _add_shaft_collision_model(self, collision_group: int):
        shaft_collision_model_positions = [[0.0, -0.5, z] for z in range(10, 110, 3)]
        shaft_collision_model_radii = [2 for _ in range(len(shaft_collision_model_positions))]
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

    def do_action(self, action: np.ndarray, absolute: bool = False):
        """Moves the gripper based on the action.

        Args:
            action (np.ndarray): pd-Action.
            absolute (bool, optional): Whether the action is absolute or relative. Defaults to False.
        """

        # Set Gripper state
        ptsd_state = np.array([action[0], 0.0, 0.0, action[1]])
        if not absolute:
            ptsd_state += self.ptsd_state
        self.set_state(ptsd_state)

        # Set Tissue target
        if self.attached:
            new_pose = self.get_pose()
            with self.rigidified_tissue_motion_target_mechanical_object.position.writeable() as state:
                state[self.side_index][:] = new_pose

    def onKeypressedEvent(self, event):
        key = event["key"]
        # increment depth
        if key == Key.uparrow:
            self.do_action(np.array([0.0, 0.0, 0.0, 1.0]))
        # decrement depth
        elif key == Key.downarrow:
            self.do_action(np.array([0.0, 0.0, 0.0, -1.0]))
        # increment tilt
        elif key == Key.F:
            self.do_action(np.array([0.0, 1.0, 0.0, 0.0]))
        # decrement tilt
        elif key == Key.D:
            self.do_action(np.array([0.0, -1.0, 0.0, 0.0]))
        # increment pan
        elif key == Key.rightarrow:
            self.do_action(np.array([-1.0, 0.0, 0.0, 0.0]))
        # decrement pan
        elif key == Key.leftarrow:
            self.do_action(np.array([1.0, 0.0, 0.0, 0.0]))
        # increment spin
        elif key == Key.T:
            self.do_action(np.array([0.0, 0.0, 1.0, 0.0]))
        # decrement spin
        elif key == Key.G:
            self.do_action(np.array([0.0, 0.0, -1.0, 0.0]))
        # print state
        elif key == Key.H:
            print(f"{self.gripper_name} pose: {self.get_pose()}")
            print(f"{self.gripper_name} physical pose: {self.get_physical_pose()}")
        else:
            pass

    def get_grasp_center_position(self) -> np.ndarray:
        """Return the position, where the tissue is attached to the gripper"""
        return self.get_pose()[:3]

    def attach_to_tissue(self, rigidified_tissue_node: Sofa.Core.Node) -> None:
        """Attach the tissue to the gripper

        Args:
            rigidified_tissue_node (Sofa.Core.Node): The Sofa node containing the rigidified tissue.

        """
        if self.attached:
            return
        else:
            rigidified_tissue_motion_target = rigidified_tissue_node.addChild(f"{self.gripper_name}_rigid_motion_target")

            self.rigidified_tissue_motion_target_mechanical_object = rigidified_tissue_motion_target.addObject(
                "MechanicalObject",
                name="MotionTarget",
                template="Rigid3d",
                position=rigidified_tissue_node.rigid.MechanicalObject.position.array(),
                showObject=self.show_object,
                showObjectScale=0.005,
            )

            rigidified_tissue_motion_target.addObject(
                ConstraintCorrectionType.UNCOUPLED.value,
                compliance=1.0,
            )

            rigidified_tissue_node.rigid.addObject(
                "RestShapeSpringsForceField",
                name=f"{self.gripper_name}_rest_shape_springs_force_field",
                stiffness=(2e6, 0.0) if self.side_index == 0 else (0.0, 2e6),  # increase this if the body drags behind the target while moving
                angularStiffness=(2e9, 0.0) if self.side_index == 0 else (0.0, 2e9),  # increase this if there is a rotational offset between body and target
                external_rest_shape=rigidified_tissue_motion_target.getLinkPath(),  # "rest_shape can be defined by the position of an external Mechanical State"
                drawSpring=self.show_object,
            )

            self.attached = True
