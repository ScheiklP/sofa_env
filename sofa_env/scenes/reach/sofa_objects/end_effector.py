import Sofa.Core
import Sofa.SofaDeformable

import numpy as np
from typing import Tuple, Optional, Union, Callable, List
from pathlib import Path
from functools import partial

from sofa_env.sofa_templates.mappings import MappingType, MAPPING_PLUGIN_LIST
from sofa_env.sofa_templates.rigid import ControllableRigidObject, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST
from sofa_env.sofa_templates.solver import add_solver, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST

from sofa_env.utils.robot import get_main_link_pose_transformation
from sofa_env.utils.math_helper import is_in

END_EFFECTOR_PLUGIN_LIST = ["SofaGeneralRigid"] + RIGID_PLUGIN_LIST + VISUAL_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST + MAPPING_PLUGIN_LIST + SOLVER_PLUGIN_LIST


class EndEffector(Sofa.Core.Controller):
    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        pose: Union[Tuple[float, float, float, float, float, float, float], np.ndarray],
        randomize_starting_position: bool = True,
        starting_box: Optional[dict] = None,
        visual_mesh_path_gripper: Optional[Union[str, Path]] = None,
        visual_mesh_path_main_link: Optional[Union[str, Path]] = None,
        remote_center_of_motion: Optional[Union[np.ndarray, Tuple[float, float, float]]] = None,
        gripper_color: Tuple[int, int, int] = (255, 0, 0),
        main_link_color: Tuple[int, int, int] = (255, 105, 0),
        scale: float = 1.0,
        add_solver_func: Callable = add_solver,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        workspace: dict = {
            "low": np.array([-np.inf] * 7),
            "high": np.array([np.inf] * 7),
        },
        show_object: bool = False,
        show_object_scale: float = 0.01,
        add_visual_marker: bool = False,
        visual_marker_color: Tuple[int, int, int] = (0, 255, 0),
        visual_marker_mesh_path: Optional[Union[str, Path]] = None,
        visual_marker_scale: float = 1.0,
    ) -> None:
        """Python object that creates SOFA objects and python logic to represent a robotic end-effector.

        Args:
            parent_node (Sofa.Core.Node): Parent node of the object.
            name (str): Name of the object.
            pose (Union[Tuple[float, float, float, float, float, float, float], np.ndarray]): 6D pose of the object described as Cartesian position and quaternion.
            randomize_starting_position (bool): Whether to pick a random starting position on reset.
            starting_box (Optional[dict]): A dictionary with keys ``high`` and ``low`` that limit the XYZ values of the random starting positions.
            visual_mesh_path_gripper (Optional[Union[str, Path]]): Path to the visual surface mesh of the gripper.
            visual_mesh_path_main_link (Optional[Union[str, Path]]): Path to the visual surface mesh of the main link of the PSM.
            remote_center_of_motion (Optional[Union[np.ndarray, Tuple[float, float, float]]]): Remote center of motion that constraints the motion of the main link.
            scale (float): Scale factor for loading the meshes.
            add_solver_func (Callable): Function that adds the numeric solvers to the object.
            add_visual_model_func (Callable): Function that defines how the visual surface from visual_mesh_path is added to the object.
            animation_loop_type (AnimationLoopType): The animation loop of the scene. Required to determine if objects for constraint correction should be added.
            workspace (dict): A dictionary with keys ``high`` and ``low`` that limit the XYZ values of the gripper workspace.
            show_object (bool): Whether to render the nodes.
            show_object_scale (float): Render size of the node if ``show_object`` is ``True``.
            add_visual_marker (bool): Whether to add a visual model at the end effector's origin.
            visual_marker_color (Tuple[int, int, int]): Color of the visual marker.
            visual_marker_mesh_path (Optional[Union[str, Path]]): Path to the visual surface mesh of the visual marker.
            visual_marker_scale (float): Scale factor for loading the visual marker mesh.
        """

        Sofa.Core.Controller.__init__(self)

        self.name = f"{name}_controller"

        assert len(workspace["low"]) == len(pose), f"Please pass the workspace limits in the same shape as the pose. Got {len(workspace['low'])} elements in workspace['low'], but expected {len(pose)}."
        assert len(workspace["high"]) == len(pose), f"Please pass the workspace limits in the same shape as the pose. Got {len(workspace['high'])} elements in workspace['high'], but expected {len(pose)}."
        self.workspace = workspace

        self.randomize_starting_position = randomize_starting_position
        if randomize_starting_position:
            assert starting_box is not None, "If you want to randomize the end effector's starting position on reset, please pass a dictionary that describes a box of possible starting positions <starting_box>."
            assert len(starting_box["low"]) == len(pose), f"Please pass the starting_box limits in the same shape as the pose. Got {len(starting_box['low'])} elements in starting_box['low'], but expected {len(pose)}."
            assert len(starting_box["high"]) == len(pose), f"Please pass the starting_box limits in the same shape as the pose. Got {len(starting_box['high'])} elements in starting_box['high'], but expected {len(pose)}."
            self.starting_box = starting_box

            # Set a default random number generator in init (for scene creation) that can be overwritten from the env with the seed method
            self.rng = np.random.default_rng()
            print(f"Initial position will be randomized instead of taking passed value for {pose=}")
            self.initial_pose = self.rng.uniform(self.starting_box["low"], self.starting_box["high"])
        else:
            self.initial_pose = np.array(pose)

        self.motion_path = []

        self.node = parent_node.addChild(name)

        # Add the gripper part of the end effector
        self.gripper = ControllableRigidObject(
            parent_node=self.node,
            name=f"{name}_gripper",
            pose=self.initial_pose,
            scale=scale,
            add_solver_func=add_solver_func,
            animation_loop_type=animation_loop_type,
            show_object=show_object,
            show_object_scale=show_object_scale,
        )

        # Visual model of the open gripper
        assert visual_mesh_path_gripper is not None
        self.visual_model_node_open = add_visual_model(
            attached_to=self.gripper.motion_target_node,
            name="visual_open",
            surface_mesh_file_path=visual_mesh_path_gripper,
            mapping_type=MappingType.RIGID,
            color=tuple(intensity / 255 for intensity in gripper_color),
        )

        if add_visual_marker:
            # Visual model of the marker
            assert visual_marker_mesh_path is not None
            self.visual_marker_node = add_visual_model(
                attached_to=self.gripper.motion_target_node,
                name="visual_marker",
                surface_mesh_file_path=visual_marker_mesh_path,
                mapping_type=MappingType.RIGID,
                color=tuple(intensity / 255 for intensity in visual_marker_color),
                scale=visual_marker_scale,
            )

        # Add the PSM main link
        if visual_mesh_path_main_link is not None:
            assert remote_center_of_motion is not None

            self._has_main_link = True

            self.transform_main_link_pose = get_main_link_pose_transformation(
                base_vector=(0.0, 1.0, 0.0),
                remote_center_of_motion=remote_center_of_motion,
                link_offset=(0.0063, 0.016, 0.0),
            )

            self.initial_main_link_pose = self.transform_main_link_pose(self.initial_pose[:3])

            add_visual_model_func = partial(add_visual_model, color=tuple(intensity / 255 for intensity in main_link_color))
            self.main_link = ControllableRigidObject(
                parent_node=self.node,
                name=f"{name}_main_link",
                pose=self.initial_main_link_pose,
                visual_mesh_path=visual_mesh_path_main_link,
                scale=scale,
                add_solver_func=add_solver_func,
                add_visual_model_func=add_visual_model_func,
                animation_loop_type=animation_loop_type,
                show_object=show_object,
            )

            if show_object:
                if isinstance(remote_center_of_motion, np.ndarray):
                    trocar_pose = np.append(remote_center_of_motion, (0.0, 0.0, 0.0, 1.0))
                else:
                    trocar_pose = remote_center_of_motion + (0.0, 0.0, 0.0, 1.0)

                self.node.addObject(
                    "MechanicalObject",
                    template="Rigid3d",
                    position=trocar_pose,
                    showObject=show_object,
                    showObjectScale=0.005,
                )
        else:
            self._has_main_link = False

    def create_linear_motion(
        self,
        target_position: np.ndarray,
        dt: float,
        velocity: float,
        single_step: bool = False,
        start_position: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Creates movement path to displace EndEffector from its current position to the final position provided, at the given velocity.

        Args:
            target_position (np.ndarray): Final position of the linear motion.
            dt (float): Delta T of the simulation.
            velocity (float): Desired velocity of the robot.
            single_step (bool): Whether to perform the motion in one single step.
            start_position (Optional[np.ndarray]): Starting position of the linear motion.
        """

        if start_position is None:
            # Only use xyz
            current_position = self.gripper.motion_target_mechanical_object.position.array()[0, :3]
        else:
            assert len(start_position) == 3
            current_position = start_position

        displacement = target_position - current_position

        motion_path = []

        if single_step:
            motion_path = [target_position]
            motion_steps = 1
        else:
            displacement_per_step = velocity * dt
            motion_steps = int(np.ceil(np.linalg.norm(displacement) / displacement_per_step))

            progress = np.linspace(0.0, 1.0, motion_steps + 1)[1:]
            motion_path = current_position + displacement * progress[:, np.newaxis]

            motion_path[-1] = target_position

        return np.split(motion_path, motion_steps, axis=0)

    def add_to_motion_path(self, motion_path: List[np.ndarray]) -> None:
        """Adds a list of points to the EndEffector's motion path.
        Motion path will be executed by iterating through the list triggered by the ``onAnimateEndEvent``.
        """

        self.motion_path.extend(motion_path)

    def clear_motion_path(self) -> None:
        """Removes all points in the EndEffector's motion path."""

        self.motion_path = []

    def get_pose(self) -> np.ndarray:
        """Reads the Rigid3d pose from the EndEffector as [x, y, z, a, b, c, w]."""
        return self.gripper.get_pose()

    def set_pose(self, pose: np.ndarray) -> np.ndarray:
        """Writes the Rigid3d pose to the EndEffector as [x, y, z, a, b, c, w]."""
        current_pose = self.get_pose()

        # check if there are any poses that are outside the workspace
        invalid_poses_mask = np.invert(list(map(is_in, pose, self.workspace["low"], self.workspace["high"])))

        # overwrite the invalide parts of the pose with the current pose
        pose[invalid_poses_mask] = current_pose[invalid_poses_mask]

        self.gripper.set_pose(pose)

        if self._has_main_link:
            main_link_pose = self.transform_main_link_pose(pose[:3])
            self.main_link.set_pose(main_link_pose)

        return invalid_poses_mask

    def onAnimateBeginEvent(self, _) -> None:
        """This function is called by SOFA's event loop at the start of an animation step."""

        # Check if there is a motion path and execute one step
        if len(self.motion_path):
            new_pose = np.append(self.motion_path.pop(0), [0.0, 0.0, 0.0, 1.0])
            self.set_pose(new_pose)

    def reset(self) -> None:
        """Reset the grasping state of the gripper and set it to it's initial pose.
        The initial position is randomly chosen from the ``starting_box`` if ``randomize_starting_position`` is set to True.
        """

        if self.randomize_starting_position:
            self.initial_pose = self.rng.uniform(self.starting_box["low"], self.starting_box["high"])

        self.set_pose(self.initial_pose)

    def seed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Creates a random number generator from a seed."""
        self.rng = np.random.default_rng(seed)


def add_waypoints_to_end_effector(way_points: List, end_effector: EndEffector, speed: float = 0.005, dt: float = 0.1) -> None:
    for i in range(len(way_points)):
        if i == 0:
            motion_path = end_effector.create_linear_motion(np.array(way_points[i]), dt, speed)
        else:
            motion_path = end_effector.create_linear_motion(np.array(way_points[i]), dt, speed, start_position=way_points[i - 1])

        end_effector.add_to_motion_path(motion_path)
