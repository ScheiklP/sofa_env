import numpy as np

from pathlib import Path
from typing import Tuple, Optional, Union, Callable, List

import Sofa.Core

from sofa_env.utils.math_helper import euler_to_rotation_matrix, rotation_matrix_to_quaternion
from sofa_env.scenes.tissue_manipulation.sofa_objects.rigidified_tissue import Tissue, TISSUE_PLUGIN_LIST
from sofa_env.sofa_templates.collision import add_collision_model, COLLISION_PLUGIN_LIST
from sofa_env.sofa_templates.rigid import ControllableRigidObject, MechanicalBinding, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.solver import add_solver, ConstraintCorrectionType, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST

from sofa_env.scenes.tissue_manipulation.sofa_robot_functions import Workspace

GRIPPER_PLUGIN_LIST = RIGID_PLUGIN_LIST + SOLVER_PLUGIN_LIST + VISUAL_PLUGIN_LIST + COLLISION_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST + TISSUE_PLUGIN_LIST + SOLVER_PLUGIN_LIST


class AttachedGripper(Sofa.Core.Controller, ControllableRigidObject):
    """Python object that creates SOFA objects and python logic to represent a gripper.

    Notes:
        - ``AttachedGripper`` can only be used with a rigidified tissue (deformable + rigid parts).
        - The gripper will follow the movement of the rigidified part of the tissue.
        - AttachProjectiveConstraint can be deactivated in order to release the grasp of the tissue.

    Args:
        parent_node (Sofa.Core.Node): Parent node of the object.
        name (str): Name of the object.
        rigidified_tissue (Tissue): Python Tissue Object.
        total_mass (float): Total mass of the deformable object.
        pose (Union[Tuple[float, float, float, float, float, float, float], np.ndarray]): 6D pose of the object described as Cartesian position and quaternion.
        position (Tuple[float, float, float]): 3D postion of the object in scene coordinates.
        orientation (Tuple[float, float, float]): Orientation of the object in euler angles ZYX.
        visual_mesh_path (Optional[Union[str, Path]]): Path to the visual surface mesh.
        collision_mesh_path (Optional[Union[str, Path]]): Path to the collision surface mesh.
        scale (float): Scale factor for loading the meshes.
         add_solver_func (Callable): Function that adds the numeric solvers to the object.
        add_collision_model_func (Callable): Function that defines how the collision surface from ``collision_mesh_path`` is added to the object.
        add_visual_model_func (Callable): Function that defines how the visual surface from ``visual_mesh_path`` is added to the object.
        animation_loop_type (AnimationLoopType): The animation loop of the scene. Required to determine if objects for constraint correction should be added.
        show_grasping_point (bool): Whether to render the grasping point node.
        randomize_grasping_point (bool): Whether to randomize grasping position. See ``Workspace.gripper_offset_limits``.
        grasping_active (bool): Whether the grasp is initially activated. ``AttachProjectiveConstraint == "1"``.
        visual_offset (Union[Tuple[float, float, float, float, float, float, float], np.ndarray])
        workspace (Workspace): Python Workspace Object.
        show_object (bool): Whether to render the nodes.

    Examples:
        >>> tissue = scene_node.addObject(Tissue(...))
        >>> rigidified_tissue = rigidify(tissue.deformable_object, rigification_indices=[...])
        >>> gripper = scene_node.addObject(AttachedGripper(rigidified_tissue=rigidified_tissue, ...))
        >>> gripper.reset()
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        rigidified_tissue: Tissue,
        total_mass: float,
        pose: Union[Tuple[float, float, float, float, float, float, float], np.ndarray, None] = None,
        position: Optional[Tuple[float, float, float]] = None,
        orientation: Optional[Tuple[float, float, float]] = None,
        visual_mesh_path: Optional[Union[str, Path]] = None,
        collision_mesh_path: Optional[Union[str, Path]] = None,
        scale: float = 1.0,
        add_solver_func: Callable = add_solver,
        add_collision_model_func: Callable = add_collision_model,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        show_grasping_point: bool = False,
        randomize_grasping_point: bool = False,
        grasping_active: bool = True,
        visual_offset: Union[Tuple[float, float, float, float, float, float, float], np.ndarray, None] = None,
        workspace: Optional[Workspace] = None,
        show_object: bool = False,
    ) -> None:
        Sofa.Core.Controller.__init__(self)

        # Overwrite pose
        if position and orientation:
            self.Rot = euler_to_rotation_matrix(orientation)

            new_pos = self.Rot @ position
            pose = tuple(new_pos.tolist()) + tuple(rotation_matrix_to_quaternion(self.Rot).tolist())

        if pose is None:
            raise ValueError("'pose' OR 'position' AND 'orientation' OR 'randomize_grasping_point' AND 'orientation' argument are required")
        self.initial_pose = pose

        ControllableRigidObject.__init__(
            self,
            parent_node=parent_node,
            name=name + "_object",
            total_mass=total_mass,
            pose=pose,
            visual_mesh_path=visual_mesh_path,
            collision_mesh_path=collision_mesh_path,
            scale=scale,
            add_solver_func=add_solver_func,
            add_collision_model_func=add_collision_model_func,
            add_visual_model_func=add_visual_model_func,
            animation_loop_type=animation_loop_type,
            show_object=show_object,
            mechanical_binding=MechanicalBinding.SPRING,
        )

        self.name = name
        self.is_attached = False
        self.grasping_force_field = None
        self.grasping_active = grasping_active
        self.grasping_constraint: Sofa.Core.Object
        self.grasping_offset = None
        self.visual_offset = visual_offset if visual_offset is not None else np.asarray([0.0, -1.0 * 1e-3, 0.0] + [0.0] * 4)
        self.rigidified_tissue = rigidified_tissue
        self.rigidified_tissue_motion_target_mechanical_object: Sofa.Core.Object
        self.randomize_grasping_point = randomize_grasping_point
        self.show_grasping_point = show_grasping_point

        self.motion_path = []
        self._workspace = workspace

        # Set a default random number generator in init (for scene creation) that can be overwritten from the env with the seed method
        self.rng = np.random.default_rng()

        # Setup motion target in rigidified tissue node
        self._setup_grasping_motion_target_on_tissue()
        self._attach_gripper_at_tissue()

    def add_to_motion_path(self, motion_path: List[np.ndarray]) -> None:
        """Adds a list of points to the Gripper's motion path.

        Motion path will be executed by iterating through the list triggered by the ``onAnimateEndEvent``.
        """

        self.motion_path.extend(motion_path)

    def _attach_gripper_at_tissue(
        self,
    ) -> None:
        """Adds an ``AttachProjectiveConstraint`` to fix the gripper to the tissue."""
        # AttachProjectiveConstraint gripper -> tissue (OK - but does not work with Motion Target on gripper -> moved to tissue)
        # AttachProjectiveConstraint tissue -> gripper (blows up)
        self.grasping_constraint = self.physical_body_node.addObject(
            "AttachProjectiveConstraint",
            name="GraspingConstraint",
            object1=self.rigidified_tissue_grasping_point.getLinkPath(),
            object2=self.physical_body_mechanical_object.getLinkPath(),
            indices1=[0],
            indices2=[0],
            constraintFactor=1 if self.grasping_active else 0,
            twoWay=False,
        )
        self.is_attached = True if self.grasping_active else False

    def clear_motion_path(self) -> None:
        """Removes all points in the Gripper's motion path."""

        self.motion_path = []

    def create_linear_motion(self, target_position: np.ndarray, dt: float, velocity: float, single_step: bool = False) -> List[np.ndarray]:
        """Creates movement path to displace Gripper from its current position to the final position provided, at the given velocity."""
        if self.is_attached:
            current_position = self.rigidified_tissue_motion_target_mechanical_object.position.array()[0]
        else:
            print("[Warning] AttachedGripper is currently not attached -> Assert that AttachProjectiveConstraint is active! Use gripper.set_attach_constraint_to(True)")
            current_position = self.motion_target_mechanical_object.position.array()[0]

        # same orientation for all steps
        current_rotation = current_position[3:]
        current_position = current_position[:3]
        displacement = target_position - current_position

        if single_step:
            motion_path = [np.append(target_position, current_rotation)]
            motion_steps = 1
        else:
            displacement_per_step = velocity * dt
            motion_steps = int(np.ceil(np.linalg.norm(displacement) / displacement_per_step))

            progress = np.linspace(0.0, 1.0, motion_steps + 1)[1:]
            motion_path = current_position + displacement * progress[:, np.newaxis]

            motion_path[-1] = target_position
            # add current rotation to all positions
            motion_path = np.hstack([motion_path, np.asarray([current_rotation] * motion_path.shape[0])])

        return np.split(motion_path, motion_steps, axis=0)

    def get_pose(self) -> np.ndarray:
        """Reads the Rigid3d pose from the Gripper as [x, y, z, a, b, c, w]."""

        if self.is_attached:
            return self.rigidified_tissue_motion_target_mechanical_object.position.array()[0]

        return self.get_physical_pose()

    def onAnimateBeginEvent(self, _) -> None:
        # check if there is a motion path and execute one step
        if len(self.motion_path):
            new_position = self.motion_path.pop(0)
            self.set_pose(new_position)

    def reset(self) -> None:
        """Reset the grasping state of the gripper and set it to it's initial pose.

        Notes:
        - Also calculate position for random grasp.
        """

        if self.randomize_grasping_point:
            # sample new grasping offset
            x_lim, _, z_lim = self._workspace.get_randomize_gripper_limits()
            if (x_lim == (0, 0) or x_lim is None) and (z_lim == (0, 0) or z_lim is None):
                raise ValueError(f"Create scene argument: randomize_grasping_point == True, " f"but grasping offset limits are Zero. {x_lim=}, {z_lim=}")

            self.grasping_offset = self._sample_random_grasping_offset(x_lim=x_lim, z_lim=z_lim)

        if not self.is_attached:
            self.set_pose(np.asarray(self.initial_pose))

    def _sample_random_grasping_offset(
        self,
        x_lim: Tuple = (0.0, 0.0),
        y_lim: Optional[Tuple] = None,
        z_lim: Tuple = (0.0, 0.0),
        scale: float = 1e-3,
    ) -> np.ndarray:
        """Sample random grasping point from phantom/tissue"""
        x_off = (self.rng.random(1) * (x_lim[1] - x_lim[0]) + x_lim[0])[0] if x_lim is not None else 0.0
        y_off = (self.rng.random(1) * (y_lim[1] - y_lim[0]) + y_lim[0])[0] if y_lim is not None else 0.0
        z_off = (self.rng.random(1) * (z_lim[1] - z_lim[0]) + z_lim[0])[0] if z_lim is not None else 0.0

        ret = x_off * scale, y_off * scale, z_off * scale, 0.0, 0.0, 0.0, 0.0

        return np.asarray(ret)

    def set_attach_constraint_to(self, value: bool) -> None:
        """Activate and deactivate AttachProjectiveConstraint (self.grasping_constraint)"""
        # Align rigid reference frame orientation with gripper orientation
        with self.grasping_constraint.constraintFactor.writeable() as factors:
            factors[0] = "1" if value else "0"

    def _setup_grasping_motion_target_on_tissue(self) -> None:
        """Adds MotionTarget to rigidified tissue, add UncoupledConstraintCorrection and RestShapeForceField.

        Notes:
            - Also adds AttachmentNode with MechanicalObject as GraspingPoint to rigidified tissue.
        """
        rigidified_tissue_motion_target = self.rigidified_tissue.addChild("rigid_motion_target")
        self.rigidified_tissue_motion_target_mechanical_object = rigidified_tissue_motion_target.addObject(
            "MechanicalObject",
            name="MotionTarget",
            template="Rigid3d",
            position=self.rigidified_tissue.rigid.MechanicalObject.position.array().tolist(),
            showObjectScale=0.005,
        )
        rigidified_tissue_motion_target.addObject(
            ConstraintCorrectionType.UNCOUPLED.value,
            compliance=1.0,
        )

        self.rigidified_tissue.rigid.addObject(
            "RestShapeSpringsForceField",
            stiffness=2e6,  # increase this if the body drags behind the target while moving
            angularStiffness=2e4,  # increase this if there is a rotational offset between body and target
            external_rest_shape=rigidified_tissue_motion_target.getLinkPath(),
        )

        # Add attachment node to attach gripper to -> node is used to allow offset to motion target on attachment
        rigid_attachment_node = rigidified_tissue_motion_target.addChild("attachment_node")
        self.rigidified_tissue_grasping_point = rigid_attachment_node.addObject(
            "MechanicalObject",
            name="GraspingPoint",
            template="Rigid3d",
            position=self.rigidified_tissue.rigid.MechanicalObject.position.array().tolist(),
            showObjectScale=0.005,
        )

    def set_pose(self, pose: np.ndarray, validate: bool = False) -> None:
        """Writes the Rigid3d pose to the Gripper as [x, y, z, a, b, c, w] (position and quaternion)."""
        pose = pose.reshape(
            -1,
        )

        if self.is_attached:
            current_pose = self.rigidified_tissue_motion_target_mechanical_object.position.array()[0]
        else:
            current_pose = self.get_pose()

        # create mask for invalid poses
        invalid_poses_mask = [False] * len(current_pose)

        # check if there are any poses that are outside the workspace if validate == True
        if self._workspace is not None and validate:
            dim = self._workspace.mode.value[-1]
            invalid_poses_mask[:dim] = (self._workspace.get_low() > pose[:dim]) | (pose[:dim] > self._workspace.get_high())

        # overwrite the invalid parts of the pose with the current pose
        pose[invalid_poses_mask] = current_pose[invalid_poses_mask]

        if self.is_attached:
            # set position of the gripper through the rigidified tissue
            with self.rigidified_tissue_motion_target_mechanical_object.position.writeable() as state:
                state[:] = pose

            # account for grasping point offset
            if self.grasping_offset is not None:
                new_pose = pose + self.grasping_offset + self.visual_offset
            else:
                new_pose = pose
            with self.rigidified_tissue_grasping_point.position.writeable() as state:
                state[:] = new_pose

        else:
            # if attach is deactivated -> set pose directly
            self.rigid_object.set_pose(pose)

    def set_position(self, position: Optional[np.ndarray] = None) -> Union[np.ndarray, None]:
        """Sets gripper position (3D, (3,)-array)"""
        if position is None:
            return

        new_pose = np.array(tuple(position) + tuple(self.get_pose()[3:]))
        self.set_pose(new_pose)

        return new_pose[:3]

    def seed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Creates a random number generator from a seed."""
        self.rng = np.random.default_rng(seed)
