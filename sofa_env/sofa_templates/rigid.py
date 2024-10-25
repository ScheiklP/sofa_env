import Sofa.Core
import numpy as np
from enum import Enum, unique
from typing import Callable, Union, Tuple, Optional, List, Dict
from pathlib import Path

from sofa_env.sofa_templates.mappings import MappingType, MAPPING_PLUGIN_LIST
from sofa_env.sofa_templates.motion_restriction import add_bounding_box
from sofa_env.sofa_templates.solver import add_solver, ConstraintCorrectionType, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST
from sofa_env.sofa_templates.collision import add_collision_model, COLLISION_PLUGIN_LIST, is_default_collision_model_function, match_collision_model_kwargs
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST

from sofa_env.utils.pivot_transform import generate_ptsd_to_pose

RIGID_PLUGIN_LIST = (
    [
        "Sofa.Component.Mass",  # <- [UniformMass]
        "Sofa.Component.StateContainer",  # <- [MechanicalObject]
        "Sofa.Component.Constraint.Lagrangian.Correction",  # <- [UncoupledConstraintCorrection]
        "ArticulatedSystemPlugin",  # <- [ArticulatedHierarchyContainer, ArticulatedSystemMapping, Articulation, ArticulationCenter]
        "Sofa.Component.Constraint.Lagrangian.Model",  # <- [StopperLagrangianConstraint]
        "Sofa.Component.Constraint.Projective",  # <- [AttachProjectiveConstraint]
        "Sofa.Component.Mapping.NonLinear",  # <- [RigidMapping]
        "Sofa.Component.SolidMechanics.Spring",  # <- [RestShapeSpringsForceField]
        "Sofa.Component.Engine.Select",  # <- [BoxROI]
    ]
    + MAPPING_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
    + COLLISION_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
)


class RigidObject:
    """Combines all the sofa components to describe a rigid object.

    Notes:
        Parameterizable components such as solvers and collision models are added through add functions (e.g. ``add_solver``).
        This way we avoid having a large number of parameters in the init function where no one remebers which parameter influences which component.
        To change the parameters of a function simply make a partial version of it or write a new one.

    Args:
        parent_node (Sofa.Core.Node): Parent node of the object.
        name (str): Name of the object.
        visual_mesh_path (Optional[Union[str, Path]]): Path to the visual surface mesh.
        collision_mesh_path (Optional[Union[str, Path]]): Path to the collision surface mesh.
        pose (Union[Tuple[float, float, float, float, float, float, float], np.ndarray]): 6D pose of the object described as Cartesian position and quaternion.
        fixed_position (bool): Whether to add a ``"FixedProjectiveConstraint"`` to hold the object in place with its original position.
        fixed_orientation (bool): Whether to add a ``"FixedRotationConstraint"`` to hold the object in place with its original orientation.
        total_mass (float): Total mass of the deformable object.
        scale: (Union[float, Tuple[float, float, float]]): Scale factor for loading the meshes.
        add_solver_func (Callable): Function that adds the numeric solvers to the object.
        add_collision_model_func (Callable): Function that defines how the collision surface from ``collision_mesh_path`` is added to the object.
        add_visual_model_func (Callable): Function that defines how the visual surface from ``visual_mesh_path`` is added to the object.
        animation_loop_type (AnimationLoopType): The animation loop of the scene. Required to determine if objects for constraint correction should be added.
        is_carving_tool (bool): If set to True, will add a ``"CarvingTool"`` tag to the collision models. Requires the SofaCarving plugin to be compiled.
        show_object (bool): Whether to render the nodes.
        show_object_scale (float): Render size of the node if show_object is True.

    Examples:
        Changing the parameterizable functions with partial

        >>> from sofa_env.sofa_templates.solver import add_solver
        >>> from functools import partial
        >>> add_solver = partial(add_solver, rayleigh_mass=0.02)
        >>> rigid = RigidObject(..., add_solver_func=add_solver)
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        pose: Union[Tuple[float, float, float, float, float, float, float], np.ndarray],
        fixed_position: bool = False,
        fixed_orientation: bool = False,
        total_mass: Optional[float] = None,
        visual_mesh_path: Optional[Union[str, Path]] = None,
        collision_mesh_path: Optional[Union[str, Path]] = None,
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        add_solver_func: Callable = add_solver,
        add_collision_model_func: Callable = add_collision_model,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        is_carving_tool: bool = False,
        show_object: bool = False,
        show_object_scale: float = 7.0,
    ) -> None:

        self.parent_node = parent_node
        self.name = name
        self.node = self.parent_node.addChild(name)

        # Add the solvers
        self.time_integration, self.linear_solver = add_solver_func(attached_to=self.node)

        # Add the mechanical object that holds the mechanical state of the rigid object
        self.mechanical_object = self.node.addObject(
            "MechanicalObject",
            template="Rigid3d",
            position=pose,
            showObject=show_object,
            showObjectScale=show_object_scale,
        )

        # Add a FixedProjectiveConstraint to hold the object in its initial pose
        if fixed_position:
            self.node.addObject("FixedProjectiveConstraint", template="Rigid3d", fixAll=True)

        if fixed_orientation:
            self.node.addObject(
                "FixedRotationConstraint",
                template="Rigid3d",
                FixedXRotation=True,
                FixedYRotation=True,
                FixedZRotation=True,
            )

        # Add mass to the object
        if total_mass is not None:
            self.mass = self.node.addObject("UniformMass", totalMass=total_mass)

        # If using the FreeMotionAnimationLoop add a constraint correction object
        if animation_loop_type == AnimationLoopType.FREEMOTION:
            self.constraint_correction = self.node.addObject(
                ConstraintCorrectionType.UNCOUPLED.value,
                compliance=1.0 / total_mass if total_mass is not None else 1.0,
            )

        # Add collision models to the object
        if not is_default_collision_model_function(add_collision_model_func):
            # If a custom function was passed to the init, check if there are local variables that should be passed to the function
            function_kwargs = match_collision_model_kwargs(add_collision_model_func, locals=locals())
            self.collision_model_node = add_collision_model_func(attached_to=self.node, **function_kwargs)
        else:
            if collision_mesh_path is not None:
                self.collision_model_node = add_collision_model_func(
                    attached_to=self.node,
                    surface_mesh_file_path=collision_mesh_path,
                    scale=scale,
                    mapping_type=MappingType.RIGID,
                    is_carving_tool=is_carving_tool,
                )

        # Add a visual model to the object
        if visual_mesh_path is not None:
            self.visual_model_node = add_visual_model_func(
                attached_to=self.node,
                surface_mesh_file_path=visual_mesh_path,
                scale=scale,
                mapping_type=MappingType.RIGID,
            )


@unique
class MechanicalBinding(Enum):
    """Enum used in ControllableRigidObject to define how motion target and physical body are attached to each other."""

    ATTACH = 0
    SPRING = 1


class ControllableRigidObject:
    """Combines all the sofa components to describe a rigid object that can be controlled by updating its pose.

    Notes:
        Nodes are separeted into a controllable part without collision models (target) and a non-controllable part that follows the target via a ``"RestShapeSpringsForceField"`` or ``"AttachProjectiveConstraint"`` (controlled by ``mechanical_binding``).
        Separation is necessary to correctly resolve large motions between time steps that would otherwise lead to unresolvable collision.

    Args:
        parent_node (Sofa.Core.Node): Parent node of the object.
        name (str): Name of the object.
        pose (Union[Tuple[float, float, float, float, float, float, float], np.ndarray]): 6D pose of the object described as Cartesian position and quaternion.
        total_mass (float): Total mass of the deformable object.
        visual_mesh_path (Optional[Union[str, Path]]): Path to the visual surface mesh.
        collision_mesh_path (Optional[Union[str, Path]]): Path to the collision surface mesh.
        scale (Union[float, Tuple[float, float, float]]): Scale factor for loading the meshes.
        add_solver_func (Callable): Function that adds the numeric solvers to the object.
        add_collision_model_func (Callable): Function that defines how the collision surface from ``collision_mesh_path`` is added to the object.
        add_visual_model_func (Callable): Function that defines how the visual surface from ``visual_mesh_path`` is added to the object.
        animation_loop_type (AnimationLoopType): The animation loop of the scene. Required to determine if objects for constraint correction should be added.
        is_carving_tool (bool): If set to True, will add a ``"CarvingTool"`` tag to the collision models. Requires the SofaCarving plugin to be compiled.
        show_object (bool): Whether to render the nodes.
        show_object_scale (float): Render size of the node if ``show_object`` is ``True``.
        mechanical_binding (MechanicalBinding): Whether to use ``"RestShapeSpringsForceField"`` or ``"AttachProjectiveConstraint"`` to combine controllable and non-controllable part of the object.
        spring_stiffness (Optional[float]): Spring stiffness of the ``"RestShapeSpringsForceField"``.
        angular_spring_stiffness (Optional[float]): Angular spring stiffness of the ``"RestShapeSpringsForceField"``.
        collision_group (int): The group for which collisions with this object should be ignored. Value has to be set since the jaws and shaft must belong to the same group.

    Examples:
        - Changing the parameterizable functions with partial

        >>> from sofa_env.sofa_templates.solver import add_solver
        >>> from functools import partial
        >>> add_solver = partial(add_solver, rayleigh_mass=0.02)
        >>> rigid = ControllableRigidObject(..., add_solver_func=add_solver)

        - Setting a new pose

        >>> import numpy as np
        >>> rigid = ControllableRigidObject(...)
        >>> new_pose = np.array([0, 5, 0, 0, 0, 0, 1])
        >>> rigid.set_pose(new_pose)

    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        pose: Union[Tuple[float, float, float, float, float, float, float], np.ndarray],
        total_mass: Optional[float] = None,
        visual_mesh_path: Optional[Union[str, Path]] = None,
        collision_mesh_path: Optional[Union[str, Path]] = None,
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        add_solver_func: Callable = add_solver,
        add_collision_model_func: Callable = add_collision_model,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        is_carving_tool: bool = False,
        show_object: bool = False,
        show_object_scale: float = 1.0,
        mechanical_binding: MechanicalBinding = MechanicalBinding.ATTACH,
        spring_stiffness: Optional[float] = 2e10,
        angular_spring_stiffness: Optional[float] = 2e10,
        collision_group: Optional[int] = None,
    ) -> None:

        self.parent_node = parent_node
        self.node = self.parent_node.addChild(name)

        # Add the solvers
        self.time_integration, self.linear_solver = add_solver_func(attached_to=self.node)

        # Add controllable part without collision or visual model
        self.motion_target_node = self.node.addChild(f"{name}_motion_target")
        self.motion_target_mechanical_object = self.motion_target_node.addObject(
            "MechanicalObject",
            template="Rigid3d",
            position=pose,
            showObject=show_object,
            showObjectScale=show_object_scale,
        )

        # Add non-controllable part with collision and visual model
        self.physical_body_node = self.node.addChild(f"{name}_physical_body")
        self.physical_body_mechanical_object = self.physical_body_node.addObject(
            "MechanicalObject",
            template="Rigid3d",
            position=pose,
            showObject=show_object,
            showObjectScale=show_object_scale,
        )

        if total_mass is not None:
            self.physical_body_node.addObject("UniformMass", totalMass=total_mass)

        if mechanical_binding == MechanicalBinding.ATTACH:
            # Introduce an AttachProjectiveConstraint to bind the position of the motion target to the body
            self.node.addObject(
                "AttachProjectiveConstraint",
                object1=self.motion_target_mechanical_object.getLinkPath(),
                object2=self.physical_body_mechanical_object.getLinkPath(),
                indices1=[0],
                indices2=[0],
                twoWay=False,
            )
        else:
            # Bind the rest shape of the body to the motion target via springs
            if spring_stiffness is None or angular_spring_stiffness is None:
                raise ValueError("When using springs to attatch motion target to mechanical body, please pass values for spring_stiffness and angular_spring_stiffness.")
            self.physical_body_node.addObject(
                "RestShapeSpringsForceField",
                stiffness=spring_stiffness,  # Increase this if the body trags behind the target while moving
                angularStiffness=angular_spring_stiffness,  # Increase this if there is a rotational offset between body and target
                external_rest_shape=self.motion_target_mechanical_object.getLinkPath(),
            )

        # If using the FreeMotionAnimationLoop add a constraint correction object
        if animation_loop_type == AnimationLoopType.FREEMOTION:
            self.physical_body_node.addObject(
                ConstraintCorrectionType.UNCOUPLED.value,
                compliance=1.0 / total_mass if total_mass is not None else 1.0,
            )

        if not is_default_collision_model_function(add_collision_model_func):
            # If a custom function was passed to the init, check if there are local variables that should be passed to the function
            function_kwargs = match_collision_model_kwargs(add_collision_model_func, locals=locals())
            self.collision_model_node = add_collision_model_func(attached_to=self.physical_body_node, **function_kwargs)
        else:
            if collision_mesh_path is not None:
                self.collision_model_node = add_collision_model_func(
                    attached_to=self.physical_body_node,
                    surface_mesh_file_path=collision_mesh_path,
                    scale=scale,
                    mapping_type=MappingType.RIGID,
                    is_carving_tool=is_carving_tool,
                    collision_group=collision_group,
                )

        # Add a visual model to the object
        if visual_mesh_path is not None:
            self.visual_model_node = add_visual_model_func(
                attached_to=self.physical_body_node,
                surface_mesh_file_path=visual_mesh_path,
                scale=scale,
                mapping_type=MappingType.RIGID,
            )

    def get_pose(self) -> np.ndarray:
        """Reads the Rigid3d pose from the controllable sofa node and returns it as [x, y, z, a, b, c, w].

        Notes:
            - returned array is read-only
        """
        return self.motion_target_mechanical_object.position.array()[0]

    def get_physical_pose(self) -> np.ndarray:
        """Reads the Rigid3d pose from the non-controllable sofa node and returns it as [x, y, z, a, b, c, w].

        Notes:
            - returned array is read-only
        """
        return self.physical_body_mechanical_object.position.array()[0]

    def set_pose(self, pose: np.ndarray) -> None:
        """Writes the Rigid3d pose from the controllable sofa node as [x, y, z, a, b, c, w].

        Notes:
            - pose values are written into the sofa array without assiging the pose array to the sofa array.
              Changes in the pose array after that will not be propagated to sofa.
        """
        with self.motion_target_mechanical_object.position.writeable() as sofa_pose:
            sofa_pose[:] = pose

    def set_rest_pose(self, pose: np.ndarray) -> None:
        """Writes the Rigid3d rest pose from the controllable sofa node as [x, y, z, a, b, c, w].

        Notes:
            - pose values are written into the sofa array without assiging the pose array to the sofa array.
              Changes in the pose array after that will not be propagated to sofa.
        """
        with self.motion_target_mechanical_object.rest_position.writeable() as sofa_pose:
            sofa_pose[:] = pose

    def get_pose_difference(self, position_norm: bool = False) -> np.ndarray:
        """Reads the Rigid3d poses from both motion target and body and returns the difference between the two.

        Args:
            position_norm: If True, the Cartesian norm of the position difference is returned.

        Returns:
            [delta_x, delta_y, delta_z, delta_rot] if position_norm == False
            [delta_xyz_norm, delta_rot] if position_norm == True
        """

        if self.physical_body_node is None:
            raise RuntimeError("Created object without collision mesh -> has no separation between motion target and body.")

        target_pose = self.motion_target_mechanical_object.position.array()[0]
        actual_pose = self.physical_body_mechanical_object.position.array()[0]
        position_delta = target_pose[:3] - actual_pose[:3]

        orientation_delta = np.rad2deg(2 * np.arccos(np.clip(np.dot(target_pose[-4:], actual_pose[-4:]), a_min=-1.0, a_max=1.0)))  # minimal rotation angle

        if position_norm:
            position_delta = np.linalg.norm(position_delta)

        return np.append(position_delta, orientation_delta)


class PivotizedRigidObject(ControllableRigidObject):
    """A controllable rigid object that is controlled in pivotized space.

    Args:
        parent_node (Sofa.Core.Node): Parent node of the object.
        name (str): Name of the object.
        total_mass (float): Total mass of the deformable object.
        visual_mesh_path (Optional[Union[str, Path]]): Path to the visual surface mesh.
        collision_mesh_path (Optional[Union[str, Path]]): Path to the collision surface mesh.
        scale (Union[float, Tuple[float, float, float]]): Scale factor for loading the meshes.
        add_solver_func (Callable): Function that adds the numeric solvers to the object.
        add_collision_model_func (Callable): Function that defines how the collision surface from ``collision_mesh_path`` is added to the object.
        add_visual_model_func (Callable): Function that defines how the visual surface from ``visual_mesh_path`` is added to the object.
        animation_loop_type (AnimationLoopType): The animation loop of the scene. Required to determine if objects for constraint correction should be added.
        is_carving_tool (bool): If set to True, will add a ``"CarvingTool"`` tag to the collision models. Requires the SofaCarving plugin to be compiled.
        show_object (bool): Whether to render the nodes.
        show_object_scale (float): Render size of the node if ``show_object`` is ``True``.
        mechanical_binding (MechanicalBinding): Whether to use ``"RestShapeSpringsForceField"`` or ``"AttachProjectiveConstraint"`` to combine controllable and non-controllable part of the object.
        spring_stiffness (Optional[float]): Spring stiffness of the ``"RestShapeSpringsForceField"``.
        angular_spring_stiffness (Optional[float]): Angular spring stiffness of the ``"RestShapeSpringsForceField"``.
        collision_group (int): The group for which collisions with this object should be ignored. Value has to be set since the jaws and shaft must belong to the same group.
        ptsd_state (np.ndarray): Pan, tilt, spin, depth state of the pivotized tool.
        rcm_pose (np.ndarray): [x, y, z, X, Y, Z] Cartesian position and XYZ euler angles of the remote center of motion.
        cartesian_workspace (Dict): Low and high values of the instrument's Cartesian workspace.
        ptsd_reset_noise (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Noise added to the ptsd state when resetting the object.
        rcm_reset_noise (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Noise added to the rcm pose when resetting the object.
        state_limits (Dict): Low and high values of the instrument's state space.
        show_remote_center_of_motion (bool): Whether to render the remote center of motion.
        show_workspace (bool): Whether to render the workspace.
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        total_mass: Optional[float] = None,
        visual_mesh_path: Optional[Union[str, Path]] = None,
        collision_mesh_path: Optional[Union[str, Path]] = None,
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        add_solver_func: Callable = add_solver,
        add_collision_model_func: Callable = add_collision_model,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        is_carving_tool: bool = False,
        show_object: bool = False,
        show_object_scale: float = 1.0,
        mechanical_binding: MechanicalBinding = MechanicalBinding.ATTACH,
        spring_stiffness: Optional[float] = 2e10,
        angular_spring_stiffness: Optional[float] = 2e10,
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
            "high": np.array([90, 90, 90, 200]),
        },
        show_remote_center_of_motion: bool = False,
        show_workspace: bool = False,
    ) -> None:

        if not isinstance(rcm_pose, np.ndarray) and not rcm_pose.shape == (6,):
            raise ValueError(f"Please pass the pose of the remote center of motion (rcm_pose) as a numpy array with [x, y, z, X, Y, Z] (position, rotation). Received {rcm_pose}.")
        self.remote_center_of_motion = rcm_pose.copy()
        self.pivot_transform = generate_ptsd_to_pose(rcm_pose=self.remote_center_of_motion)

        if not isinstance(ptsd_state, np.ndarray) and not ptsd_state.shape == (4,):
            raise ValueError(f"Please pass the instruments state as a numpy array with [pan, tilt, spin, depth]. Received {ptsd_state}.")
        self.ptsd_state = ptsd_state

        self.initial_state = np.copy(self.ptsd_state)
        self.initial_pose = self.pivot_transform(self.initial_state)
        self.initial_remote_center_of_motion = rcm_pose.copy()

        self.cartesian_workspace = cartesian_workspace
        self.state_limits = state_limits

        self.last_set_state_violated_state_limits = False
        self.last_set_state_violated_workspace_limits = False

        self.ptsd_reset_noise = ptsd_reset_noise
        self.rcm_reset_noise = rcm_reset_noise

        super().__init__(
            parent_node=parent_node,
            name=name,
            pose=self.initial_pose,
            total_mass=total_mass,
            visual_mesh_path=visual_mesh_path,
            collision_mesh_path=collision_mesh_path,
            scale=scale,
            add_solver_func=add_solver_func,
            add_collision_model_func=add_collision_model_func,
            add_visual_model_func=add_visual_model_func,
            animation_loop_type=animation_loop_type,
            is_carving_tool=is_carving_tool,
            show_object=show_object,
            show_object_scale=show_object_scale,
            mechanical_binding=mechanical_binding,
            spring_stiffness=spring_stiffness,
            angular_spring_stiffness=angular_spring_stiffness,
            collision_group=collision_group,
        )

        self.show_remote_center_of_motion = show_remote_center_of_motion
        if show_remote_center_of_motion:
            self.rcm_node = self.parent_node.addChild(f"rcm_{name}")
            self.rcm_mechanical_object = self.rcm_node.addObject(
                "MechanicalObject",
                template="Rigid3d",
                position=self.pivot_transform((0, 0, 0, 0)),
                showObject=True,
                showObjectScale=show_object_scale,
            )

        self.show_workspace = show_workspace
        if show_workspace:
            self.workspace_node = self.parent_node.addChild(f"workspace_{name}")
            self.workspace_mechanical_object = self.workspace_node.addObject("MechanicalObject")
            add_bounding_box(self.workspace_node, min=cartesian_workspace["low"], max=cartesian_workspace["high"], show_bounding_box=True, show_bounding_box_scale=show_object_scale)

    def get_state(self) -> np.ndarray:
        """Gets the current state of the instrument."""
        read_only_state = self.ptsd_state.view()
        read_only_state.flags.writeable = False
        return read_only_state

    def set_state(self, state: np.ndarray) -> None:
        """Sets the state of the instrument withing the defined state limits and Cartesian workspace.

        Warning:
            The components of a Cartesian pose are not independently changeable, since this object has a remote center of motion.
            We thus cannton simple ignore one part (e.g. the x component) and still write the other components (e.g. y).
            Poses that are not validly withing the workspace will be ignored.
            The state, however, is independently constrainable so only invalid components (e.g. tilt) will be ignored.
        """

        # Check if there are any states that are outside the state limits
        invalid_states_mask = (self.state_limits["low"] > state) | (state > self.state_limits["high"])

        # Overwrite the invalide parts of the states with the current state
        state[invalid_states_mask] = self.ptsd_state[invalid_states_mask]

        # Get the corresponding pose from the state
        pose = self.pivot_transform(state)

        # Save info about violation of state limits
        self.last_set_state_violated_state_limits = np.any(invalid_states_mask)

        # Only set the pose, if all components are within the Cartesian workspace
        if not np.any((self.cartesian_workspace["low"] > pose[:3]) | (pose[:3] > self.cartesian_workspace["high"])):
            self.set_pose(pose)

            # Only overwrite the internal value of ptsd_state, if that was successful
            self.ptsd_state[:] = state

            # Save info about violation of Cartesian workspace limits
            self.last_set_state_violated_workspace_limits = False
        else:
            self.last_set_state_violated_workspace_limits = True

    def reset_state(self) -> None:
        """Resets the object to its initial ptsd state. Optionally adds noise to rcm pose and ptsd state."""

        # Generate a new pivot_transform by adding noise to the initial remote_center_of_motion pose
        if self.rcm_reset_noise is not None:
            if isinstance(self.rcm_reset_noise, np.ndarray):
                # Uniformly sample from -noise to +noise and add it to the rcm pose
                new_rcm_pose = self.initial_remote_center_of_motion + self.rng.uniform(-self.rcm_reset_noise, self.rcm_reset_noise)
            elif isinstance(self.rcm_reset_noise, dict):
                # Uniformly sample from low to high and add it to the rcm pose
                new_rcm_pose = self.initial_remote_center_of_motion + self.rng.uniform(self.rcm_reset_noise["low"], self.rcm_reset_noise["high"])
            else:
                raise TypeError("Please pass the rcm_reset_noise as a numpy array or a dictionary with 'low' and 'high' keys.")

            self.pivot_transform = generate_ptsd_to_pose(new_rcm_pose)
            self.remote_center_of_motion[:] = new_rcm_pose
            if self.show_remote_center_of_motion:
                with self.rcm_mechanical_object.position.writeable() as rcm_pose:
                    rcm_pose[:] = self.pivot_transform((0, 0, 0, 0))

        # Select a new ptsd state by adding noise to the initial state
        if self.ptsd_reset_noise is not None:
            if isinstance(self.ptsd_reset_noise, np.ndarray):
                # Uniformly sample from -noise to +noise and add it to the initial state
                new_state = self.initial_state + self.rng.uniform(-self.ptsd_reset_noise, self.ptsd_reset_noise)
            elif isinstance(self.ptsd_reset_noise, dict):
                # Uniformly sample from low to high and add it to the initial state
                new_state = self.initial_state + self.rng.uniform(self.ptsd_reset_noise["low"], self.ptsd_reset_noise["high"])
            else:
                raise TypeError("Please pass the ptsd_reset_noise as a numpy array or a dictionary with 'low' and 'high' keys.")

            # Do that until a pose is found that fits in the Cartesian workspace and the state limits
            while np.any((self.state_limits["low"] > new_state) | (new_state > self.state_limits["high"])) or np.any((self.cartesian_workspace["low"] > self.pivot_transform(new_state)[:3]) | (self.pivot_transform(new_state)[:3] > self.cartesian_workspace["high"])):
                if isinstance(self.ptsd_reset_noise, np.ndarray):
                    # Uniformly sample from -noise to +noise and add it to the initial state
                    new_state = self.initial_state + self.rng.uniform(-self.ptsd_reset_noise, self.ptsd_reset_noise)
                else:
                    # Uniformly sample from low to high and add it to the initial state
                    new_state = self.initial_state + self.rng.uniform(self.ptsd_reset_noise["low"], self.ptsd_reset_noise["high"])
        else:
            new_state = self.initial_state

        self.set_state(new_state)

    def get_rcm_position(self) -> np.ndarray:
        """Returns the position of the remote center of motion

        Returns:
            rcm_position (np.ndarray): XYZ position of the remote center of motion
        """
        return self.remote_center_of_motion[:3]

    def seed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Creates a random number generator from a seed."""
        self.rng = np.random.default_rng(seed)


class ArticulatedInstrument:
    """Combines all the sofa components to describe a rigid object that can be controlled by updating its pose and features one or two controllable jaws.

    Notes:
        Nodes are separeted into a controllable part without collision models (target) and a non-controllable part that follows the target via a ``"RestShapeSpringsForceField"`` or ``"AttachProjectiveConstraint"`` (controlled by ``mechanical_binding``).
        Separation is necessary to correctly resolve large motions between time steps that would otherwise lead to unresolvable collision.

    Args:
        parent_node (Sofa.Core.Node): Parent node of the object.
        name (str): Name of the object.
        pose (Union[Tuple[float, float, float, float, float, float, float], np.ndarray]): 6D pose of the object described as Cartesian position and quaternion.
        visual_mesh_path_shaft (Union[str, Path]): Path to the visual surface mesh of the instrument's shaft.
        visual_mesh_paths_jaws (List[Union[str, Path]]): A list of paths to the visual surface meshes of the instrument's jaws. Can be one or two, depending on ``two_jaws``.
        angle (float): Initial opening angle of the jaws in degrees.
        angle_limits (Union[Tuple[float, float], Dict[str, float]]): Minimum and maximum angle value in degrees.
        total_mass (float): Total mass of the deformable object.
        two_jaws (bool): Whether the instrument has two or one jaws.
        collision_mesh_path_shaft (Optional[Union[str, Path]]): Path to the collision surface mesh of the instrument's shaft.
        collision_mesh_paths_jaws (Optional[List[Union[str, Path]]]): A list of paths to the collision surface meshes of the instrument's jaws. Can be one or two, depending on ``two_jaws``.
        rotation_axis (Tuple[int, int, int]): Which axis of the jaw models to use for rotation. Only one of them should be set to 1. The rest should be set to 0.
        scale: (Union[float, Tuple[float, float, float]]): Scale factor for loading the meshes.
        add_solver_func (Callable): Function that adds the numeric solvers to the object.
        add_shaft_collision_model_func (Callable): Function that defines how the collision surface from ``collision_mesh_path_shaft`` is added to the object.
        add_jaws_collision_model_func (Callable): Function that defines how the collision surfaces from ``collision_mesh_paths_jaws`` are added to the object.
        add_visual_model_func (Callable): Function that defines how the visual surfaces from ``visual_mesh_path_shaft`` and ``visual_mesh_paths_jaws`` are added to the object.
        animation_loop_type (AnimationLoopType): The animation loop of the scene. Required to determine if objects for constraint correction should be added.
        show_object (bool): Whether to render the nodes.
        show_object_scale (float): Render size of the node if ``show_object`` is ``True``.
        mechanical_binding (MechanicalBinding): Whether to use ``"RestShapeSpringsForceField"`` or ``"AttachProjectiveConstraint"`` to combine controllable and non-controllable part of the object.
        spring_stiffness (Optional[float]): Spring stiffness of the ``"RestShapeSpringsForceField"``.
        angular_spring_stiffness (Optional[float]): Angular spring stiffness of the ``"RestShapeSpringsForceField"``.
        articulation_spring_stiffness (Optional[float]): Spring stiffness of the ``"RestShapeSpringsForceField"`` of the articulation.
        collision_group (int): The group for which collisions with this object should be ignored. Value has to be set since the jaws and shaft must belong to the same group.
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        pose: Union[Tuple[float, float, float, float, float, float, float], np.ndarray],
        angle: float = 0.0,
        angle_limits: Union[Tuple[float, float], Dict[str, float]] = {
            "low": float(np.finfo(np.float16).min),
            "high": float(np.finfo(np.float16).max),
        },
        total_mass: Optional[float] = None,
        two_jaws: bool = True,
        collision_mesh_path_shaft: Optional[Union[str, Path]] = None,
        collision_mesh_paths_jaws: Optional[List[Union[str, Path]]] = None,
        visual_mesh_path_shaft: Optional[Union[str, Path]] = None,
        visual_mesh_paths_jaws: Optional[List[Union[str, Path]]] = None,
        rotation_axis: Tuple[int, int, int] = (1, 0, 0),
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        add_solver_func: Callable = add_solver,
        add_shaft_collision_model_func: Callable = add_collision_model,
        add_jaws_collision_model_func: Union[Callable, List[Callable]] = add_collision_model,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.FREEMOTION,
        show_object: bool = False,
        show_object_scale: float = 1.0,
        mechanical_binding: MechanicalBinding = MechanicalBinding.ATTACH,
        spring_stiffness: Optional[float] = 2e10,
        angular_spring_stiffness: Optional[float] = 2e10,
        articulation_spring_stiffness: Optional[float] = 1e12,
        collision_group: int = 0,
    ) -> None:
        self.parent_node = parent_node
        self.node = self.parent_node.addChild(name)
        self.initial_pose = pose

        if not isinstance(angle_limits, Dict):
            angle_limits = {"low": angle_limits[0], "high": angle_limits[1]}
        if angle_limits["low"] > angle > angle_limits["high"]:
            raise ValueError(f"Initial {angle=} is not within {angle_limits=}.")
        self.initial_angle = angle
        self.angle_limits = angle_limits
        self.last_set_angle_violated_jaw_limits = False
        self.two_jaws = two_jaws

        # Add the solvers
        self.time_integration, self.linear_solver = add_solver_func(attached_to=self.node)

        # Add a motion target for the angle
        angle = np.deg2rad(angle)
        self.angle_motion_target_node = self.node.addChild("angle_motion_target")
        self.angle_motion_target_mechanical_object = self.angle_motion_target_node.addObject(
            "MechanicalObject",
            template="Vec1d",
            position=[float(angle), -float(angle)] if two_jaws else [float(angle)],
        )

        # Add controllable part without collision or visual model
        self.motion_target_node = self.node.addChild("motion_target")
        self.motion_target_mechanical_object = self.motion_target_node.addObject(
            "MechanicalObject",
            template="Rigid3d",
            position=pose.tolist(),
            showObject=show_object,
            showObjectScale=show_object_scale,
        )

        self.physical_shaft_node = self.node.addChild("physical_shaft")

        self.physical_shaft_mechanical_object = self.physical_shaft_node.addObject(
            "MechanicalObject",
            template="Rigid3d",
            position=self.initial_pose.tolist(),
            showObject=show_object,
            showObjectScale=show_object_scale,
        )

        if total_mass is not None:
            self.physical_shaft_node.addObject("UniformMass", totalMass=total_mass)

        if mechanical_binding == MechanicalBinding.ATTACH:
            # Add a constraint that binds the physical body to the motion target
            self.node.addObject(
                "AttachProjectiveConstraint",
                object1=self.motion_target_mechanical_object.getLinkPath(),
                object2=self.physical_shaft_mechanical_object.getLinkPath(),
                indices1=[0],  # The first Rigid3d pose of the motion target is mapped to
                indices2=[0],  # The first Rigid3d pose of the physical shaft
                twoWay=False,
            )
        else:
            assert spring_stiffness is not None and angular_spring_stiffness is not None, "When using springs to attatch motion target to mechanical body, please pass values for spring_stiffness and angular_spring_stiffness."
            self.physical_shaft_node.addObject(
                "RestShapeSpringsForceField",
                stiffness=spring_stiffness,  # Increase this if the body trags behind the target while moving
                angularStiffness=angular_spring_stiffness,  # Increase this if there is a rotational offset between body and target
                external_rest_shape=self.motion_target_mechanical_object.getLinkPath(),
                points=[0],
                external_points=[0],
                drawSpring=show_object,
            )

        # If using the FreeMotionAnimationLoop add a constraint correction object
        if animation_loop_type == AnimationLoopType.FREEMOTION:
            self.physical_shaft_node.addObject(
                ConstraintCorrectionType.UNCOUPLED.value,
                compliance=1.0 / total_mass if total_mass is not None else 1.0,
            )

        if visual_mesh_path_shaft is not None:
            self.shaft_visual_model_node = add_visual_model_func(
                attached_to=self.physical_shaft_node,
                name="visual_shaft",
                surface_mesh_file_path=visual_mesh_path_shaft,
                mapping_type=MappingType.RIGID,
                mapping_kwargs={"input": self.physical_shaft_mechanical_object.getLinkPath(), "index": 0},
                scale=scale,
            )

        if not is_default_collision_model_function(add_shaft_collision_model_func):
            function_kwargs = match_collision_model_kwargs(add_shaft_collision_model_func, locals=locals())
            self.shaft_collision_model_node = add_shaft_collision_model_func(attached_to=self.physical_shaft_node, **function_kwargs)
        else:
            if collision_mesh_path_shaft is not None:
                # Add collision models to the shaft
                self.shaft_collision_model_node = add_shaft_collision_model_func(
                    name="collision_shaft",
                    attached_to=self.physical_shaft_node,
                    surface_mesh_file_path=collision_mesh_path_shaft,
                    scale=scale,
                    mapping_type=MappingType.RIGID,
                    collision_group=collision_group,
                )

        self.articulation_description_node = self.node.addChild("articulation_description")

        # Add the MechanicalObject that holds the jaw angles
        self.angle_mechanical_object = self.articulation_description_node.addObject(
            "MechanicalObject",
            template="Vec1d",
            position=[float(angle), -float(angle)] if two_jaws else [float(angle)],
        )
        if two_jaws:
            self.articulation_description_node.addObject(
                "StopperLagrangianConstraint",
                name="angle_limit_jaw_0",
                min=angle_limits["low"],
                max=angle_limits["high"],
                index=0,
            )

            self.articulation_description_node.addObject(
                "StopperLagrangianConstraint",
                name="angle_limit_jaw_1",
                min=-angle_limits["high"],
                max=-angle_limits["low"],
                index=1,
            )
        else:
            self.articulation_description_node.addObject(
                "StopperLagrangianConstraint",
                min=-angle_limits["high"],
                max=-angle_limits["low"],
            )

        # If using the FreeMotionAnimationLoop add a constraint correction object
        if animation_loop_type == AnimationLoopType.FREEMOTION:
            self.articulation_description_node.addObject(
                ConstraintCorrectionType.UNCOUPLED.value,
                compliance=1.0 / total_mass if total_mass is not None else 1.0,
            )

        if spring_stiffness is None and angular_spring_stiffness is None:
            raise ValueError("When using springs to attatch motion target to mechanical body, please pass values for spring_stiffness and angular_spring_stiffness.")

        self.articulation_description_node.addObject(
            "RestShapeSpringsForceField",
            stiffness=articulation_spring_stiffness,
            angularStiffness=articulation_spring_stiffness,
            external_rest_shape=self.angle_motion_target_mechanical_object.getLinkPath(),
            points=[0, 1] if two_jaws else 0,
            external_points=[0, 1] if two_jaws else 0,
            drawSpring=show_object,
        )

        self.joint = self.articulation_description_node.addChild("joint")

        number_of_rigid_objects = 3 if two_jaws else 2

        self.joint_mechanical_object = self.joint.addObject(
            "MechanicalObject",
            template="Rigid3d",
            showObject=show_object,
            showObjectScale=show_object_scale,
            position=np.tile(pose, number_of_rigid_objects),
        )

        self.joint.addObject(
            "ArticulatedSystemMapping",
            input1=self.angle_mechanical_object.getLinkPath(),
            input2=self.physical_shaft_mechanical_object.getLinkPath(),
            output=self.joint_mechanical_object.getLinkPath(),
        )

        self.articulation_center_nodes = []
        self.articulation_nodes = []
        for index in range(2 if two_jaws else 1):
            # New node in the hierarchy
            self.articulation_center_nodes.append(self.joint.addChild(f"articulation_center_{index}"))

            # Add the articulation center to the hierarchy
            self.articulation_center_nodes[index].addObject(
                "ArticulationCenter",
                parentIndex=0,
                childIndex=index + 1,
                posOnParent=[0, 0, 0],
                posOnChild=[0, 0, 0],
            )

            # Add a child of the center that holds the actual articulation
            self.articulation_nodes.append(self.articulation_center_nodes[index].addChild("articulation"))

            # Add the articulation object to the articulation node
            if sum(rotation_axis) != 1:
                raise ValueError(f"Only one axis can be set as rotation axis of the articulation. E.g. ([1, 0, 0] for rotation around the X-axis). Received {rotation_axis=}")
            self.articulation_nodes[index].addObject(
                "Articulation",
                translation=False,
                rotation=True,
                rotationAxis=rotation_axis,
                articulationIndex=index,
            )

        self.articulation_description_node.addObject("ArticulatedHierarchyContainer")

        self.physical_jaw_node = self.node.addChild("physical_jaw")

        if visual_mesh_paths_jaws is not None:
            if not len(visual_mesh_paths_jaws) in (1, 2):
                raise ValueError(f"Received the wrong number of paths to visual meshes for the instrument jaws. For {two_jaws=} please pass {1 if two_jaws else 2} paths. Received {len(visual_mesh_paths_jaws)}.")
        else:
            visual_mesh_paths_jaws = []

        self.jaw_collision_model_nodes = []
        self.jaw_visual_model_nodes = []
        for index, visual_mesh_path in enumerate(visual_mesh_paths_jaws):
            self.jaw_visual_model_nodes.append(
                add_visual_model_func(
                    attached_to=self.physical_jaw_node,
                    name=f"visual_jaw_{index}",
                    surface_mesh_file_path=visual_mesh_path,
                    mapping_type=MappingType.RIGID,
                    mapping_kwargs={"input": self.joint_mechanical_object.getLinkPath(), "index": index + 1},
                    scale=scale,
                )
            )

        if collision_mesh_paths_jaws is not None:
            if not len(collision_mesh_paths_jaws) in (1, 2):
                raise ValueError(f"Received the wrong number of paths to collision meshes for the instrument jaws. For {two_jaws=} please pass {1 if two_jaws else 2} paths. Received {len(collision_mesh_paths_jaws)}.")
        else:
            collision_mesh_paths_jaws = []

        if not isinstance(add_jaws_collision_model_func, list):
            # TODO: Find a way to make this valid, when only one function is passed that adds models to both jaws.
            add_jaws_collision_model_func = [add_jaws_collision_model_func] * (2 if two_jaws else 1)

        for index, function in enumerate(add_jaws_collision_model_func):
            if not is_default_collision_model_function(function):
                function_kwargs = match_collision_model_kwargs(function, locals=locals())
                self.jaw_collision_model_nodes.append(function(attached_to=self.physical_jaw_node, **function_kwargs))
            else:
                if collision_mesh_paths_jaws:
                    try:
                        self.jaw_collision_model_nodes.append(
                            function(
                                attached_to=self.physical_jaw_node,
                                name=f"collision_jaw_{index}",
                                surface_mesh_file_path=collision_mesh_paths_jaws[index],
                                scale=scale,
                                mapping_type=MappingType.RIGID,
                                mapping_kwargs={"input": self.joint_mechanical_object.getLinkPath(), "index": index + 1},
                                collision_group=collision_group,
                            )
                        )
                    except IndexError:
                        raise IndexError(f"Tried to add jaw collision model number {index+1}, but could not find a corresponding collision mesh in {collision_mesh_paths_jaws=}.")

    def get_pose(self) -> np.ndarray:
        """Reads the Rigid3d pose from the controllable sofa node and returns it as [x, y, z, a, b, c, w].

        Notes:
            - returned array is read-only

        """
        return self.motion_target_mechanical_object.position.array()[0]

    def get_physical_pose(self) -> np.ndarray:
        """Reads the Rigid3d pose from the non-controllable sofa node and returns it as [x, y, z, a, b, c, w].

        Notes:
            - returned array is read-only

        """
        return self.physical_shaft_mechanical_object.position.array()[0]

    def set_pose(self, pose: np.ndarray) -> None:
        """Writes the Rigid3d pose from the controllable sofa node as [x, y, z, a, b, c, w].

        Notes:
            - pose values are written into the sofa array without assiging the pose array to the sofa array.
              Changes in the pose array after that will not be propagated to sofa.
        """
        with self.motion_target_mechanical_object.position.writeable() as sofa_pose:
            sofa_pose[:] = pose

    def set_rest_pose(self, pose: np.ndarray) -> None:
        """Writes the Rigid3d rest pose from the controllable sofa node as [x, y, z, a, b, c, w].

        Notes:
            - pose values are written into the sofa array without assiging the pose array to the sofa array.
              Changes in the pose array after that will not be propagated to sofa.
        """
        with self.motion_target_mechanical_object.rest_position.writeable() as sofa_pose:
            # Write values into array (instead of assigning one array to the other)
            sofa_pose[:] = pose

    def get_pose_difference(self, position_norm: bool = False) -> np.ndarray:
        """Reads the Rigid3d poses from both motion target and body and returns the difference between the two.

        Args:
            position_norm: If True, the Cartesian norm of the position difference is returned.

        Returns:
            [d_trans_x, d_trans_y, d_trans_z, d_rot] if position_norm == False
            [d_trans, d_rot] if position_norm == True
        """

        target_pose = self.motion_target_mechanical_object.position.array()[0]
        actual_pose = self.physical_shaft_mechanical_object.position.array()[0]
        position_delta = target_pose[:3] - actual_pose[:3]

        orientation_delta = np.rad2deg(2 * np.arccos(np.clip(np.dot(target_pose[-4:], actual_pose[-4:]), a_min=-1.0, a_max=1.0)))  # minimal rotation angle

        if position_norm:
            position_delta = np.linalg.norm(position_delta)

        return np.append(position_delta, orientation_delta)

    def get_angle_difference(self) -> Union[float, np.ndarray]:
        """Reads the Vec1d valuef from both motion target and body and returns the difference between the two."""
        target_angles = np.rad2deg(self.angle_motion_target_mechanical_object.position.array()[0])
        actual_angles = np.rad2deg(self.angle_mechanical_object.position.array()[0])
        return target_angles - actual_angles

    def get_angle(self) -> float:
        """Reads the Vec1d value of the controllable part of the angle and returns it as degrees."""
        return np.rad2deg(self.angle_motion_target_mechanical_object.position.array()[0])

    def get_actual_angle(self) -> float:
        """Reads the Vec1d value of the physical part of the angle and returns it as degrees."""
        return np.rad2deg(self.angle_mechanical_object.position.array()[0])

    def set_angle(self, angle: float) -> None:
        """Writes the Vec1d value of the controllable part of the angle as degrees.

        Notes:
            - angle is limited by ``angle_limits``.
        """
        if self.angle_limits["low"] <= angle <= self.angle_limits["high"]:
            with self.angle_motion_target_mechanical_object.position.writeable() as sofa_angle:
                sofa_angle[0] = np.deg2rad(angle)
                if self.two_jaws:
                    sofa_angle[1] = -np.deg2rad(angle)
            self.last_set_angle_violated_jaw_limits = False
        else:
            self.last_set_angle_violated_jaw_limits = True


class PivotizedArticulatedInstrument(ArticulatedInstrument):
    """Extends the ArticulatedInstrument with pivotized motion.

    Args:
        parent_node (Sofa.Core.Node): Parent node of the object.
        name (str): Name of the object.
        visual_mesh_path_shaft (Union[str, Path]): Path to the visual surface mesh of the instrument's shaft.
        visual_mesh_paths_jaws (List[Union[str, Path]]): A list of paths to the visual surface meshes of the instrument's jaws. Can be one or two, depending on ``two_jaws``.
        angle (float): Initial opening angle of the jaws in degrees.
        angle_limits (Union[Tuple[float, float], Dict[str, float]]): Minimum and maximum angle value in degrees.
        total_mass (float): Total mass of the deformable object.
        two_jaws (bool): Whether the instrument has two or one jaws.
        collision_mesh_path_shaft (Optional[Union[str, Path]]): Path to the collision surface mesh of the instrument's shaft.
        collision_mesh_paths_jaws (Optional[List[Union[str, Path]]]): A list of paths to the collision surface meshes of the instrument's jaws. Can be one or two, depending on ``two_jaws``.
        rotation_axis (Tuple[int, int, int]): Which axis of the jaw models to use for rotation. Only one of them should be set to 1. The rest should be set to 0.
        scale: (Union[float, Tuple[float, float, float]]): Scale factor for loading the meshes.
        add_solver_func (Callable): Function that adds the numeric solvers to the object.
        add_shaft_collision_model_func (Callable): Function that defines how the collision surface from ``collision_mesh_path_shaft`` is added to the object.
        add_jaws_collision_model_func (Callable): Function that defines how the collision surfaces from ``collision_mesh_paths_jaws`` are added to the object.
        add_visual_model_func (Callable): Function that defines how the visual surfaces from ``visual_mesh_path_shaft`` and ``visual_mesh_paths_jaws`` are added to the object.
        animation_loop_type (AnimationLoopType): The animation loop of the scene. Required to determine if objects for constraint correction should be added.
        show_object (bool): Whether to render the nodes.
        show_object_scale (float): Render size of the node if ``show_object`` is ``True``.
        mechanical_binding (MechanicalBinding): Whether to use ``"RestShapeSpringsForceField"`` or ``"AttachProjectiveConstraint"`` to combine controllable and non-controllable part of the object.
        spring_stiffness (Optional[float]): Spring stiffness of the ``"RestShapeSpringsForceField"``.
        angular_spring_stiffness (Optional[float]): Angular spring stiffness of the ``"RestShapeSpringsForceField"``.
        articulation_spring_stiffness (Optional[float]): Spring stiffness of the ``"RestShapeSpringsForceField"`` of the articulation.
        collision_group (int): The group for which collisions with this object should be ignored. Value has to be set since the jaws and shaft must belong to the same group.
        ptsd_state (np.ndarray): Pan, tilt, spin, depth state of the pivotized tool.
        rcm_pose (np.ndarray): [x, y, z, X, Y, Z] Cartesian position and XYZ euler angles of the remote center of motion.
        cartesian_workspace (Dict): Low and high values of the instrument's Cartesian workspace.
        ptsd_reset_noise (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Noise added to the ptsd state when resetting the object.
        rcm_reset_noise (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Noise added to the rcm pose when resetting the object.
        angle_reset_noise (Optional[Union[float, Dict[str, float]]]): Noise added to the angle when resetting the object.
        state_limits (Dict): Low and high values of the instrument's state space.
        show_remote_center_of_motion (bool): Whether to render the remote center of motion.
        show_workspace (bool): Whether to render the workspace.
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        angle: float = 0.0,
        angle_limits: Union[Tuple[float, float], Dict[str, float]] = {
            "low": float(np.finfo(np.float16).min),
            "high": float(np.finfo(np.float16).max),
        },
        total_mass: Optional[float] = None,
        two_jaws: bool = True,
        collision_mesh_path_shaft: Optional[Union[str, Path]] = None,
        collision_mesh_paths_jaws: Optional[List[Union[str, Path]]] = None,
        visual_mesh_path_shaft: Optional[Union[str, Path]] = None,
        visual_mesh_paths_jaws: Optional[List[Union[str, Path]]] = None,
        rotation_axis: Tuple[int, int, int] = (1, 0, 0),
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        add_solver_func: Callable = add_solver,
        add_shaft_collision_model_func: Callable = add_collision_model,
        add_jaws_collision_model_func: Union[Callable, List[Callable]] = add_collision_model,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.FREEMOTION,
        show_object: bool = False,
        show_object_scale: float = 1.0,
        mechanical_binding: MechanicalBinding = MechanicalBinding.ATTACH,
        spring_stiffness: Optional[float] = 2e10,
        angular_spring_stiffness: Optional[float] = 2e10,
        articulation_spring_stiffness: Optional[float] = 1e12,
        collision_group: int = 0,
        ptsd_state: np.ndarray = np.zeros(4),
        rcm_pose: np.ndarray = np.zeros(6),
        cartesian_workspace: Dict = {
            "low": np.array([-np.inf, -np.inf, -np.inf]),
            "high": np.array([np.inf, np.inf, np.inf]),
        },
        ptsd_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        rcm_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        angle_reset_noise: Optional[Union[float, Dict[str, float]]] = None,
        state_limits: Dict = {
            "low": np.array([-90, -90, -90, 0]),
            "high": np.array([90, 90, 90, 200]),
        },
        show_remote_center_of_motion: bool = False,
        show_workspace: bool = False,
    ) -> None:

        if not isinstance(rcm_pose, np.ndarray) and not rcm_pose.shape == (6,):
            raise ValueError(f"Please pass the pose of the remote center of motion (rcm_pose) as a numpy array with [x, y, z, X, Y, Z] (position, rotation). Received {rcm_pose}.")
        self.remote_center_of_motion = rcm_pose.copy()
        self.pivot_transform = generate_ptsd_to_pose(rcm_pose=self.remote_center_of_motion)

        if not isinstance(ptsd_state, np.ndarray) and not ptsd_state.shape == (4,):
            raise ValueError(f"Please pass the instruments state as a numpy array with [pan, tilt, spin, depth]. Received {ptsd_state}.")

        self.ptsd_state = ptsd_state
        self.articulated_state = np.zeros(5)
        self.articulated_state[:4] = ptsd_state
        self.articulated_state[-1] = angle

        self.initial_state = np.copy(self.ptsd_state)
        self.initial_pose = self.pivot_transform(self.initial_state)
        self.initial_angle = angle
        self.initial_remote_center_of_motion = rcm_pose.copy()

        self.cartesian_workspace = cartesian_workspace
        self.state_limits = state_limits

        self.last_set_state_violated_state_limits = False
        self.last_set_state_violated_workspace_limits = False

        self.ptsd_reset_noise = ptsd_reset_noise
        self.rcm_reset_noise = rcm_reset_noise
        self.angle_reset_noise = angle_reset_noise

        super().__init__(
            parent_node=parent_node,
            name=name,
            pose=self.initial_pose,
            visual_mesh_path_shaft=visual_mesh_path_shaft,
            visual_mesh_paths_jaws=visual_mesh_paths_jaws,
            angle=angle,
            angle_limits=angle_limits,
            total_mass=total_mass,
            two_jaws=two_jaws,
            collision_mesh_path_shaft=collision_mesh_path_shaft,
            collision_mesh_paths_jaws=collision_mesh_paths_jaws,
            rotation_axis=rotation_axis,
            scale=scale,
            add_solver_func=add_solver_func,
            add_shaft_collision_model_func=add_shaft_collision_model_func,
            add_jaws_collision_model_func=add_jaws_collision_model_func,
            add_visual_model_func=add_visual_model_func,
            animation_loop_type=animation_loop_type,
            show_object=show_object,
            show_object_scale=show_object_scale,
            mechanical_binding=mechanical_binding,
            spring_stiffness=spring_stiffness,
            angular_spring_stiffness=angular_spring_stiffness,
            articulation_spring_stiffness=articulation_spring_stiffness,
            collision_group=collision_group,
        )

        self.show_remote_center_of_motion = show_remote_center_of_motion
        if show_remote_center_of_motion:
            self.rcm_node = self.parent_node.addChild(f"rcm_{name}")
            self.rcm_mechanical_object = self.rcm_node.addObject(
                "MechanicalObject",
                template="Rigid3d",
                position=self.pivot_transform((0, 0, 0, 0)),
                showObject=True,
                showObjectScale=show_object_scale,
            )

        self.show_workspace = show_workspace
        if show_workspace:
            self.workspace_node = self.parent_node.addChild(f"workspace_{name}")
            self.workspace_mechanical_object = self.workspace_node.addObject("MechanicalObject")
            add_bounding_box(self.workspace_node, min=cartesian_workspace["low"], max=cartesian_workspace["high"], show_bounding_box=True, show_bounding_box_scale=show_object_scale)

    def get_state(self) -> np.ndarray:
        """Gets the current state of the instrument."""
        read_only_state = self.ptsd_state.view()
        read_only_state.flags.writeable = False
        return read_only_state

    def set_articulated_state(self, articulated_state: np.ndarray) -> None:
        """Sets the state of the instrument including the articulation angle withing the defined state limits."""
        self.set_state(articulated_state[:4])
        self.set_angle(articulated_state[-1])

    def get_articulated_state(self) -> np.ndarray:
        """Gets the state of the instrument including the articulation angle withing the defined state limits."""
        self.articulated_state[:4] = self.ptsd_state
        self.articulated_state[-1] = self.get_angle()
        read_only_state = self.articulated_state.view()
        read_only_state.flags.writeable = False
        return read_only_state

    def set_state(self, state: np.ndarray) -> None:
        """Sets the state of the instrument withing the defined state limits and Cartesian workspace.

        Warning:
            The components of a Cartesian pose are not independently changeable, since this object has a remote center of motion.
            We thus cannton simple ignore one part (e.g. the x component) and still write the other components (e.g. y).
            Poses that are not validly withing the workspace will be ignored.
            The state, however, is independently constrainable so only invalid components (e.g. tilt) will be ignored.
        """

        # Check if there are any states that are outside the state limits
        invalid_states_mask = (self.state_limits["low"] > state) | (state > self.state_limits["high"])

        # Overwrite the invalide parts of the states with the current state
        state[invalid_states_mask] = self.ptsd_state[invalid_states_mask]

        # Get the corresponding pose from the state
        pose = self.pivot_transform(state)

        # Save info about violation of state limits
        self.last_set_state_violated_state_limits = np.any(invalid_states_mask)

        # Only set the pose, if all components are within the Cartesian workspace
        if not np.any((self.cartesian_workspace["low"] > pose[:3]) | (pose[:3] > self.cartesian_workspace["high"])):
            self.set_pose(pose)

            # Only overwrite the internal value of ptsd_state, if that was successful
            self.ptsd_state[:] = state

            # Save info about violation of Cartesian workspace limits
            self.last_set_state_violated_workspace_limits = False
        else:
            self.last_set_state_violated_workspace_limits = True

    def reset_state(self) -> None:
        """Resets the object to its initial ptsd state. Optionally adds noise to rcm pose and ptsd state."""
        #############
        # Reset state
        #############
        # Generate a new pivot_transform by adding noise to the initial remote_center_of_motion pose
        if self.rcm_reset_noise is not None:
            if isinstance(self.rcm_reset_noise, np.ndarray):
                # Uniformly sample from -noise to +noise and add it to the rcm pose
                new_rcm_pose = self.initial_remote_center_of_motion + self.rng.uniform(-self.rcm_reset_noise, self.rcm_reset_noise)
            elif isinstance(self.rcm_reset_noise, dict):
                # Uniformly sample from low to high and add it to the rcm pose
                new_rcm_pose = self.initial_remote_center_of_motion + self.rng.uniform(self.rcm_reset_noise["low"], self.rcm_reset_noise["high"])
            else:
                raise TypeError("Please pass the rcm_reset_noise as a numpy array or a dictionary with 'low' and 'high' keys.")
            self.pivot_transform = generate_ptsd_to_pose(new_rcm_pose)
            self.remote_center_of_motion[:] = new_rcm_pose
            if self.show_remote_center_of_motion:
                with self.rcm_mechanical_object.position.writeable() as rcm_pose:
                    rcm_pose[:] = self.pivot_transform((0, 0, 0, 0))

        # Select a new ptsd state by adding noise to the initial state
        if self.ptsd_reset_noise is not None:
            if isinstance(self.ptsd_reset_noise, np.ndarray):
                # Uniformly sample from -noise to +noise and add it to the initial state
                new_state = self.initial_state + self.rng.uniform(-self.ptsd_reset_noise, self.ptsd_reset_noise)
            elif isinstance(self.ptsd_reset_noise, dict):
                # Uniformly sample from low to high and add it to the initial state
                new_state = self.initial_state + self.rng.uniform(self.ptsd_reset_noise["low"], self.ptsd_reset_noise["high"])
            else:
                raise TypeError("Please pass the ptsd_reset_noise as a numpy array or a dictionary with 'low' and 'high' keys.")

            # Do that until a pose is found that fits in the Cartesian workspace and the state limits
            while np.any((self.state_limits["low"] > new_state) | (new_state > self.state_limits["high"])) or np.any((self.cartesian_workspace["low"] > self.pivot_transform(new_state)[:3]) | (self.pivot_transform(new_state)[:3] > self.cartesian_workspace["high"])):
                if isinstance(self.ptsd_reset_noise, np.ndarray):
                    # Uniformly sample from -noise to +noise and add it to the initial state
                    new_state = self.initial_state + self.rng.uniform(-self.ptsd_reset_noise, self.ptsd_reset_noise)
                else:
                    # Uniformly sample from low to high and add it to the initial state
                    new_state = self.initial_state + self.rng.uniform(self.ptsd_reset_noise["low"], self.ptsd_reset_noise["high"])
        else:
            new_state = self.initial_state

        #############
        # Reset angle
        #############
        # Select a new angle by adding noise to the initial angle
        if self.angle_reset_noise is not None:
            if isinstance(self.angle_reset_noise, float):
                # Uniformly sample from -noise to +noise and add it to the initial angle
                new_angle = self.initial_angle + self.rng.uniform(-self.angle_reset_noise, self.angle_reset_noise)
            elif isinstance(self.angle_reset_noise, dict):
                # Uniformly sample from low to high and add it to the initial angle
                new_angle = self.initial_angle + self.rng.uniform(self.angle_reset_noise["low"], self.angle_reset_noise["high"])
            else:
                raise TypeError("Please pass the angle_reset_noise as a float or a dictionary with 'low' and 'high' keys.")

            new_angle = np.clip(new_angle, self.angle_limits["low"], self.angle_limits["high"])

        else:
            new_angle = self.initial_angle

        # Update SOFA and internal values
        self.set_state(new_state)
        self.set_angle(new_angle)
        self.ptsd_state[:] = new_state
        self.articulated_state[:4] = new_state
        self.articulated_state[-1] = new_angle

    def seed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Creates a random number generator from a seed."""
        self.rng = np.random.default_rng(seed)
