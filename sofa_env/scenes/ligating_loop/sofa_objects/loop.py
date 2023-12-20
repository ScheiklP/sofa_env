import numpy as np
from typing import Optional, Union, Callable, Dict
from enum import Enum, unique

import Sofa.Core
import Sofa.SofaDeformable
from sofa_env.sofa_templates.motion_restriction import add_bounding_box

from sofa_env.utils.math_helper import euler_to_rotation_matrix, homogeneous_transform_to_pose, rotated_x_axis

from sofa_env.sofa_templates.topology import cylinder_shell_triangle_topology_data, TOPOLOGY_PLUGIN_LIST
from sofa_env.sofa_templates.rigid import MechanicalBinding, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.rope import Rope, ROPE_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.solver import ConstraintCorrectionType, add_solver, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.rope import Rope, RopeCollisionType, poses_for_circular_rope, ROPE_PLUGIN_LIST


LOOP_PLUGIN_LIST = (
    [
        "Sofa.Component.SolidMechanics.Spring",  # <- [RestShapeSpringsForceField]
        "Sofa.Component.Mapping.NonLinear",  # <- [RigidMapping]
    ]
    + RIGID_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
    + ROPE_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
    + TOPOLOGY_PLUGIN_LIST
    + ROPE_PLUGIN_LIST
)


@unique
class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1
    POSITION = 2


class LigatingLoop(Sofa.Core.Controller):
    """Object to describe a ligation loop

    Args:
        parent_node (Sofa.Core.Node): Parent node of the SOFA object.
        name (str): Name of the object.
        num_rope_points (int): Number of points to describe the loop's rope.
        loop_radius (float): Radius of the loop.
        mechanical_rope_parameters (dict): Mechanical parameters of the loop's rope.
            "beam_radius": Mechanical radius of the beam.
            "radius": Radius of the rope for visual and collision.
            "total_mass": Total mass of the rope.
            "poisson_ratio": Poisson ratio of the rope.
            "young_modulus": The beam's Young modulus.
            "mechanical_damping": Proportional velocity damping.
        ptsd_state (np.ndarray): PTSD state of the loop.
        rcm_pose (np.ndarray): XYZ position and XYZ Euler angles of the remote center of motion.
        total_mass (Optional[float]): Total mass of the instrument.
        animation_loop_type (AnimationLoopType): The scenes animation loop in order to correctly add constraint correction objects.
        mechanical_binding (MechanicalBinding): Whether to use ``"RestShapeSpringsForceField"`` or ``"AttachConstraint"`` to combine controllable and non-controllable part of the object.
        show_object (bool): Whether to render the nodes.
        show_object_scale (float): Render size of the node if ``show_object`` is ``True``.
        spring_stiffness (Optional[float]): Spring stiffness of the ``"RestShapeSpringsForceField"``.
        angular_spring_stiffness (Optional[float]): Angular spring stiffness of the ``"RestShapeSpringsForceField"``.
        add_solver_func (Callable): Function that adds the numeric solvers to the object.
        collision_group (int): Add the model to a collision group to disable collision checking between those models.
        cartesian_workspace (Dict): Low and high values of the instrument's Cartesian workspace.
        state_limits (Dict[str, np.ndarray]): Low and high values of the instrument's PTSD state space.
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        num_rope_points: int = 50,
        loop_radius: float = 15.0,
        mechanical_rope_parameters: dict = {
            "beam_radius": 3.0,
            "radius": 0.3,
            "total_mass": 5.0,
            "poisson_ratio": 0.0,
            "young_modulus": 1e7,
            "mechanical_damping": 1.0,
        },
        ptsd_state: np.ndarray = np.zeros(4),
        rcm_pose: np.ndarray = np.zeros(6),
        total_mass: Optional[float] = None,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        mechanical_binding: MechanicalBinding = MechanicalBinding.SPRING,
        show_object: bool = False,
        show_object_scale: float = 1.0,
        spring_stiffness: float = 1e22,
        angular_spring_stiffness: float = 1e22,
        add_solver_func: Callable = add_solver,
        collision_group: Optional[int] = None,
        cartesian_workspace: Dict = {
            "low": np.array([-np.inf, -np.inf, -np.inf]),
            "high": np.array([np.inf, np.inf, np.inf]),
        },
        state_limits: Dict[str, np.ndarray] = {
            "low": np.array([-90.0, -90.0, np.finfo(np.float16).min, 0.0]),
            "high": np.array([90.0, 90.0, np.finfo(np.float16).max, 100.0]),
        },
        action_type: ActionType = ActionType.CONTINUOUS,
    ) -> None:
        Sofa.Core.Controller.__init__(self)
        self.name: Sofa.Core.DataString = f"{name}_controller"
        self.node = parent_node.addChild(f"{self.name.value}_node")
        self.time_integration, self.linear_solver = add_solver_func(attached_to=self.node)
        self.action_type = action_type

        ##############
        # Pivotization
        ##############
        if not isinstance(rcm_pose, np.ndarray) and not rcm_pose.shape == (6,):
            raise ValueError(f"Please pass the pose of the remote center of motion (rcm_pose) as a numpy array with [x, y, z, X, Y, Z] (position, rotation). Received {rcm_pose}.")
        self.remote_center_of_motion = rcm_pose.copy()
        self.pivot_transform = generate_ptsd_to_pose(rcm_pose=self.remote_center_of_motion)

        if not isinstance(ptsd_state, np.ndarray) and not ptsd_state.shape == (4,):
            raise ValueError(f"Please pass the instruments state as a numpy array with [pan, tilt, spin, depth]. Received {ptsd_state}.")
        self.ptsd_state = ptsd_state.copy()

        self.initial_state = self.ptsd_state.copy()
        self.initial_pose = self.pivot_transform(self.initial_state)
        self.initial_remote_center_of_motion = rcm_pose.copy()

        self.cartesian_workspace = cartesian_workspace
        self.state_limits = state_limits

        self.last_set_state_violated_state_limits = False
        self.last_set_state_violated_workspace_limits = False

        if show_object:
            rcm_node = self.node.addChild("remote_center_of_motion")
            self.rcm_mechanical_object = rcm_node.addObject(
                "MechanicalObject",
                template="Rigid3d",
                position=self.pivot_transform((0, 0, 0, 0)),
                showObject=show_object,
                showObjectScale=show_object_scale,
            )
            add_bounding_box(rcm_node, min=cartesian_workspace["low"], max=cartesian_workspace["high"], show_bounding_box=True)

        ######
        # Rope
        ######
        rope_start_position = self.initial_pose[:3] - np.array([loop_radius, 0.0, 0.0])
        rope_poses = poses_for_circular_rope(radius=loop_radius, start_position=rope_start_position, num_points=num_rope_points)
        self.rope = Rope(
            parent_node=parent_node,
            name="rope",
            poses=rope_poses,
            fix_start=False,
            collision_type=RopeCollisionType.SPHERES,
            animation_loop_type=animation_loop_type,
            collision_group=collision_group,
            show_object=show_object,
            show_object_scale=show_object_scale,
            **mechanical_rope_parameters,
        )

        ###############
        # Motion Target
        ###############
        # Motion target for all points of the rope, and the physical body of the instrument
        self.motion_target_mechanical_object = self.node.addObject(
            "MechanicalObject",
            template="Rigid3d",
            position=np.tile(self.initial_pose, self.rope.num_points + 1),  # instrument, rope point 0, rope point 1, ...
            showObject=show_object,
            showObjectScale=3.0,
        )

        ###############
        # Physical Body
        ###############
        # Node to attach the instrument shell
        self.physical_node = self.node.addChild(f"{self.name.value}_physical_node")
        self.physical_mechanical_object = self.physical_node.addObject(
            "MechanicalObject",
            template="Rigid3d",
            position=self.initial_pose,
            showObject=show_object,
            showObjectScale=show_object_scale,
        )

        if total_mass is not None:
            self.physical_node.addObject("UniformMass", totalMass=total_mass)

        # If using the FreeMotionAnimationLoop add a constraint correction object
        if animation_loop_type == AnimationLoopType.FREEMOTION:
            self.physical_node.addObject(
                ConstraintCorrectionType.UNCOUPLED.value,
                compliance=1.0 / total_mass if total_mass is not None else 1.0,
            )

        if mechanical_binding == MechanicalBinding.SPRING:
            self.physical_node.addObject(
                "RestShapeSpringsForceField",
                stiffness=spring_stiffness,
                angularStiffness=angular_spring_stiffness,
                external_rest_shape=self.motion_target_mechanical_object.getLinkPath(),
                points=[0],
                external_points=[0],
            )
        else:
            self.physical_node.addObject(
                "AttachConstraint",
                object1=self.motion_target_mechanical_object.getLinkPath(),
                object2=self.physical_mechanical_object.getLinkPath(),
                indices1=[0],  # The first Rigid3d pose of the motion target is mapped to
                indices2=[0],  # The first Rigid3d pose of the physical body
                twoWay=False,
            )

        points, triangles = cylinder_shell_triangle_topology_data(
            radius=self.rope.radius,
            height=-100,
            num_phi=6,
            num_z=3,
        )
        sphere_collision_positions = np.linspace([0, 0, -100], [0, 0, 0], int(100 / self.rope.radius / 2))
        shell_node = self.node.addChild("shell")
        # The shell's body is rotated by -90 degrees around Y, to offset the +90 degrees rotation around Y that aligns the X axis of the rope with the motion target (pivot transform)
        shell_mechanical_object = shell_node.addObject(
            "MechanicalObject",
            position=points,
            showObject=show_object,
            showObjectScale=show_object_scale,
            template="Vec3d",
            rotation=[0, -90, 0],
        )
        shell_node.addObject("TriangleSetTopologyContainer", triangles=triangles)
        shell_node.addObject("TriangleSetTopologyModifier")
        shell_node.addObject("TriangleSetGeometryAlgorithms")
        shell_node.addObject("RigidMapping", input=self.physical_mechanical_object.getLinkPath(), output=shell_mechanical_object.getLinkPath())

        collision_node = shell_node.addChild("collision")
        collision_node.addObject(
            "MechanicalObject",
            template="Vec3d",
            position=sphere_collision_positions,
            rotation=[0, -90, 0],
        )
        self.shell_collision_model = collision_node.addObject("SphereCollisionModel", group=collision_group, radius=self.rope.radius)
        collision_node.addObject("RigidMapping", input=self.physical_mechanical_object.getLinkPath(), output=collision_node.MechanicalObject.getLinkPath())

        # If using the FreeMotionAnimationLoop add a constraint correction object
        if animation_loop_type == AnimationLoopType.FREEMOTION:
            shell_node.addObject(
                ConstraintCorrectionType.UNCOUPLED.value,
                compliance=1.0 / total_mass if total_mass is not None else 1.0,
            )

        shell_visual_node = shell_node.addChild("ogl")
        shell_ogl_model = shell_visual_node.addObject("OglModel")
        shell_visual_node.addObject("IdentityMapping", input=shell_mechanical_object.getLinkPath(), output=shell_ogl_model.getLinkPath())

        ##############
        # Rope Control
        ##############

        self.stiffness = spring_stiffness
        self.angular_stiffness = angular_spring_stiffness

        # Index to control how closed the loop is
        self.active_index = self.rope.num_points - 1

        stiffness = np.zeros(self.rope.num_points)
        angular_stiffness = np.zeros(self.rope.num_points)
        # Fix the first point
        stiffness[0] = spring_stiffness
        angular_stiffness[0] = angular_spring_stiffness
        # Fix all the points that come after the active index
        stiffness[self.active_index :] = spring_stiffness
        angular_stiffness[self.active_index :] = angular_spring_stiffness

        self.springs = self.rope.node.addObject(
            "RestShapeSpringsForceField",
            external_rest_shape=self.motion_target_mechanical_object.getLinkPath(),
            drawSpring=True,
            stiffness=stiffness,
            angularStiffness=angular_stiffness,
            points=list(range(self.rope.num_points)),
            external_points=[list(range(1, self.rope.num_points + 1))],
        )

        self.segment_length = np.linalg.norm(self.rope.start_poses[0] - self.rope.start_poses[1])

    def get_state(self) -> np.ndarray:
        """Gets the current state of the instrument."""
        read_only_state = self.ptsd_state.view()
        read_only_state.flags.writeable = False
        return read_only_state

    def get_articulated_state(self) -> np.ndarray:
        """Gets the current state of the instrument including ratio of how closed the loop is."""
        return np.append(self.get_state(), self.get_ratio_loop_closed())

    def get_loop_state(self) -> np.ndarray:
        return self.rope.get_state()

    def get_loop_positions(self) -> np.ndarray:
        return self.rope.get_positions()

    def get_loop_velocities(self) -> np.ndarray:
        return self.rope.get_velocities()

    def get_pose(self) -> np.ndarray:
        return self.motion_target_mechanical_object.position.array()[0]

    def get_ratio_loop_closed(self) -> float:
        # 5 -> 1.0
        # self.rope.num_points - 1 -> 0.0
        return (self.active_index - self.rope.num_points + 1) / (6 - self.rope.num_points)

    def seed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Creates a random number generator from a seed."""
        self.rng = np.random.default_rng(seed)

    def onKeypressedEvent(self, event):
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
            new_index = self.active_index - 1
            self.set_active_index(new_index)

        elif key == "P":
            new_index = self.active_index + 1
            self.set_active_index(new_index)

        elif ord(key) == 32:  # space
            print(repr(self.ptsd_state))

        else:
            pass

    def set_active_index(self, index: int) -> None:
        # Limit the index to [5, last point on rope]
        new_index = max(index, 5)
        new_index = min(new_index, self.rope.num_points - 1)
        self.active_index = new_index
        # Call set_state to update springs and positions
        self.set_state(self.ptsd_state)

    def get_active_index(self) -> int:
        return self.active_index

    def get_center_of_mass(self) -> np.ndarray:
        return np.mean(self.rope.get_positions()[0 : self.active_index], axis=0)

    def do_action(self, action: np.ndarray) -> None:
        """Performs an action on the instrument."""
        if self.action_type == ActionType.POSITION:
            loop_closedness = action[-1]
            # denormalize loop_closedness from [0, 1] to range [min_active_index, max_active index]
            # but inverse, i.e. loop_closedness=0 corresponds to max_active_index
            min_active_index = 5
            max_active_index = self.rope.num_points - 1
            new_active_index_float = min_active_index + (1 - loop_closedness) * (max_active_index - min_active_index)
            new_active_index_int = round(new_active_index_float)
            new_active_index_int_clipped = max(min_active_index, min(new_active_index_int, max_active_index))
            self.active_index = new_active_index_int_clipped
            # set ptsd state to desired position
            self.set_state(action[:-1])

        else:
            # The first four elements of the action will be added to the instrument's ptsd state.
            # The last element closes the loop, if the value is <= -0.5, and openes the loop, if the value is >= 0.5.
            loop_closing_action = action[-1]
            if loop_closing_action <= -0.5:
                new_index = self.active_index + 1
            elif loop_closing_action >= 0.5:
                new_index = self.active_index - 1
            else:
                new_index = self.active_index

            new_index = max(new_index, 5)
            new_index = min(new_index, self.rope.num_points - 1)
            self.active_index = new_index

            self.set_state(self.get_state() + action[:-1])

    def set_state(self, state: np.ndarray) -> None:
        """Sets the state of the instrument withing the defined state limits and Cartesian workspace.

        Warning:
            The components of a Cartesian pose are not independently changeable, since this object has a remote center of motion.
            We thus cannton simple ignore one part (e.g. the x component) and still write the other components (e.g. y).
            Poses that are not validly withing the workspace will be ignored.
            The state, however, is independently constrainable so only invalid components (e.g. tilt) will be ignored.

        """

        # Update all the springs with the given active index
        with self.springs.stiffness.writeable() as stiffness:
            # The first point of the rope is always fixed -> reset all after the first to 0.0 stiffness
            stiffness[1:] = 0.0
            # Starting from the active index -> set stiffness
            stiffness[self.active_index :] = self.stiffness

        # Same for angular stiffness
        with self.springs.angularStiffness.writeable() as angular_stiffness:
            angular_stiffness[1:] = 0.0
            angular_stiffness[self.active_index :] = self.angular_stiffness

        # Check if there are any states that are outside the state limits
        invalid_states_mask = (self.state_limits["low"] > state) | (state > self.state_limits["high"])

        # Overwrite the invalide parts of the states with the current state.
        state[invalid_states_mask] = self.ptsd_state[invalid_states_mask]

        # Get the corresponding pose from the state.
        pose = self.pivot_transform(state)

        # Save info about violation of state limits.
        self.last_set_state_violated_state_limits = np.any(invalid_states_mask)

        # Only set the pose, if all components are within the Cartesian workspace.
        if not np.any((self.cartesian_workspace["low"] > pose[:3]) | (pose[:3] > self.cartesian_workspace["high"])):
            with self.motion_target_mechanical_object.position.writeable() as sofa_pose:
                # Set the first (instrument shell) and second (first point on the rope) pose in the motion target to the instrument tip's pose.
                sofa_pose[0] = pose
                sofa_pose[1] = pose

                # Get poses for the number of rope points that are within the instrument (+1 because the first is of the instrument shell).
                poses_in_instrument = np.zeros_like(sofa_pose[self.active_index + 1 :])

                # For each of these points, take the tip's pose, and translate the points into the instrument by n*segment_length for n points in the instrument.
                # This way the rope follows a linear patch in the instrument.
                for i, rope_point_pose in enumerate(poses_in_instrument):
                    translation_offset = rotated_x_axis(pose[3:]) * self.segment_length * i
                    pose_offset = np.append(translation_offset, [0.0, 0.0, 0.0, 0.0])
                    rope_point_pose[:] = pose + pose_offset

                # Set the poses for the rope points within the instrument.
                sofa_pose[self.active_index + 1 :] = poses_in_instrument

            # Only overwrite the internal value of ptsd_state, if that was successful.
            self.ptsd_state[:] = state

            # Save info about violation of Cartesian workspace limits.
            self.last_set_state_violated_workspace_limits = False
        else:
            self.last_set_state_violated_workspace_limits = True

    def reset_state(self) -> None:
        self.ptsd_state[:] = self.initial_state
        self.active_index = self.rope.num_points - 1
        self.set_state(self.ptsd_state)


def generate_ptsd_to_pose(rcm_pose: np.ndarray) -> Callable:
    """Parametrizes ``ptsd_to_pose`` with a fixed remote center of motion.

    Notes:
        In contrast to sofa_env.utils.pivot_transform generate_ptsd_to_pose, this function includes a rotation, that points
        the X axis into the instrument. This is necessary, because the main axis of the rope is X and has stiffness
        against bending and torsion.

    Args:
        rcm_pose (np.ndarray): [x, y, z, X, Y, Z] Cartesian position and XYZ euler angles of the remote center of motion.

    Returns:
        ptsd_to_pose (Callable): A parametrized ``ptsd_to_pose`` function with a fixed remote center of motion.
    """

    def ptsd_to_pose(ptsd: np.ndarray) -> np.ndarray:
        rcm_transform = np.eye(4)
        rcm_transform[:3, :3] = euler_to_rotation_matrix(rcm_pose[3:])
        rcm_transform[:3, 3] = rcm_pose[:3]

        transform = rcm_transform

        tool_rotation = np.eye(4)
        pan = ptsd[0]
        tilt = ptsd[1]
        spin = ptsd[2]
        tool_euler_angles = np.array([tilt, pan, spin])
        tool_rotation[:3, :3] = euler_to_rotation_matrix(tool_euler_angles)

        transform = transform @ tool_rotation

        tool_translation = np.eye(4)
        tool_translation[:3, 3] = np.array([0.0, 0.0, ptsd[3]])

        transform = transform @ tool_translation

        align_x_axis_rotation = np.eye(4)
        align_x_axis_rotation[:3, :3] = euler_to_rotation_matrix(np.array([0.0, 90.0, 0.0]))

        transform = transform @ align_x_axis_rotation

        return homogeneous_transform_to_pose(transform)

    return ptsd_to_pose
