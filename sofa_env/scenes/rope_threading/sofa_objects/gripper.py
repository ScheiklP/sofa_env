import numpy as np

from pathlib import Path
from copy import deepcopy
from typing import Tuple, Optional, Union, Callable, List, Dict

import Sofa.Core
import Sofa.SofaDeformable
from sofa_env.sofa_templates.motion_restriction import add_bounding_box

from sofa_env.sofa_templates.rigid import ArticulatedInstrument, MechanicalBinding, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.rope import Rope
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.solver import add_solver, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST
from sofa_env.utils.math_helper import conjugate_quaternion, multiply_quaternions, point_rotation_by_quaternion, quaternion_from_vectors, rotated_y_axis, rotated_z_axis

from sofa_env.utils.pivot_transform import generate_ptsd_to_pose


GRIPPER_PLUGIN_LIST = (
    [
        "Sofa.Component.SolidMechanics.Spring",  # <- [RestShapeSpringsForceField]
        "Sofa.Component.Mapping.NonLinear",  # [RigidMapping]
    ]
    + RIGID_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
)


def add_shaft_collision_model_func(
    attached_to: Sofa.Core.Node,
    self: ArticulatedInstrument,
    collision_group: int,
) -> Sofa.Core.Node:
    name = "collision_shaft"
    shaft_radius = 2.25
    shaft_length = 125
    sphere_collision_positions = np.arange(
        start=-shaft_radius,
        stop=shaft_length + shaft_radius,
        step=2 * shaft_radius,
    )
    sphere_collision_positions = np.column_stack([np.zeros_like(sphere_collision_positions), np.zeros_like(sphere_collision_positions), -sphere_collision_positions])
    collision_node = attached_to.addChild(name)
    collision_node.addObject(
        "MechanicalObject",
        template="Vec3d",
        position=sphere_collision_positions,
    )
    collision_node.addObject(
        "SphereCollisionModel",
        group=collision_group,
        radius=shaft_radius,
    )
    collision_node.addObject(
        "RigidMapping",
        input=self.physical_shaft_mechanical_object.getLinkPath(),
        output=collision_node.MechanicalObject.getLinkPath(),
    )
    return collision_node


class ArticulatedGripper(Sofa.Core.Controller, ArticulatedInstrument):
    """
    TODO:
        - deactivate_collision_while_grasped -> change that to setting the collision group to the same on as the rope. (Currently not possible because set value of collision group is ignored in SofaPython3.)
        - inherit from PivotizedArticulatedInstrument
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        visual_mesh_path_shaft: Union[str, Path],
        visual_mesh_paths_jaws: List[Union[str, Path]],
        rope_to_grasp: Rope,
        ptsd_state: np.ndarray = np.zeros(4),
        rcm_pose: np.ndarray = np.zeros(6),
        collision_spheres_config: dict = {
            "positions": [[0, 0, 5 + i * 2] for i in range(10)],
            "backside": [[0, -1.5, 5 + i * 2] for i in range(10)],
            "radii": [1] * 10,
        },
        jaw_length: float = 25.0,
        angle: float = 0.0,
        angle_limits: Tuple[float, float] = (0.0, 60.0),
        total_mass: Optional[float] = None,
        rotation_axis: Tuple[int, int, int] = (1, 0, 0),
        scale: float = 1.0,
        add_solver_func: Callable = add_solver,
        add_shaft_collision_model_func=add_shaft_collision_model_func,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        show_object: bool = False,
        show_object_scale: float = 1.0,
        mechanical_binding: MechanicalBinding = MechanicalBinding.SPRING,
        spring_stiffness: Optional[float] = 1e8,
        angular_spring_stiffness: Optional[float] = 1e8,
        articulation_spring_stiffness: Optional[float] = 1e15,
        spring_stiffness_grasping: Optional[float] = 1e9,
        angular_spring_stiffness_grasping: Optional[float] = 1e9,
        angle_to_grasp_threshold: float = 10.0,
        angle_to_release_threshold: float = 15.0,
        collision_group: int = 0,
        collision_contact_stiffness: Union[int, float] = 100,
        cartesian_workspace: Dict = {
            "low": np.array([-np.inf, -np.inf, -np.inf]),
            "high": np.array([np.inf, np.inf, np.inf]),
        },
        ptsd_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        rcm_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        angle_reset_noise: Optional[Union[float, Dict[str, float]]] = None,
        grasp_index_reset_noise: Optional[Union[int, Dict[str, int]]] = None,
        state_limits: Dict = {
            "low": np.array([-90, -90, -90, 0]),
            "high": np.array([90, 90, 90, 200]),
        },
        show_remote_center_of_motion: bool = False,
        start_grasped: bool = False,
        recalculate_orientation_offset: bool = True,
        grasp_index_pair: Tuple[int, int] = (0, 0),
        deactivate_collision_while_grasped: bool = False,
    ) -> None:
        Sofa.Core.Controller.__init__(self)
        self.name: Sofa.Core.DataString = f"{name}_controller"

        self.gripper_node = parent_node.addChild(f"{name}_node")

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

        ArticulatedInstrument.__init__(
            self,
            parent_node=self.gripper_node,
            name=f"{name}_instrument",
            pose=self.initial_pose,
            visual_mesh_path_shaft=visual_mesh_path_shaft,
            visual_mesh_paths_jaws=visual_mesh_paths_jaws,
            angle=angle,
            angle_limits=angle_limits,
            total_mass=total_mass,
            two_jaws=True,
            rotation_axis=rotation_axis,
            scale=scale,
            add_solver_func=add_solver_func,
            add_shaft_collision_model_func=add_shaft_collision_model_func,
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

        self.cartesian_workspace = cartesian_workspace
        self.state_limits = state_limits

        self.last_set_state_violated_state_limits = False
        self.last_set_state_violated_workspace_limits = False

        self.ptsd_reset_noise = ptsd_reset_noise
        self.rcm_reset_noise = rcm_reset_noise
        self.angle_reset_noise = angle_reset_noise
        self.grasp_index_reset_noise = grasp_index_reset_noise

        self.rope = rope_to_grasp
        self.step_counter = 0
        self.spring_stiffness_grasping = spring_stiffness_grasping
        self.angular_spring_stiffness_grasping = angular_spring_stiffness_grasping

        self.orientation_delta = np.array([0.0, 0.0, 0.0, 1.0])

        self.show_remote_center_of_motion = show_remote_center_of_motion
        if show_remote_center_of_motion:
            self.rcm_mechanical_object = self.gripper_node.addObject(
                "MechanicalObject",
                template="Rigid3d",
                position=self.pivot_transform((0, 0, 0, 0)),
                showObject=show_object,
                showObjectScale=show_object_scale,
            )

        # Visualize the jaw lenght that is used during grasp attempts to figure out whether a point is between the gripper jaws
        # and the workspace
        if show_object:
            jaw_length_node = parent_node.addChild(f"{self.name.value}_cone")
            cone_base_point = self.initial_pose[:3] + point_rotation_by_quaternion(np.array([0, 0, jaw_length]), np.array(self.initial_pose[3:]))
            jaw_length_node.addObject("MechanicalObject", template="Vec3d", position=[cone_base_point], showObject=show_object, showObjectScale=show_object_scale, showColor=(0, 0, 1))
            add_bounding_box(jaw_length_node, min=cartesian_workspace["low"], max=cartesian_workspace["high"], show_bounding_box=True)

        # Add sphere collision models to the gripper jaws
        self.collision_node_jaw_0 = self.physical_jaw_node.addChild("collision_jaw_0")
        self.collision_node_jaw_1 = self.physical_jaw_node.addChild("collision_jaw_1")
        self.deactivate_collision_while_grasped = deactivate_collision_while_grasped

        self.jaw_length = jaw_length

        # Define the z positions of the sphere collision models
        self.num_spheres = len(collision_spheres_config["positions"])

        # Add Rigid3d points that will be used for grasping
        self.grasping_node = self.gripper_node.addChild("grasping")

        # A reference object that has as many points as points as there are collisions models, and is mapped to the motion target
        motion_reference_node = self.grasping_node.addChild("motion_reference")
        self.motion_mapping_mechanical_object = motion_reference_node.addObject(
            "MechanicalObject",
            template="Vec3d",
            position=[position for position in collision_spheres_config["positions"]],
            showObject=show_object,
            showObjectScale=show_object_scale / 2,
        )
        motion_reference_node.addObject("RigidMapping", input=self.physical_shaft_mechanical_object.getLinkPath())

        # The MechanicalObject that is used for grasping by introducing springs. No mapping, because position and orientation are set in onAnimateBeginEvent
        self.grasping_mechanical_object = self.grasping_node.addObject(
            "MechanicalObject",
            template="Rigid3d",
            position=[append_orientation(position) for position in collision_spheres_config["positions"]],
            showObject=show_object,
            showObjectScale=show_object_scale / 2,
        )

        # Add MechanicalObjects to both jaws
        self.collision_mechanical_object = {
            "jaw_0": self.collision_node_jaw_0.addObject("MechanicalObject", template="Vec3d", position=collision_spheres_config["positions"]),
            "jaw_1": self.collision_node_jaw_1.addObject("MechanicalObject", template="Vec3d", position=collision_spheres_config["positions"]),
        }

        extra_collision_kwargs = {}
        if animation_loop_type == AnimationLoopType.DEFAULT:
            extra_collision_kwargs["contactStiffness"] = collision_contact_stiffness

        # Add CollisionModel, and RigidMapping to jaw 0
        self.sphere_collisions_jaw_0 = self.collision_node_jaw_0.addObject(
            "SphereCollisionModel",
            radius=[1] * self.num_spheres,
            group=0 if collision_group is None else collision_group,
            **extra_collision_kwargs,
        )
        self.collision_node_jaw_0.addObject(
            "RigidMapping",
            input=self.joint_mechanical_object.getLinkPath(),
            index=1,
        )

        # Add CollisionModel, and RigidMapping to jaw 1
        self.sphere_collisions_jaw_1 = self.collision_node_jaw_1.addObject(
            "SphereCollisionModel",
            radius=[1] * self.num_spheres,
            group=0 if collision_group is None else collision_group,
            **extra_collision_kwargs,
        )
        self.collision_node_jaw_1.addObject(
            "RigidMapping",
            input=self.joint_mechanical_object.getLinkPath(),
            index=2,
        )

        # Create ContactListener between the SphereCollisionModel of the jaws and the SphereCollisionModel of the rope
        self.contact_listener = {
            "jaw_0": self.gripper_node.addObject(
                "ContactListener",
                name=f"contact_listener_{rope_to_grasp.name}_jaw_0",
                collisionModel1=self.rope.sphere_collision_models.getLinkPath(),
                collisionModel2=self.sphere_collisions_jaw_0.getLinkPath(),
            ),
            "jaw_1": self.gripper_node.addObject(
                "ContactListener",
                name=f"contact_listener_{rope_to_grasp.name}_jaw_1",
                collisionModel1=self.rope.sphere_collision_models.getLinkPath(),
                collisionModel2=self.sphere_collisions_jaw_1.getLinkPath(),
            ),
        }

        # Add collision models to the backside of the gripper
        self.gripper_backside_collision_models = []
        gripper_backside_jaw_0 = self.physical_jaw_node.addChild("collision_backside_jaw_0")
        gripper_backside_jaw_1 = self.physical_jaw_node.addChild("collision_backside_jaw_1")
        gripper_backside_jaw_0.addObject("MechanicalObject", template="Vec3d", position=collision_spheres_config["backside"])
        gripper_backside_jaw_1.addObject("MechanicalObject", template="Vec3d", position=[point * np.array([1.0, -1.0, 1.0]) for point in collision_spheres_config["backside"]])
        self.gripper_backside_collision_models.append(
            gripper_backside_jaw_0.addObject(
                "SphereCollisionModel",
                radius=[1] * len(collision_spheres_config["backside"]),
                group=0 if collision_group is None else collision_group,
                **extra_collision_kwargs,
            )
        )
        gripper_backside_jaw_0.addObject("RigidMapping", input=self.joint_mechanical_object.getLinkPath(), index=1)
        self.gripper_backside_collision_models.append(
            gripper_backside_jaw_1.addObject(
                "SphereCollisionModel",
                radius=[1] * len(collision_spheres_config["backside"]),
                group=0 if collision_group is None else collision_group,
                **extra_collision_kwargs,
            )
        )
        gripper_backside_jaw_1.addObject("RigidMapping", input=self.joint_mechanical_object.getLinkPath(), index=2)

        # When to trigger grasping
        self.angle_to_grasp_threshold = angle_to_grasp_threshold

        # When to trigger releasing
        self.angle_to_release_threshold = angle_to_release_threshold

        # Define stiff springs to handle grasping the rope.
        # Points on the rope are set to get the right number of springs.
        # The correct indices are determined during grasping.
        if start_grasped:
            self.grasping_force_field = self.rope.node.addObject(
                "RestShapeSpringsForceField",
                name=f"grasping_force_field_{name}",
                external_rest_shape=self.grasping_mechanical_object.getLinkPath(),
                drawSpring=True,
                stiffness=self.spring_stiffness_grasping,
                angularStiffness=self.angular_spring_stiffness_grasping,
                listening=True,
                points=grasp_index_pair[1],
                external_points=grasp_index_pair[0],
            )

            self.grasping_active = True
            self.grasp_established = True
            self.lost_grasp = False
            if self.deactivate_collision_while_grasped:
                self.sphere_collisions_jaw_0.active.value = False
                self.sphere_collisions_jaw_1.active.value = False
                self.gripper_backside_collision_models[0].active.value = False
                self.gripper_backside_collision_models[1].active.value = False

        else:
            self.grasping_force_field = self.rope.node.addObject(
                "RestShapeSpringsForceField",
                name=f"grasping_force_field_{name}",
                external_rest_shape=self.grasping_mechanical_object.getLinkPath(),
                drawSpring=True,
                stiffness=[0],
                angularStiffness=[0],
                listening=True,
                points=[0],
                external_points=[0],
            )

            self.grasping_active = False
            self.grasp_established = False
            self.lost_grasp = False

        self.start_grasped = start_grasped
        self.recalculate_orientation_offset = recalculate_orientation_offset
        self.grasp_index_pair = grasp_index_pair

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
            angle = self.get_angle() - 1
            self.set_angle(angle)

        elif key == "P":
            angle = self.get_angle() + 1
            self.set_angle(angle)

        elif ord(key) == 32:  # space
            print(repr(self.ptsd_state), self.get_angle())
        else:
            pass

    def onAnimateBeginEvent(self, event) -> None:
        """Function called at the beginning of every simulation step."""

        # The grasping_mechanical_object is not mapped to the motion_target_mechanical_object.
        # Instead, the motion_mapping_mechanical_object is a Vec3d MechanicalObject that gets the mapped positions
        # from the motion_target_mechanical_object.
        # Here, we then write the positions of the motion_mapping_mechanical_object into the grasping_mechanical_object,
        # and set the orientation to the one of the motion_target_mechanical_object, but rotate it with an orientation_delta,
        # which is updated when the rope is grasped.
        # The rotation difference C between quaternions A and B is
        # C = A * inverse(B);
        # To add the difference to D
        # D = C * D;

        with self.grasping_mechanical_object.position.writeable() as grasping_frames:
            grasping_frames[:, :3] = self.motion_mapping_mechanical_object.position.array()
            grasping_frames[:, 3:] = multiply_quaternions(
                self.motion_target_mechanical_object.position.array()[0, 3:],
                self.orientation_delta,
            )

        # Check if angle is small enough to be grasping
        self.grasping_active = self.get_actual_angle() < self.angle_to_grasp_threshold
        self.lost_grasp = False

        if self.grasping_active and not self.grasp_established:
            # TODO introduce a check, that the angle is big enough to actually grasp the rope.
            # At the moment, the gripper can just "sweep" over the rope, fully closed, and still grasp.
            # Grasp
            self.grasp_established = self._attempt_grasp()

        elif not self.grasping_active and self.grasp_established:
            # Release
            self._release_grasp()

        elif self.grasping_active and self.grasp_established:
            # Check if motion target and physical body deviate in the opening of the gripper,
            # and release the grasp, if necessary
            if self.get_actual_angle() > self.angle_to_release_threshold:
                self._release_grasp()
                self.lost_grasp = True

    def _release_grasp(self) -> None:
        """Release the rope by setting all rope indices in the RestShapeSpringsForceField and their stiffness to zero."""
        with self.grasping_force_field.stiffness.writeable() as stiffness, self.grasping_force_field.angularStiffness.writeable() as angular_stiffness:
            stiffness[0] = 0.0
            angular_stiffness[0] = 0.0

        self.grasp_established = False

        if self.deactivate_collision_while_grasped:
            self.sphere_collisions_jaw_0.active.value = True
            self.sphere_collisions_jaw_1.active.value = True
            self.gripper_backside_collision_models[0].active.value = True
            self.gripper_backside_collision_models[1].active.value = True

    def _attempt_grasp(self) -> bool:
        """Try to grasp the rope.

        Steps:
            1. Look for collisions between rope and gripper jaws
            2. Check if these collisions happened between the jaws
            3. Keep only the contacts with the smallest distance between rope and jaws
            4. Set spring stiffnesses, if both jaws have valid collisions with the rope
        """

        jaw_poses = self.joint_mechanical_object.position.array()[1:]
        jaw_length = self.jaw_length
        gripper_pose = self.get_physical_pose()

        jaw_has_contact = {jaw: False for jaw in self.contact_listener}
        # rope_index, gripper_index, distance
        best_mapping = [0, 0, np.inf]
        point_orientation = np.zeros(4)

        # Look for collisions between the rope and the jaws
        for jaw, contact_listener in self.contact_listener.items():

            # Get contact data from the contact listener
            contacts = contact_listener.getContactElements()

            # Look for the closest collisions between jaw and rope collision models
            for contact in contacts:
                # Get the point indices of elements in collision
                # (object, index, object, index)
                point_index_on_rope = contact[1] if contact[0] == 0 else contact[3]
                sphere_index_on_jaw = contact[3] if contact[0] == 0 else contact[1]

                # Look up the corresponding index on the FEM node (because there are the springs)
                point_index_on_rope = self.rope.collision_node.SubsetMapping.indices.array()[point_index_on_rope]

                # Position on the rope
                rope_pose = self.rope.mechanical_object.position.array()[point_index_on_rope]
                rope_position = rope_pose[:3]

                # Position on the gripper
                grasping_point_position = self.grasping_mechanical_object.position.array()[sphere_index_on_jaw][:3]

                # Check if the detected collision happened between the gripper jaws
                rope_point_is_in_gripper = point_is_in_grasp_cone(gripper_pose=gripper_pose, jaw_poses=jaw_poses, jaw_length=jaw_length, query_point=rope_position)
                # rope_point_is_in_gripper = True

                # Check if the detected collision is closer than previously detected ones
                if rope_point_is_in_gripper:
                    jaw_has_contact[jaw] = True
                    distance = np.linalg.norm(rope_position - grasping_point_position)
                    if distance < best_mapping[2]:
                        best_mapping[2] = distance
                        best_mapping[1] = sphere_index_on_jaw
                        best_mapping[0] = point_index_on_rope
                        point_orientation[:] = rope_pose[3:]

        # If there were contacts between rope and both jaws -> grasp
        found_contacts_on_both_jaws = all(jaw_has_contact.values())
        if found_contacts_on_both_jaws:
            # Calculate the new orientation difference between the rope, and the gripper's motion target
            self.orientation_delta[:] = multiply_quaternions(
                conjugate_quaternion(self.motion_target_mechanical_object.position.array()[0, 3:].copy()),
                point_orientation,
            )

            # Find the quaternion, that rotates the transformed (by orientation_delta) y axis into the original y axis.
            transformed_y_axis = rotated_y_axis(self.orientation_delta)
            original_y_axis = np.array([0.0, 1.0, 0.0])
            if not np.allclose(transformed_y_axis, original_y_axis):
                rotation_into_local_y_axis = quaternion_from_vectors(transformed_y_axis, original_y_axis)
                # Add this rotation to the orientation_delta to "undo" the part of the original orientation_delta that rotates
                # the y axis out of alignment -> orientation_delta will now only rotate around y of the gripper jaws
                # TODO if the gripper is flipped 180 degrees, so that y points in the other direction, the rope is flipped aswell on a grasp.
                # Should be changed so that -y is also valid and the smaller rotation is used.
                self.orientation_delta[:] = multiply_quaternions(
                    rotation_into_local_y_axis,
                    self.orientation_delta,
                )

            with self.grasping_force_field.external_points.writeable() as external_indices, self.grasping_force_field.points.writeable() as indices, self.grasping_force_field.stiffness.writeable() as stiffness, self.grasping_force_field.angularStiffness.writeable() as angular_stiffness:
                indices[0] = best_mapping[0]
                external_indices[0] = best_mapping[1]
                stiffness[:] = self.spring_stiffness_grasping
                angular_stiffness[:] = self.angular_spring_stiffness_grasping

            if self.deactivate_collision_while_grasped:
                self.sphere_collisions_jaw_0.active.value = False
                self.sphere_collisions_jaw_1.active.value = False
                self.gripper_backside_collision_models[0].active.value = False
                self.gripper_backside_collision_models[1].active.value = False

        return found_contacts_on_both_jaws

    def onAnimateEndEvent(self, event) -> None:
        pass

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

    def set_angle(self, angle: float) -> None:
        """Sets the angle of the instrument."""
        # Do not close the gripper further, if a grasp is already established
        if self.grasp_established:
            if angle > self.get_angle():
                super().set_angle(angle)
        else:
            super().set_angle(angle)

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

    def get_grasp_center_position(self) -> np.ndarray:
        """Reads the position of the middle of the jaws."""
        return self.grasping_mechanical_object.position.array()[int(len(self.grasping_mechanical_object.position) / 2), :3]

    def get_collision_object_positions(self) -> np.ndarray:
        """Get the Cartesian positions of the SphereCollisionModels on both jaws."""
        positions_jaw_0 = self.collision_mechanical_object["jaw_0"].position.array()
        positions_jaw_1 = self.collision_mechanical_object["jaw_1"].position.array()

        return np.concatenate([positions_jaw_0, positions_jaw_1])

    def reset_gripper(
        self,
        rcm_pose: Optional[np.ndarray] = None,
        state: Optional[np.ndarray] = None,
        angle: Optional[float] = None,
    ) -> None:

        #############
        # Reset state
        #############
        def reset_rcm_pose(new_rcm_pose: np.ndarray):
            self.pivot_transform = generate_ptsd_to_pose(new_rcm_pose)
            self.remote_center_of_motion[:] = new_rcm_pose
            if self.show_remote_center_of_motion:
                with self.rcm_mechanical_object.position.writeable() as rcm_pose:
                    rcm_pose[:] = self.pivot_transform((0, 0, 0, 0))

        if rcm_pose is not None:
            reset_rcm_pose(rcm_pose)
        elif self.rcm_reset_noise is not None:
            # Generate a new pivot_transform by adding noise to the initial remote_center_of_motion pose
            if isinstance(self.rcm_reset_noise, np.ndarray):
                # Uniformly sample from -noise to +noise and add it to the rcm pose
                new_rcm_pose = self.initial_remote_center_of_motion + self.rng.uniform(-self.rcm_reset_noise, self.rcm_reset_noise)
            elif isinstance(self.rcm_reset_noise, dict):
                # Uniformly sample from low to high and add it to the rcm pose
                new_rcm_pose = self.initial_remote_center_of_motion + self.rng.uniform(self.rcm_reset_noise["low"], self.rcm_reset_noise["high"])
            else:
                raise TypeError("Please pass the rcm_reset_noise as a numpy array or a dictionary with 'low' and 'high' keys.")
            reset_rcm_pose(new_rcm_pose)

        if state is not None:
            new_state = state
        elif self.ptsd_reset_noise is not None:
            # Select a new ptsd state by adding noise to the initial state
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
        if angle is not None:
            new_angle = angle
        elif self.angle_reset_noise is not None:
            # Select a new angle by adding noise to the initial angle
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

        ################
        # Reset grasping
        ################
        # Sample a new grasping index on the rope
        if self.grasp_index_reset_noise is not None:
            if isinstance(self.grasp_index_reset_noise, int):
                # Uniformly sample from -noise to +noise and add it to the initial grasping index
                new_index_on_rope = self.grasp_index_pair[1] + self.rng.integers(-self.grasp_index_reset_noise, self.grasp_index_reset_noise)
            elif isinstance(self.grasp_index_reset_noise, dict):
                # Uniformly sample from low to high and add it to the initial grasping index
                new_index_on_rope = self.grasp_index_pair[1] + self.rng.integers(self.grasp_index_reset_noise["low"], self.grasp_index_reset_noise["high"])
            else:
                raise TypeError("Please pass the grasp_index_reset_noise as an int or a dictionary with 'low' and 'high' keys.")
            # Clip the grasping index to the rope length
            new_index_on_rope = np.clip(new_index_on_rope, 0, self.rope.num_points - 1)
        else:
            new_index_on_rope = self.grasp_index_pair[1]

        # Reset the grasping state
        if self.start_grasped:
            with self.grasping_force_field.external_points.writeable() as external_indices, self.grasping_force_field.points.writeable() as indices, self.grasping_force_field.stiffness.writeable() as stiffness, self.grasping_force_field.angularStiffness.writeable() as angular_stiffness:
                indices[0] = new_index_on_rope
                external_indices[0] = self.grasp_index_pair[0]
                stiffness[:] = self.spring_stiffness_grasping
                angular_stiffness[:] = self.angular_spring_stiffness_grasping

            if self.recalculate_orientation_offset:
                # Recalculate orientation delta between gripper and rope
                self.orientation_delta[:] = multiply_quaternions(
                    conjugate_quaternion(self.rope.get_reset_state()[new_index_on_rope, 3:].copy()),
                    self.motion_target_mechanical_object.position.array()[0, 3:],
                )
                # Find the quaternion, that rotates the transformed (by orientation_delta) y axis into the original y axis.
                transformed_y_axis = rotated_y_axis(self.orientation_delta)
                original_y_axis = np.array([0.0, 1.0, 0.0])
                if not np.allclose(transformed_y_axis, original_y_axis):
                    rotation_into_local_y_axis = quaternion_from_vectors(transformed_y_axis, original_y_axis)
                    # Add this rotation to the orientation_delta to "undo" the part of the original orientation_delta that rotates
                    # the y axis out of alignment -> orientation_delta will now only rotate around y of the gripper jaws
                    self.orientation_delta[:] = multiply_quaternions(
                        rotation_into_local_y_axis,
                        self.orientation_delta,
                    )

            self.grasping_active = True
            self.grasp_established = True

            # Deactivet the collision models, if required
            if self.deactivate_collision_while_grasped:
                self.sphere_collisions_jaw_0.active.value = False
                self.sphere_collisions_jaw_1.active.value = False
                self.gripper_backside_collision_models[0].active.value = False
                self.gripper_backside_collision_models[1].active.value = False

        else:
            self._release_grasp()

    def seed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Creates a random number generator from a seed."""
        self.rng = np.random.default_rng(seed)


def point_is_in_grasp_cone(gripper_pose: np.ndarray, jaw_poses: np.ndarray, query_point: np.ndarray, jaw_length: float) -> bool:
    """Checks whether a query_point is within the cone that is spanned by the gripper jaws and the shaft end of the gripper.

    Note:
        Based on https://stackoverflow.com/questions/12826117/how-can-i-detect-if-a-point-is-inside-a-cone-or-not-in-3d-space#:~:text=To%20expand%20on%20Ignacio%27s%20answer%3A
    """

    tip_of_the_cone = gripper_pose[:3]  # TODO: could be offset a bit in the direction of the cone axis

    cone_axis = rotated_z_axis(gripper_pose[3:])

    query_points_are_in_cone = []

    for jaw_pose in jaw_poses:
        jaw_axis = rotated_z_axis(jaw_pose[3:])
        cone_height = np.dot(jaw_axis * jaw_length, cone_axis)  # scalar product of cone_axis and jaw_axis*length
        cone_radius = np.sqrt(max(np.linalg.norm(jaw_axis * jaw_length) ** 2 - cone_height**2, 0.0))  # countercathete of the jaw axis and the cone height

        distance_to_cone_axis = np.dot(query_point - tip_of_the_cone, cone_axis)

        if 0 <= distance_to_cone_axis <= cone_height:
            query_cone_radius = (distance_to_cone_axis / cone_height) * cone_radius
            orthogonal_query_distance = np.linalg.norm((query_point - tip_of_the_cone) - distance_to_cone_axis * cone_axis)
            query_point_is_in_cone = orthogonal_query_distance < query_cone_radius

        else:
            query_point_is_in_cone = False

        query_points_are_in_cone.append(query_point_is_in_cone)

    return any(query_points_are_in_cone)


def append_orientation(position: list) -> list:
    """Takes a cartesian point [x, y, z] and extends the list with a quaternion for orientation."""
    pose = deepcopy(position)
    pose.extend([0, 0, 0, 1])
    return pose
