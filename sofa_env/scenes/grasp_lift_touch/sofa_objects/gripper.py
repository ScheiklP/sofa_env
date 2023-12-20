import numpy as np

from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import Sofa
import Sofa.Core

from sofa_env.sofa_templates.rigid import PivotizedArticulatedInstrument, MechanicalBinding, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.solver import add_solver, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST

from sofa_env.scenes.grasp_lift_touch.sofa_objects.gallbladder import Gallbladder

GRIPPER_PLUGIN_LIST = (
    [
        "SofaDeformable",
        "SofaGeneralRigid",
    ]
    + RIGID_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
)


class Gripper(Sofa.Core.Controller, PivotizedArticulatedInstrument):
    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        visual_mesh_path_shaft: Union[str, Path],
        visual_mesh_path_jaw: Union[str, Path],
        gallbladder_to_grasp: Gallbladder,
        ptsd_state: np.ndarray = np.zeros(4, dtype=np.float32),
        rcm_pose: np.ndarray = np.zeros(6, dtype=np.float32),
        collision_spheres_config: dict = {
            "positions": [[0, 0, 5 + i * 2] for i in range(12)],
            "radii": [1] * 12,
        },
        max_grasp_distance: float = 3.2,
        max_grasp_force: Optional[float] = None,
        angle: float = 0.0,
        angle_limits: Tuple[float, float] = (0.0, 60.0),
        total_mass: Optional[float] = None,
        rotation_axis: Tuple[int, int, int] = (1, 0, 0),
        scale: float = 1.0,
        add_solver_func: Callable = add_solver,
        add_visual_model_func: Callable = partial(add_visual_model, color=(0, 0, 1)),
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        show_object: bool = False,
        show_object_scale: float = 1.0,
        mechanical_binding: MechanicalBinding = MechanicalBinding.SPRING,
        spring_stiffness: float = 1e8,
        angular_spring_stiffness: float = 1e8,
        articulation_spring_stiffness: float = 1e15,
        spring_stiffness_grasping: float = 1e9,
        angular_spring_stiffness_grasping: float = 1e9,
        angle_to_grasp_threshold: float = 10.0,
        angle_to_release_threshold: float = 15.0,
        collision_group: int = 0,
        collision_contact_stiffness: Union[int, float] = 100,
        collision_deactivated_steps: int = 10,
        cartesian_workspace: Dict = {
            "low": np.array([-100.0, -100.0, -100.0]),
            "high": np.array([100.0, 100.0, 100.0]),
        },
        ptsd_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        angle_reset_noise: Optional[Union[float, Dict[str, float]]] = None,
        state_limits: Dict = {
            "low": np.array([-75.0, -40.0, -1000.0, 12.0]),
            "high": np.array([75.0, 75.0, 1000.0, 300.0]),
        },
        initial_state_limits: Dict = {
            "low": np.array([-20.0, 25.0, -35.0, 120.0]),
            "high": np.array([-10.0, 35.0, -25.0, 140.0]),
        },
        show_remote_center_of_motion: bool = False,
        deactivate_collision_while_grasped: bool = True,
    ) -> None:
        Sofa.Core.Controller.__init__(self)
        self.name: Sofa.Core.DataString = f"{name}_controller"
        self.gripper_node = parent_node.addChild(f"{name}_node")

        self.initial_state_limits = initial_state_limits

        # When to trigger grasping
        self.angle_to_grasp_threshold = angle_to_grasp_threshold

        # When to trigger releasing
        self.angle_to_release_threshold = angle_to_release_threshold

        PivotizedArticulatedInstrument.__init__(
            self,
            parent_node=self.gripper_node,
            name=f"{name}_instrument",
            ptsd_state=ptsd_state,
            rcm_pose=rcm_pose,
            cartesian_workspace=cartesian_workspace,
            state_limits=state_limits,
            visual_mesh_path_shaft=visual_mesh_path_shaft,
            visual_mesh_paths_jaws=[visual_mesh_path_jaw],
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
            articulation_spring_stiffness=articulation_spring_stiffness,
            collision_group=collision_group,
            ptsd_reset_noise=ptsd_reset_noise,
            angle_reset_noise=angle_reset_noise,
        )

        self.gallbladder = gallbladder_to_grasp
        self.max_grasp_distance = max_grasp_distance
        self.max_grasp_force = max_grasp_force
        self.collision_deactivated_steps = collision_deactivated_steps
        self.step_counter = 0
        self.spring_stiffness_grasping = spring_stiffness_grasping
        self.angular_spring_stiffness_grasping = angular_spring_stiffness_grasping

        if show_remote_center_of_motion:
            self.gripper_node.addObject(
                "MechanicalObject",
                template="Rigid3d",
                position=self.pivot_transform((0, 0, 0, 0)),
                showObject=show_object,
                showObjectScale=show_object_scale,
            )

        # Visualize the remote center of motion
        if show_object:
            rcm_node = parent_node.addChild(f"{self.name.value}_rcm")
            rcm_node.addObject("MechanicalObject", template="Rigid3d", translation=rcm_pose[:3], rotation=rcm_pose[3:], showObject=show_object, showObjectScale=show_object_scale)

        # Define the z positions of the sphere collision models
        self.num_spheres = len(collision_spheres_config["positions"])

        # Add sphere collision models to the gripper jaw amd the static gripper jaw
        self.collision_node_shaft = self.physical_shaft_node.addChild("approximated_collision_shaft_jaw")
        self.collision_node_jaw = self.physical_jaw_node.addChild("approximated_collision_jaw_0")

        # Add MechanicalObjects to both jaws
        self.collision_mechanical_object = {
            "shaft": self.collision_node_shaft.addObject("MechanicalObject", template="Vec3d", position=collision_spheres_config["positions"]),
            "jaw": self.collision_node_jaw.addObject("MechanicalObject", template="Vec3d", position=collision_spheres_config["positions"]),
        }

        # Add CollisionModel, and RigidMapping to jaw 0
        self.sphere_collisions_shaft = self.collision_node_shaft.addObject("SphereCollisionModel", radius=[1] * self.num_spheres, group=0, contactStiffness=collision_contact_stiffness)
        self.collision_node_shaft.addObject("RigidMapping", input=self.physical_shaft_mechanical_object.getLinkPath(), index=0)

        # Add CollisionModel, and RigidMapping to jaw 1
        self.sphere_collisions_jaw = self.collision_node_jaw.addObject("SphereCollisionModel", radius=[1] * self.num_spheres, group=0, contactStiffness=collision_contact_stiffness)
        self.collision_node_jaw.addObject("RigidMapping", input=self.joint_mechanical_object.getLinkPath(), index=1)

        self.contact_listener = {
            "shaft": self.gripper_node.addObject(
                "ContactListener",
                name="contact_listener_gallbladder_shaft",
                collisionModel1=self.gallbladder.collision_model_node.TriangleCollisionModel.getLinkPath(),
                collisionModel2=self.sphere_collisions_shaft.getLinkPath(),
            ),
            "jaw": self.gripper_node.addObject(
                "ContactListener",
                name="contact_listener_gallbladder_jaw",
                collisionModel1=self.gallbladder.collision_model_node.TriangleCollisionModel.getLinkPath(),
                collisionModel2=self.sphere_collisions_jaw.getLinkPath(),
            ),
        }

        # Define stiff springs to handle grasping the cloth.
        # Points on the cloth are set to get the right number of springs.
        # The correct indices are determined during grasping.
        self.grasping_force_field = {
            "shaft": self.gallbladder.collision_model_node.addObject(
                "RestShapeSpringsForceField",
                name="grasping_force_field_gripper_shaft",
                external_rest_shape=self.collision_mechanical_object["shaft"].getLinkPath(),
                drawSpring=True,
                stiffness=np.zeros(self.num_spheres),
                angularStiffness=np.zeros(self.num_spheres),
                listening=True,
                points=list(range(self.num_spheres)),
                external_points=list(range(self.num_spheres)),
            ),
            "jaw": self.gallbladder.collision_model_node.addObject(
                "RestShapeSpringsForceField",
                name="grasping_force_field_gripper_jaw",
                external_rest_shape=self.collision_mechanical_object["jaw"].getLinkPath(),
                drawSpring=True,
                stiffness=np.zeros(self.num_spheres),
                angularStiffness=np.zeros(self.num_spheres),
                listening=True,
                points=list(range(self.num_spheres)),
                external_points=list(range(self.num_spheres)),
            ),
        }

        self.collision_deactivated_steps = collision_deactivated_steps
        self.deactivate_collision_while_grasped = deactivate_collision_while_grasped
        self.grasp_established = False

    def onAnimateBeginEvent(self, _) -> None:
        # Reactivate the collision models after collision_deactivated_steps steps
        if self.step_counter > 0:
            self.step_counter -= 1
            if self.step_counter == 0:
                self.sphere_collisions_shaft.active = True
                self.sphere_collisions_jaw.active = True

        self.grasping_active = self.get_angle() < self.angle_to_grasp_threshold

        if self.grasping_active and self.grasp_established:
            self.identify_execessive_force()
        if self.grasping_active and not self.grasp_established:
            self.grasp_established = self._attempt_grasp()
        elif not self.grasping_active and self.grasp_established:
            self._release()
            self.grasp_established = False
        else:
            pass

    def identify_execessive_force(self):
        if self.max_grasp_force is not None:
            for jaw in self.contact_listener:
                indices = self.grasping_force_field[jaw].points.array()
                stiffness = self.grasping_force_field[jaw].stiffness.array()
                active_grasps = stiffness > 0
                jaw_forces = self.collision_mechanical_object["jaw"].force.array()[active_grasps]
                gallbladder_forces = self.gallbladder.collision_model_node.MechanicalObject.force.array()[indices[active_grasps]]
                excessive_jaw_indices = np.argwhere(np.abs(jaw_forces) > self.max_grasp_force)
                excessive_gallbladder_indices = np.argwhere(np.abs(gallbladder_forces) > self.max_grasp_force)
                if excessive_jaw_indices.any() or excessive_gallbladder_indices.any():
                    [print(f"Gallbladder: {gallbladder_forces[tuple(index)]}") for index in excessive_gallbladder_indices]
                    [print(f"Jaw: {jaw_forces[tuple(index)]}") for index in excessive_jaw_indices]

    def _attempt_grasp(self):
        mapping = []
        for jaw, contact_listener in self.contact_listener.items():
            contacts = contact_listener.getContactElements()

            contact_points = contact_listener.getContactPoints()

            # [[index, distance], [index, distance], ...]
            best_mappings = np.zeros((self.num_spheres, 2))
            best_mappings[:, 1] = np.inf
            mapping.append(best_mappings)

            for id, contact in enumerate(contacts):
                triangle_index_on_gallbladder = contact[1] if contact[0] == 0 else contact[3]
                sphere_index_on_jaw = contact[3] if contact[0] == 0 else contact[1]

                triangle_point_index = self.gallbladder.collision_model_node.loader.triangles.array()[triangle_index_on_gallbladder][0]

                gallbladder_position = self.gallbladder.collision_model_node.MechanicalObject.position.array()[triangle_point_index]
                sphere_position = np.array(list(contact_points[id][1]))

                distance = np.linalg.norm(gallbladder_position - sphere_position)
                if distance < best_mappings[sphere_index_on_jaw, 1] and distance < self.max_grasp_distance:
                    best_mappings[sphere_index_on_jaw, 0] = triangle_point_index
                    best_mappings[sphere_index_on_jaw, 1] = distance

        found_contacts_on_both_jaws = np.any(np.where(mapping[0][:, 1] < np.inf, True, False)) and np.any(np.where(mapping[1][:, 1] < np.inf, True, False))
        if found_contacts_on_both_jaws:
            for num, jaw in enumerate(self.contact_listener):
                with self.grasping_force_field[jaw].points.writeable() as indices, self.grasping_force_field[jaw].stiffness.writeable() as stiffness, self.grasping_force_field[jaw].angularStiffness.writeable() as angular_stiffness:
                    indices[:] = mapping[num][:, 0]  # gallbladder indices, mapped to the sphere indices
                    stiffness[:] = np.where(mapping[num][:, 1] < np.inf, self.spring_stiffness_grasping, 0)  # set stiffness for those with a non-inf distance
                    angular_stiffness[:] = np.where(mapping[num][:, 1] < np.inf, self.angular_spring_stiffness_grasping, 0)  # set stiffness for those with a non-inf distance
            if self.deactivate_collision_while_grasped:
                self.sphere_collisions_shaft.active.value = False
                self.sphere_collisions_jaw.active.value = False

        return found_contacts_on_both_jaws

    def _release(self):
        self.step_counter = self.collision_deactivated_steps
        self.sphere_collisions_shaft.active = False
        self.sphere_collisions_jaw.active = False

        for jaw in self.contact_listener:
            with self.grasping_force_field[jaw].points.writeable() as indices, self.grasping_force_field[jaw].stiffness.writeable() as stiffness, self.grasping_force_field[jaw].angularStiffness.writeable() as angular_stiffness:
                indices[:] = np.zeros(self.num_spheres)
                stiffness[:] = np.zeros(self.num_spheres)
                angular_stiffness[:] = np.zeros(self.num_spheres)

        self.grasp_established = False

    def get_number_of_springs(self):
        number_springs = 0
        for jaw in self.contact_listener:
            number_springs = number_springs + np.count_nonzero(self.grasping_force_field[jaw].stiffness.array())
        return number_springs

    def do_action(self, action: np.ndarray) -> None:
        """Combine state (python) and pose (sofa) update in one function to move the tool.

        Args:
            action: Action in the form of an array with ptsda values.
        """

        self.set_state(state=self.ptsd_state + action[:4])
        self.set_angle(angle=self.get_angle() + action[-1])
        self.articulated_state[:4] = self.ptsd_state
        self.articulated_state[-1] = self.get_angle()

    def set_articulated_state(self, state: np.ndarray) -> None:
        """Set the state of the articulated instrument.

        Args:
            state: The state to set the articulated instrument to.
        """
        self.set_state(state=state[:4])
        self.set_angle(angle=state[-1])
        self.articulated_state[:4] = self.ptsd_state
        self.articulated_state[-1] = self.get_angle()

    def reset_gripper(self) -> None:
        """Resets the gripper state and releases any potential grasps."""

        PivotizedArticulatedInstrument.reset_state(self)

        self._release()

        self.sphere_collisions_shaft.active.value = True
        self.sphere_collisions_jaw.active.value = True

    def get_ptsda_state(self):
        """Returns the current state of the instrument in the form of a ptsda state."""
        return self.articulated_state.copy()

    def seed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Creates a random number generator from a seed."""
        self.rng = np.random.default_rng(seed)

    def get_grasp_center_position(self) -> np.ndarray:
        """Reads the mean position of the collision models on the jaws."""
        return np.mean((self.collision_mechanical_object["shaft"].position.array(), self.collision_mechanical_object["jaw"].position.array()), axis=(0, 1))
