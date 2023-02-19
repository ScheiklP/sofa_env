import numpy as np

from pathlib import Path
from typing import Tuple, Optional, Union, Callable, List, Dict

import Sofa.Core
import Sofa.SofaDeformable

from sofa_env.sofa_templates.rigid import MechanicalBinding, RIGID_PLUGIN_LIST, PivotizedArticulatedInstrument
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.solver import add_solver, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST

from sofa_env.utils.math_helper import rotated_z_axis

from sofa_env.scenes.precision_cutting.sofa_objects.cloth import Cloth


GRIPPER_PLUGIN_LIST = (
    [
        "SofaGeneralRigid",
    ]
    + RIGID_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
)


class ArticulatedGripper(Sofa.Core.Controller, PivotizedArticulatedInstrument):
    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        visual_mesh_path_shaft: Union[str, Path],
        visual_mesh_paths_jaws: List[Union[str, Path]],
        cloth_to_grasp: Cloth,
        max_grasp_distance: float = 3.2,
        angle: float = 0.0,
        angle_limits: Tuple[float, float] = (-0.0, 60.0),
        ptsd_state: np.ndarray = np.zeros(4),
        rcm_pose: np.ndarray = np.zeros(6),
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
        spring_stiffness_grasping: float = 1e9,
        angular_spring_stiffness_grasping: float = 1e9,
        collision_group: int = 0,
        collision_contact_stiffness: Union[int, float] = 100,
        collision_deactivated_steps: int = 10,
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
    ) -> None:
        Sofa.Core.Controller.__init__(self)
        self.name = f"{name}_controller"

        self.cloth = cloth_to_grasp
        self.max_grasp_distance = max_grasp_distance
        self.collision_deactivated_steps = collision_deactivated_steps
        self.step_counter = 0
        self.spring_stiffness_grasping = spring_stiffness_grasping
        self.angular_spring_stiffness_grasping = angular_spring_stiffness_grasping

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

        # Add sphere collision models to the gripper jaws
        self.collision_node_jaw_0 = self.physical_shaft_node.addChild("collision_jaw_0")
        self.collision_node_jaw_1 = self.physical_shaft_node.addChild("collision_jaw_1")

        self.jaw_length = 15

        # Define the positions of the sphere collision models
        start = np.array([0.0, 0.0, 5.0])
        stop = np.array([0.0, -0.5, 13.0])
        positions = np.linspace(start, stop, int(np.floor(np.linalg.norm(stop - start)))).tolist()
        start = np.array([0.0, -0.5, 13.0])
        stop = np.array([0.0, -4.5, 25.0])
        positions.extend(np.linspace(start, stop, int(np.floor(np.linalg.norm(stop - start)))))
        self.num_spheres = len(positions)

        # Add MechanicalObjects to both jaws
        self.collision_mechanical_object = {
            "jaw_0": self.collision_node_jaw_0.addObject("MechanicalObject", template="Vec3d", position=positions),
            "jaw_1": self.collision_node_jaw_1.addObject("MechanicalObject", template="Vec3d", position=positions),
        }

        # Add CollisionModel, and RigidMapping to jaw 0
        self.sphere_collisions_jaw_0 = self.collision_node_jaw_0.addObject("SphereCollisionModel", radius=[1] * len(positions), group=0, contactStiffness=collision_contact_stiffness)
        self.collision_node_jaw_0.addObject("RigidMapping", input=self.joint_mechanical_object.getLinkPath(), index=1)

        # Add CollisionModel, and RigidMapping to jaw 1
        self.sphere_collisions_jaw_1 = self.collision_node_jaw_1.addObject("SphereCollisionModel", radius=[1] * len(positions), group=0, contactStiffness=collision_contact_stiffness)
        self.collision_node_jaw_1.addObject("RigidMapping", input=self.joint_mechanical_object.getLinkPath(), index=2)

        # Create ContactListener between the SphereCollisionModel of the jaws and the TriangleCollisionModel of the Cloth
        self.contact_listener = {
            "jaw_0": self.node.addObject(
                "ContactListener",
                name=f"contact_listener_{cloth_to_grasp.name}_jaw_0",
                collisionModel1=self.cloth.node.PointCollisionModel.getLinkPath(),
                collisionModel2=self.sphere_collisions_jaw_0.getLinkPath(),
            ),
            "jaw_1": self.node.addObject(
                "ContactListener",
                name=f"contact_listener_{cloth_to_grasp.name}_jaw_1",
                collisionModel1=self.cloth.node.PointCollisionModel.getLinkPath(),
                collisionModel2=self.sphere_collisions_jaw_1.getLinkPath(),
            ),
        }

        # Define stiff springs to handle grasping the cloth.
        # Points on the cloth are set to get the right number of springs.
        # The correct indices are determined during grasping.
        self.grasping_force_field = {
            "jaw_0": self.cloth.node.addObject(
                "RestShapeSpringsForceField",
                name=f"grasping_force_field_{name}_jaw_0",
                external_rest_shape=self.collision_mechanical_object["jaw_0"].getLinkPath(),
                drawSpring=True,
                stiffness=np.zeros(self.num_spheres),
                angularStiffness=np.zeros(self.num_spheres),
                listening=True,
                points=list(range(self.num_spheres)),
                external_points=list(range(self.num_spheres)),
            ),
            "jaw_1": self.cloth.node.addObject(
                "RestShapeSpringsForceField",
                name=f"grasping_force_field_{name}_jaw_1",
                external_rest_shape=self.collision_mechanical_object["jaw_1"].getLinkPath(),
                drawSpring=True,
                stiffness=np.zeros(self.num_spheres),
                angularStiffness=np.zeros(self.num_spheres),
                listening=True,
                points=list(range(self.num_spheres)),
                external_points=list(range(self.num_spheres)),
            ),
        }

        self.angle_to_grasp_threshold = 20.0
        self.grasping_active = False
        self.grasp_established = False

    def onKeypressedEvent(self, event):
        key = event["key"]
        if ord(key) == 19:  # up
            pose = np.array(self.get_pose())
            pose[1] = pose[1] + 1
            self.set_pose(pose)

        elif ord(key) == 21:  # down
            pose = np.array(self.get_pose())
            pose[1] = pose[1] - 1
            self.set_pose(pose)

        elif ord(key) == 18:  # left
            pose = np.array(self.get_pose())
            pose[0] = pose[0] - 1
            self.set_pose(pose)

        elif ord(key) == 20:  # right
            pose = np.array(self.get_pose())
            pose[0] = pose[0] + 1
            self.set_pose(pose)

        elif key == "T":
            angle = np.array(self.get_angle())
            angle[0] = angle[0] + 1
            self.set_angle(angle[0])

        elif key == "G":
            angle = np.array(self.get_angle())
            angle[0] = angle[0] - 1
            self.set_angle(angle[0])

        else:
            pass

    def onAnimateBeginEvent(self, event) -> None:
        """Function called at the beginning of every simulation step."""
        # Check if angle is small enough to be grasping
        self.grasping_active = self.get_angle() < self.angle_to_grasp_threshold

        # Reactivate the collision models after collision_deactivated_steps steps
        if self.step_counter > 0:
            self.step_counter -= 1
            if self.step_counter == 0:
                self.sphere_collisions_jaw_0.active = True
                self.sphere_collisions_jaw_1.active = True

        if self.grasping_active and not self.grasp_established:
            # Grasp
            self.grasp_established = self._attempt_grasp()
        elif not self.grasping_active and self.grasp_established:
            # Release
            self._release_grasp()
            self.grasp_established = False
        else:
            # Pass if both active and established have the same boolean value
            pass

    def _release_grasp(self) -> None:
        """Release the cloth by setting all cloth indices in the RestShapeSpringsForceField and their stiffness to zero."""
        for jaw in self.contact_listener:
            with self.grasping_force_field[jaw].points.writeable() as indices, self.grasping_force_field[jaw].stiffness.writeable() as stiffness, self.grasping_force_field[jaw].angularStiffness.writeable() as angular_stiffness:
                zeros = np.zeros(self.num_spheres)
                indices[:] = zeros
                stiffness[:] = zeros
                angular_stiffness[:] = zeros

        self.step_counter = self.collision_deactivated_steps
        self.sphere_collisions_jaw_0.active = False
        self.sphere_collisions_jaw_1.active = False

    def _attempt_grasp(self) -> np.bool_:
        """Try to grasp the cloth.

        Steps:
            1. Check for collision between cloth and jaws
            2. Discard the collisions that have a distance more than self.max_grasp_distance
            3. Filter the collisions, so that only the one with the minimal distance is kept per SphereCollisionModel
            4. If there are contacts on both jaws, set the cloth indices on the RestShapeSpringsForceField and set their stiffness
        """

        jaw_pose = self.joint_mechanical_object.position.array()[1]
        jaw_length = self.jaw_length
        gripper_pose = self.get_pose()

        mappings = []

        for jaw, contact_listener in self.contact_listener.items():

            # Get contact data
            contacts = contact_listener.getContactElements()

            # [[index, distance], [index, distance], ...]
            best_mappings = np.zeros((self.num_spheres, 2))
            best_mappings[:, 1] = np.inf
            mappings.append(best_mappings)

            for contact in contacts:

                # Get the point indices of elements in collision
                # (object, index, object, index)
                point_index_on_cloth = contact[1] if contact[0] == 0 else contact[3]
                sphere_index_on_jaw = contact[3] if contact[0] == 0 else contact[1]

                cloth_position = self.cloth.mechanical_object.position.array()[point_index_on_cloth]
                sphere_position = self.collision_mechanical_object[jaw].position.array()[sphere_index_on_jaw]
                cloth_point_is_in_gripper = point_is_in_grasp_cone(gripper_pose=gripper_pose, jaw_pose=jaw_pose, jaw_length=jaw_length, query_point=cloth_position)

                if cloth_point_is_in_gripper:
                    current_best_distance = best_mappings[sphere_index_on_jaw, 1]
                    distance = np.linalg.norm(cloth_position - sphere_position)
                    if distance < current_best_distance and distance < self.max_grasp_distance:
                        best_mappings[sphere_index_on_jaw, 0] = point_index_on_cloth
                        best_mappings[sphere_index_on_jaw, 1] = distance

        found_contacts_on_both_jaws = np.any(np.where(mappings[0][:, 1] < np.inf, True, False)) and np.any(np.where(mappings[1][:, 1] < np.inf, True, False))
        if found_contacts_on_both_jaws:
            for num, jaw in enumerate(self.contact_listener):
                with self.grasping_force_field[jaw].points.writeable() as indices, self.grasping_force_field[jaw].stiffness.writeable() as stiffness, self.grasping_force_field[jaw].angularStiffness.writeable() as angular_stiffness:
                    indices[:] = mappings[num][:, 0]  # cloth indices, mapped to the sphere indices
                    stiffness[:] = np.where(mappings[num][:, 1] < np.inf, self.spring_stiffness_grasping, 0)  # set stiffness for those with a non-inf distance
                    angular_stiffness[:] = np.where(mappings[num][:, 1] < np.inf, self.angular_spring_stiffness_grasping, 0)  # set stiffness for those with a non-inf distance

        return found_contacts_on_both_jaws

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

        elif key == "H":
            print(repr(self.ptsd_state), self.get_angle())
        else:
            pass


def point_is_in_grasp_cone(gripper_pose: np.ndarray, jaw_pose: np.ndarray, query_point: np.ndarray, jaw_length: float) -> bool:
    """Checks whether a query_point is within the cone that is spanned by the gripper jaws and the shaft end of the gripper.

    Note:
        Based on https://stackoverflow.com/questions/12826117/how-can-i-detect-if-a-point-is-inside-a-cone-or-not-in-3d-space#:~:text=To%20expand%20on%20Ignacio%27s%20answer%3A
    """

    tip_of_the_cone = gripper_pose[:3]

    cone_axis = rotated_z_axis(gripper_pose[3:])
    jaw_axis = rotated_z_axis(jaw_pose[3:])
    cone_height = np.dot(jaw_axis * jaw_length, cone_axis)  # scalar product of cone_axis and jaw_axis*length
    cone_radius = np.sqrt(np.linalg.norm(jaw_axis * jaw_length) ** 2 - cone_height ** 2)  # countercathete of the jaw axis and the cone height

    distance_to_cone_axis = np.dot(query_point - tip_of_the_cone, cone_axis)

    if 0 <= distance_to_cone_axis <= cone_height:
        query_cone_radius = (distance_to_cone_axis / cone_height) * cone_radius
        orthogonal_query_distance = np.linalg.norm((query_point - tip_of_the_cone) - distance_to_cone_axis * cone_axis)
        query_point_is_in_cone = orthogonal_query_distance < query_cone_radius

    else:
        query_point_is_in_cone = False

    return query_point_is_in_cone
