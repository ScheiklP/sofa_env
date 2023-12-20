import numpy as np

from pathlib import Path
from typing import Tuple, Optional, Union, Callable, List, Dict

import Sofa.Core
import Sofa.SofaDeformable

from sofa_env.sofa_templates.rigid import ArticulatedInstrument, MechanicalBinding, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.solver import add_solver, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST

from sofa_env.utils.pivot_transform import generate_ptsd_to_pose


GRIPPER_PLUGIN_LIST = (
    RIGID_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
)


class ArticulatedGripper(Sofa.Core.Controller, ArticulatedInstrument):
    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        visual_mesh_path_shaft: Union[str, Path],
        visual_mesh_paths_jaws: List[Union[str, Path]],
        ptsd_state: np.ndarray = np.zeros(4),
        rcm_pose: np.ndarray = np.zeros(6),
        collision_spheres_config: dict = {
            "positions": [[0, 0, 5 + i * 2] for i in range(10)],
            "radii": [1] * 10,
        },
        angle: float = 0.0,
        angle_limits: Tuple[float, float] = (0.0, 60.0),
        total_mass: Optional[float] = None,
        rotation_axis: Tuple[int, int, int] = (1, 0, 0),
        scale: float = 1.0,
        add_solver_func: Callable = add_solver,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        show_object: bool = False,
        show_object_scale: float = 1.0,
        mechanical_binding: MechanicalBinding = MechanicalBinding.SPRING,
        spring_stiffness: Optional[float] = 1e8,
        angular_spring_stiffness: Optional[float] = 1e8,
        articulation_spring_stiffness: Optional[float] = 1e15,
        collision_group: Optional[int] = None,
        collision_contact_stiffness: Union[int, float] = 100,
        cartesian_workspace: Dict = {
            "low": np.array([-np.inf, -np.inf, -np.inf]),
            "high": np.array([np.inf, np.inf, np.inf]),
        },
        ptsd_reset_noise: Optional[np.ndarray] = None,
        rcm_reset_noise: Optional[np.ndarray] = None,
        angle_reset_noise: Optional[float] = None,
        state_limits: Dict = {
            "low": np.array([-90, -90, -90, 0]),
            "high": np.array([90, 90, 90, 100]),
        },
        show_remote_center_of_motion: bool = False,
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
            two_jaws=len(visual_mesh_paths_jaws) > 1,
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
        )

        self.cartesian_workspace = cartesian_workspace
        self.state_limits = state_limits
        if not isinstance(angle_limits, Dict):
            angle_limits = {"low": angle_limits[0], "high": angle_limits[1]}
        self.angle_limits = angle_limits

        self.last_set_state_violated_state_limits = False
        self.last_set_state_violated_workspace_limits = False

        self.ptsd_reset_noise = ptsd_reset_noise
        self.rcm_reset_noise = rcm_reset_noise
        self.angle_reset_noise = angle_reset_noise

        self.show_remote_center_of_motion = show_remote_center_of_motion
        if show_remote_center_of_motion:
            self.rcm_mechanical_object = self.gripper_node.addObject(
                "MechanicalObject",
                template="Rigid3d",
                position=self.pivot_transform((0, 0, 0, 0)),
                showObject=show_object,
                showObjectScale=show_object_scale,
            )

        # Add sphere collision models to the gripper jaws
        self.collision_node_jaw_0 = self.physical_jaw_node.addChild("collision_jaw_0")
        self.collision_node_jaw_1 = self.physical_jaw_node.addChild("collision_jaw_1")

        # Define the z positions of the sphere collision models
        self.num_spheres = len(collision_spheres_config["positions"])

        # Add MechanicalObjects to both jaws
        self.collision_mechanical_object = {
            "jaw_0": self.collision_node_jaw_0.addObject("MechanicalObject", template="Vec3d", position=collision_spheres_config["positions"]),
            "jaw_1": self.collision_node_jaw_1.addObject("MechanicalObject", template="Vec3d", position=collision_spheres_config["positions"]),
        }

        # Add CollisionModel, and RigidMapping to jaw 0
        self.sphere_collisions_jaw_0 = self.collision_node_jaw_0.addObject(
            "SphereCollisionModel",
            radius=[1] * self.num_spheres,
            group=0 if collision_group is None else collision_group,
            contactStiffness=collision_contact_stiffness,
        )
        self.collision_node_jaw_0.addObject(
            "RigidMapping",
            input=self.joint_mechanical_object.getLinkPath(),
            index=1 if self.two_jaws else 0,
        )

        # Add CollisionModel, and RigidMapping to jaw 1
        self.sphere_collisions_jaw_1 = self.collision_node_jaw_1.addObject(
            "SphereCollisionModel",
            radius=[1] * self.num_spheres,
            group=0 if collision_group is None else collision_group,
            contactStiffness=collision_contact_stiffness,
        )
        self.collision_node_jaw_1.addObject(
            "RigidMapping",
            input=self.joint_mechanical_object.getLinkPath(),
            index=2 if self.two_jaws else 1,
        )

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

    def get_collision_object_positions(self) -> np.ndarray:
        """Get the Cartesian positions of the SphereCollisionModels on both jaws."""
        positions_jaw_0 = self.collision_mechanical_object["jaw_0"].position.array()
        positions_jaw_1 = self.collision_mechanical_object["jaw_1"].position.array()

        return np.concatenate([positions_jaw_0, positions_jaw_1])

    def reset_gripper(self) -> None:
        """Reset the grippers state and optionally add noise to remote center of motion and initial state."""

        #############
        # Reset state
        #############

        # Generate a new pivot_transform by adding noise to the initial remote_center_of_motion pose
        if self.rcm_reset_noise is not None:
            new_rcm_pose = self.initial_remote_center_of_motion + self.rng.uniform(-self.rcm_reset_noise, self.rcm_reset_noise)
            self.pivot_transform = generate_ptsd_to_pose(new_rcm_pose)
            self.remote_center_of_motion[:] = new_rcm_pose
            if self.show_remote_center_of_motion:
                with self.rcm_mechanical_object.position.writeable() as rcm_pose:
                    rcm_pose[:] = self.pivot_transform((0, 0, 0, 0))

        # Select a new ptsd state by adding noise to the initial state
        if self.ptsd_reset_noise is not None:
            # Uniformly sample from -noise to +noise and add it to the initial state
            new_state = self.initial_state + self.rng.uniform(-self.ptsd_reset_noise, self.ptsd_reset_noise)

            # Do that until a pose is found that fits in the Cartesian workspace and the state limits
            while np.any((self.state_limits["low"] > new_state) | (new_state > self.state_limits["high"])) or np.any(
                (self.cartesian_workspace["low"] > self.pivot_transform(new_state)[:3]) | (self.pivot_transform(new_state)[:3] > self.cartesian_workspace["high"])
            ):
                new_state = self.initial_state + self.rng.uniform(-self.ptsd_reset_noise, self.ptsd_reset_noise)
        else:
            new_state = self.initial_state

        #############
        # Reset angle
        #############
        # Select a new angle by adding noise to the initial angle
        if self.angle_reset_noise is not None:
            new_angle = np.clip(self.initial_angle + self.rng.uniform(-self.angle_reset_noise, self.angle_reset_noise), self.angle_limits[0], self.angle_limits[1])
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
