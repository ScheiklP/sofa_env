import Sofa.Core

from typing import Optional, Union, Callable, Tuple, Dict
from pathlib import Path
import numpy as np

from sofa_env.sofa_templates.rigid import ControllableRigidObject, MechanicalBinding, PivotizedRigidObject, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.solver import add_solver, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST
from sofa_env.sofa_templates.collision import add_collision_model, COLLISION_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.utils.pivot_transform import generate_oblique_viewing_endoscope_ptsd_to_pose
from sofa_env.utils.math_helper import rotated_z_axis
from sofa_env.utils.camera import determine_look_at

CAMERA_PLUGIN_LIST = (
    [
        "Sofa.Component.Visual",  # <- [InteractiveCamera]
    ]
    + RIGID_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
    + COLLISION_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
)

POSE_KWARGS = {
    "look_at": {  # camera orientation is selected in a way that global y axis points up.
        "position": [0.0, 0.0, 0.0],
        "lookAt": [0.0, 0.0, 0.0],
    },
    "pose": {
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "lookAt": [0.0, 0.0, 0.0],  # yes, this is reduntant information. It is only necessary because of a bug in SOFA.
    },
    "unknown": {
        "distance": 0.0,
    },
}

DEFAULT_LIGHT_SOURCE_KWARGS = {
    "color": [1.0, 1.0, 1.0, 1.0],
    "cutoff": 30.0,
    "attenuation": 0.0,
    "exponent": 1.0,
    "shadowsEnabled": False,
}


class Camera:
    """Interactive camera

    This class adds an ``"InteractiveCamera"`` SOFA object to the scene.

    Notes:
        - The camera should be added directly to the root node as we experienced some issues if the camera was added anywhere else.
        - If ``z_near`` and ``z_far`` are not set, the ``"InteractiveCamera"`` will try to automatically compute the values during simulation.

    Args:
        root_node (Sofa.Core.Node): The SOFA node to which the ``"InteractiveCamera"`` object is added. Should the the root node.
        placement_kwargs (dict): Dictionary to specify the initial placement of the camera. The most common options can be found in ``sofa_env.sofa_templates.camera.POSE_KWARGS``.
        vertical_field_of_view (int): The vertical field of view in degrees of the camera. The horizontal field of view is determined by the aspect ratio through width and height of the viewport.
        z_near (Optional[float]): Minimum distance of objects to the camera. Objects that are closer than this value will not be rendered.
        z_far (Optional[float]): Maximum distance of objects to the camera. Objects further away than this value will not be rendered.
        width_viewport (Optional[int]): Width of the rendered images.
        height_viewport (Optional[int]): Height of the rendered images.
        show_object (bool): Show the Rigid3d frame in SOFA.
        show_object_scale (float): Size of the visualized Rigid3d frame.
        with_light_source (bool): Whether to add a controllable ``SpotLight`` to the camera's tip.
        light_source_kwargs (dict): Dictionary to specify the setup of the light source.
        The most common options can be found in ``sofa_env.sofa_templates.camera.DEFAULT_LIGHT_SOURCE_KWARGS``.

    Examples:
        >>> root_node = Sofa.Core.Node("root")
        >>> camera = Camera(root_node=root_node, width_viewport=128, height_viewport=128, placement_kwargs={"position": [0.0, 1.0, 0.0], "lookAt": [0.0, 0.0, 0.0]})
        >>> pose = camera.get_pose()
        >>> pose[1] = 2.0
        >>> camera.set_pose(pose)
    """

    def __init__(
        self,
        root_node: Sofa.Core.Node,
        placement_kwargs: dict = {
            "position": [0.0, 0.0, 0.0],
            "lookAt": [0.0, 0.0, 0.0],
        },
        vertical_field_of_view: Union[float, int] = 45,
        z_near: Optional[float] = None,
        z_far: Optional[float] = None,
        width_viewport: Optional[int] = None,
        height_viewport: Optional[int] = None,
        show_object: bool = False,
        show_object_scale: float = 1.0,
        with_light_source: bool = False,
        light_source_kwargs: dict = DEFAULT_LIGHT_SOURCE_KWARGS,
    ):

        self.parent = root_node

        compute_z_clip = True if z_near is None and z_far is None else False

        self.sofa_object = root_node.addObject(
            "InteractiveCamera",
            name="camera",
            zNear=z_near,
            zFar=z_far,
            fieldOfView=vertical_field_of_view,
            widthViewport=width_viewport,
            heightViewport=height_viewport,
            computeZClip=compute_z_clip,
            **placement_kwargs,
        )

        # Call init, so that position, orientation, and lookAt arrays are correctly initialized
        self.sofa_object.init()
        self.initial_pose = np.append(self.sofa_object.position.array(), self.sofa_object.orientation.array(), axis=0)
        self.initial_look_at = self.sofa_object.lookAt.array().copy()

        self.show_object = show_object
        if show_object:
            self.frame_node = root_node.addChild("camera_frame")
            self.frame_node.addObject(
                "MechanicalObject",
                template="Rigid3d",
                position=self.initial_pose,
                showObject=True,
                showObjectScale=show_object_scale,
            )

        self.with_light_source = with_light_source
        if with_light_source:
            self.light_node = self.parent.addChild("light")

            if "zNear" not in light_source_kwargs and light_source_kwargs.get("shadowsEnabled", False):
                light_source_kwargs["zNear"] = z_near

            if "zFar" not in light_source_kwargs and light_source_kwargs.get("shadowsEnabled", False):
                light_source_kwargs["zFar"] = z_far

            self.light_node.addObject(
                "SpotLight",
                position=self.initial_pose[:3],
                direction=-rotated_z_axis(self.initial_pose[3:]),
                drawSource=show_object,
                **light_source_kwargs,
            )

    def set_pose(self, pose: np.ndarray) -> None:
        """Sets the camera pose as a numpy array containing position and quaternion."""

        look_at = determine_look_at(camera_position=pose[:3], camera_orientation=pose[3:])

        with self.sofa_object.position.writeable() as camera_position:
            camera_position[:] = pose[:3]
        with self.sofa_object.orientation.writeable() as camera_orientation:
            camera_orientation[:] = pose[3:]
        with self.sofa_object.lookAt.writeable() as camera_look_at:
            camera_look_at[:] = look_at

        if self.with_light_source:
            with self.light_node.SpotLight.position.writeable() as light_position:
                light_position[:] = pose[:3]
            with self.light_node.SpotLight.direction.writeable() as light_direction:
                light_direction[:] = -rotated_z_axis(pose[3:])

        if self.show_object:
            with self.frame_node.MechanicalObject.position.writeable() as frame_pose:
                frame_pose[:] = pose

    def get_pose(self) -> np.ndarray:
        """Returns the camera pose as a numpy array containing position and quaternion."""
        return np.append(self.sofa_object.position.array(), self.sofa_object.orientation.array(), axis=0)

    def set_position(self, position: np.ndarray) -> None:
        """Sets the camera position as a numpy array."""
        with self.sofa_object.position.writeable() as camera_position:
            camera_position[:] = position

    def set_look_at(self, look_at: np.ndarray) -> None:
        """Sets the camera lookAt as a numpy array."""
        with self.sofa_object.lookAt.writeable() as camera_look_at:
            camera_look_at[:] = look_at

    def get_look_at(self) -> np.ndarray:
        """Returns the camera lookAt as a numpy array."""
        return self.sofa_object.lookAt.array()

    def set_orientation(self, orientation: np.ndarray) -> None:
        """Sets the camera orientation as a numpy array quaternion."""
        with self.sofa_object.orientation.writeable() as camera_orientation:
            camera_orientation[:] = orientation[3:]


class PhysicalCamera(Camera, ControllableRigidObject):
    """Physical camera object.

    Extends the camera class with a ControllableRigidObject to include visual and collision models.

    Args:
        root_node (Sofa.Core.Node): The SOFA node to which the ``"InteractiveCamera"`` object is added. Should the the root node.
        placement_kwargs (dict): Dictionary to specify the initial placement of the camera. The most common options can be found in ``sofa_env.sofa_templates.camera.POSE_KWARGS``.
        vertical_field_of_view (int): The vertical field of view in degrees of the camera. The horizontal field of view is determined by the aspect ratio through width and height of the viewport.
        z_near (Optional[float]): Minimum distance of objects to the camera. Objects that are closer than this value will not be rendered.
        z_far (Optional[float]): Maximum distance of objects to the camera. Objects further away than this value will not be rendered.
        width_viewport (Optional[int]): Width of the rendered images.
        height_viewport (Optional[int]): Height of the rendered images.
        show_object (bool): Show the Rigid3d frame in SOFA.
        show_object_scale (float): Size of the visualized Rigid3d frame.
        total_mass (float): Total mass of the deformable object.
        visual_mesh_path (Optional[Union[str, Path]]): Path to the visual surface mesh.
        collision_mesh_path (Optional[Union[str, Path]]): Path to the collision surface mesh.
        scale (Union[float, Tuple[float, float, float]]): Scale factor for loading the meshes.
        add_solver_func (Callable): Function that adds the numeric solvers to the object.
        add_collision_model_func (Callable): Function that defines how the collision surface from ``collision_mesh_path`` is added to the object.
        add_visual_model_func (Callable): Function that defines how the visual surface from ``visual_mesh_path`` is added to the object.
        animation_loop_type (AnimationLoopType): The animation loop of the scene. Required to determine if objects for constraint correction should be added.
        mechanical_binding (MechanicalBinding): Whether to use ``"RestShapeSpringsForceField"`` or ``"AttachConstraint"`` to combine controllable and non-controllable part of the object.
        spring_stiffness (Optional[float]): Spring stiffness of the ``"RestShapeSpringsForceField"``.
        angular_spring_stiffness (Optional[float]): Angular spring stiffness of the ``"RestShapeSpringsForceField"``.
        collision_group (int): The group for which collisions with this object should be ignored. Value has to be set since the jaws and shaft must belong to the same group.
        with_light_source (bool): Whether to add a controllable ``SpotLight`` to the camera's tip.
        light_source_kwargs (dict): Dictionary to specify the setup of the light source.
        The most common options can be found in ``sofa_env.sofa_templates.camera.DEFAULT_LIGHT_SOURCE_KWARGS``.
    """
    def __init__(
        self,
        root_node: Sofa.Core.Node,
        placement_kwargs: dict = {
            "position": [0.0, 0.0, 0.0],
            "lookAt": [0.0, 0.0, 0.0],
        },
        vertical_field_of_view: Union[float, int] = 45,
        z_near: Optional[float] = None,
        z_far: Optional[float] = None,
        width_viewport: Optional[int] = None,
        height_viewport: Optional[int] = None,
        show_object: bool = False,
        show_object_scale: float = 1.0,
        total_mass: Optional[float] = None,
        visual_mesh_path: Optional[Union[str, Path]] = None,
        collision_mesh_path: Optional[Union[str, Path]] = None,
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        add_solver_func: Callable = add_solver,
        add_collision_model_func: Callable = add_collision_model,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        mechanical_binding: MechanicalBinding = MechanicalBinding.ATTACH,
        spring_stiffness: Optional[float] = 2e10,
        angular_spring_stiffness: Optional[float] = 2e10,
        collision_group: Optional[int] = None,
        with_light_source: bool = False,
        light_source_kwargs: dict = DEFAULT_LIGHT_SOURCE_KWARGS,
    ):
        Camera.__init__(
            self,
            root_node=root_node,
            placement_kwargs=placement_kwargs,
            vertical_field_of_view=vertical_field_of_view,
            z_near=z_near,
            z_far=z_far,
            width_viewport=width_viewport,
            height_viewport=height_viewport,
            with_light_source=with_light_source,
            light_source_kwargs=light_source_kwargs,
            show_object=show_object,
        )

        parent_node = root_node.addChild("physical_camera")

        ControllableRigidObject.__init__(
            self,
            parent_node=parent_node,
            name="controllable",
            pose=self.initial_pose,
            total_mass=total_mass,
            visual_mesh_path=visual_mesh_path,
            collision_mesh_path=collision_mesh_path,
            scale=scale,
            add_solver_func=add_solver_func,
            add_collision_model_func=add_collision_model_func,
            add_visual_model_func=add_visual_model_func,
            animation_loop_type=animation_loop_type,
            is_carving_tool=False,
            show_object=show_object,
            show_object_scale=show_object_scale,
            mechanical_binding=mechanical_binding,
            spring_stiffness=spring_stiffness,
            angular_spring_stiffness=angular_spring_stiffness,
            collision_group=collision_group,
        )

    def set_pose(self, pose: np.ndarray) -> None:
        """Sets the camera pose as a numpy array containing position and quaternion."""

        Camera.set_pose(self, pose)

        ControllableRigidObject.set_pose(self, pose)

    def get_camera_pose(self) -> np.ndarray:
        return Camera.get_pose(self)

    def get_physical_body_pose(self) -> np.ndarray:
        return ControllableRigidObject.get_pose(self)


class PivotizedCamera(Camera, PivotizedRigidObject):
    """Pivotized physical camera.

    Extends the camera with a PivotizedRigidObject with visual and collision models.
    The camera position is controlled in pivotized ptsd space to simulate a laparoscopic camera.

    Args:
        root_node (Sofa.Core.Node): The SOFA node to which the ``"InteractiveCamera"`` object is added. Should the the root node.
        vertical_field_of_view (int): The vertical field of view in degrees of the camera. The horizontal field of view is determined by the aspect ratio through width and height of the viewport.
        z_near (Optional[float]): Minimum distance of objects to the camera. Objects that are closer than this value will not be rendered.
        z_far (Optional[float]): Maximum distance of objects to the camera. Objects further away than this value will not be rendered.
        width_viewport (Optional[int]): Width of the rendered images.
        height_viewport (Optional[int]): Height of the rendered images.
        show_object (bool): Show the Rigid3d frame in SOFA.
        show_object_scale (float): Size of the visualized Rigid3d frame.
        total_mass (float): Total mass of the deformable object.
        visual_mesh_path (Optional[Union[str, Path]]): Path to the visual surface mesh.
        collision_mesh_path (Optional[Union[str, Path]]): Path to the collision surface mesh.
        scale (Union[float, Tuple[float, float, float]]): Scale factor for loading the meshes.
        add_solver_func (Callable): Function that adds the numeric solvers to the object.
        add_collision_model_func (Callable): Function that defines how the collision surface from ``collision_mesh_path`` is added to the object.
        add_visual_model_func (Callable): Function that defines how the visual surface from ``visual_mesh_path`` is added to the object.
        animation_loop_type (AnimationLoopType): The animation loop of the scene. Required to determine if objects for constraint correction should be added.
        mechanical_binding (MechanicalBinding): Whether to use ``"RestShapeSpringsForceField"`` or ``"AttachConstraint"`` to combine controllable and non-controllable part of the object.
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
        with_light_source (bool): Whether to add a controllable ``SpotLight`` to the camera's tip.
        light_source_kwargs (dict): Dictionary to specify the setup of the light source.
        The most common options can be found in ``sofa_env.sofa_templates.camera.DEFAULT_LIGHT_SOURCE_KWARGS``.
        oblique_viewing_angle (float): Viewing angle of the camera's oblique viewing optics in degrees.
    """

    def __init__(
        self,
        root_node: Sofa.Core.Node,
        name: str = "pivotized_camera",
        total_mass: Optional[float] = None,
        visual_mesh_path: Optional[Union[str, Path]] = None,
        collision_mesh_path: Optional[Union[str, Path]] = None,
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        add_solver_func: Callable = add_solver,
        add_collision_model_func: Callable = add_collision_model,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
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
            "low": np.array([-90.0, -90.0, np.finfo(np.float16).min, 0.0]),
            "high": np.array([90.0, 90.0, np.finfo(np.float16).max, 200.0]),
        },
        show_remote_center_of_motion: bool = False,
        vertical_field_of_view: Union[float, int] = 45,
        z_near: Optional[float] = None,
        z_far: Optional[float] = None,
        width_viewport: Optional[int] = None,
        height_viewport: Optional[int] = None,
        with_light_source: bool = False,
        light_source_kwargs: dict = DEFAULT_LIGHT_SOURCE_KWARGS,
        oblique_viewing_angle: float = 0.0,
    ):
        parent_node = root_node.addChild("physical_camera")

        PivotizedRigidObject.__init__(
            self,
            parent_node=parent_node,
            name=name,
            total_mass=total_mass,
            visual_mesh_path=visual_mesh_path,
            collision_mesh_path=collision_mesh_path,
            scale=scale,
            add_solver_func=add_solver_func,
            add_collision_model_func=add_collision_model_func,
            add_visual_model_func=add_visual_model_func,
            animation_loop_type=animation_loop_type,
            is_carving_tool=False,
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
            show_remote_center_of_motion=show_remote_center_of_motion,
        )

        self.oblique_viewing_angle = oblique_viewing_angle
        self.oblique_pivot_transform = generate_oblique_viewing_endoscope_ptsd_to_pose(rcm_pose=self.remote_center_of_motion, viewing_angle=oblique_viewing_angle)
        self.initial_oblique_pose = self.oblique_pivot_transform(self.initial_state)

        placement_kwargs = {
            "position": self.initial_oblique_pose[:3],
            "orientation": self.initial_oblique_pose[3:],
            "lookAt": determine_look_at(camera_position=self.initial_oblique_pose[:3], camera_orientation=self.initial_oblique_pose[3:]),
        }

        Camera.__init__(
            self,
            root_node=root_node,
            placement_kwargs=placement_kwargs,
            vertical_field_of_view=vertical_field_of_view,
            z_near=z_near,
            z_far=z_far,
            width_viewport=width_viewport,
            height_viewport=height_viewport,
            with_light_source=with_light_source,
            light_source_kwargs=light_source_kwargs,
            show_object=show_object,
            show_object_scale=show_object_scale,
        )

    def set_pose(self, oblique_pose: np.ndarray, pose: np.ndarray) -> None:
        """Sets the camera pose as a numpy array containing position and quaternion."""
        Camera.set_pose(self, oblique_pose)
        PivotizedRigidObject.set_pose(self, pose)

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
        oblique_pose = self.oblique_pivot_transform(state)

        # Save info about violation of state limits
        self.last_set_state_violated_workspace_limits = np.any(invalid_states_mask)

        # Only set the pose, if all components are within the Cartesian workspace
        if not np.any((self.cartesian_workspace["low"] > pose[:3]) | (pose[:3] > self.cartesian_workspace["high"])):
            self.set_pose(oblique_pose, pose)

            # Only overwrite the internal value of ptsd_state, if that was successful
            self.ptsd_state[:] = state

            # Save info about violation of Cartesian workspace limits
            self.last_set_state_violated_workspace_limits = False
        else:
            self.last_set_state_violated_workspace_limits = True

    def reset_state(self) -> None:
        """Resets the object to its initial ptsd state. Optionally adds noise to rcm pose and ptsd state."""

        # Also calls self.set_state() with the new, sampled state -> writes into self.ptsd_state
        PivotizedRigidObject.reset_state(self)

        # Update the oblique pivot transform function
        self.oblique_pivot_transform = generate_oblique_viewing_endoscope_ptsd_to_pose(rcm_pose=self.remote_center_of_motion, viewing_angle=self.oblique_viewing_angle)

        # Update the poses again with the new oblique_pivot_transform
        self.set_state(self.get_state().copy())

    def seed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Creates a random number generator from a seed."""
        self.rng = np.random.default_rng(seed)
