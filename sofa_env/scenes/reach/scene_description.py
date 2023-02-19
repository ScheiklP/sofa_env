import Sofa
import Sofa.Core
import numpy as np

from functools import partial
from typing import Optional, Tuple, Union, Dict
from pathlib import Path

from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST
from sofa_env.sofa_templates.rigid import ControllableRigidObject, RigidObject, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, add_scene_header, VISUAL_STYLES, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.camera import CAMERA_PLUGIN_LIST, Camera
from sofa_env.sofa_templates.motion_restriction import MOTION_RESTRICTION_PLUGIN_LIST, add_bounding_box

from sofa_env.utils.camera import determine_look_at

from sofa_env.scenes.reach.sofa_objects.end_effector import EndEffector, END_EFFECTOR_PLUGIN_LIST

PLUGIN_LIST = ["SofaPython3"] + END_EFFECTOR_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST + CAMERA_PLUGIN_LIST + MOTION_RESTRICTION_PLUGIN_LIST + VISUAL_PLUGIN_LIST + RIGID_PLUGIN_LIST

LENGTH_UNIT = "m"
TIME_UNIT = "s"
WEIGHT_UNIT = "kg"

HERE = Path(__file__).resolve().parent
MESH_DIR = HERE.parent.parent.parent / "assets/meshes/models"
ROBOT_DIR = HERE.parent.parent.parent / "assets/meshes/robots"


def createScene(
    root_node: Sofa.Core.Node,
    image_shape: Tuple[Optional[int], Optional[int]] = (None, None),
    debug_rendering: bool = False,
    show_bounding_boxes: bool = True,
    show_remote_center_of_motion: bool = False,
    remote_center_of_motion: Union[np.ndarray, Tuple] = (0.09036183, 0.15260103, 0.01567807),
    randomize_starting_position: bool = True,
    animation_loop: AnimationLoopType = AnimationLoopType.DEFAULT,
    camera_position: Union[np.ndarray, Tuple] = (0.0, 0.07, 0.55),
    camera_orientation: Union[np.ndarray, Tuple] = (0.0, 0.0, 0.0, 1.0),
    camera_field_of_view_vertical: int = 42,
    camera_placement_kwargs: Optional[dict] = None,
    sphere_radius: float = 0.008,
) -> Dict:
    """
    Creates the scene of the ReachEnv.

    Args:
        root_node (Sofa.Core.Node): SET BY ENV. The simulation tree's root node.
        image_shape (Tuple[Optional[int], Optional[int]]): SET BY ENV. Height and Width of the rendered images.
        debug_rendering (bool): Whether to render more information for debugging purposes.
        show_bounding_boxes (bool): Whether to render the workspace bounding box.
        show_remote_center_of_motion (bool): Whether to render the robot's remote center of motion.
        remote_center_of_motion (Union[np.ndarray, Tuple]): Cartesian position of the remote center of motion.
        randomize_starting_position (bool): Whether to randomly select a new position of the end effector on reset.
        animation_loop (AnimationLoopType): Animation loop of the simulation.
        camera_position (Union[np.ndarray, Tuple]): Cartesian position of the camera.
        camera_orientation (Union[np.ndarray, Tuple]): Quaternion describing the orientation of the camera.
        camera_field_of_view_vertical (int): Vertical field of view of the camera.
        camera_placement_kwargs (Optional[dict]): Additional kwargs to overwrite position and orientation of the camera.
        sphere_radius (float): Radius of the target sphere in meters.

    Returns:
        scene_creation_result = {
            "camera": camera,
            "interactive_objects": {
                "end_effector": end_effector,
                "visual_target": visual_target,
            },
            "workspace": workspace,
        }
    """

    workspace = {
        "low": np.array([-0.15, -0.05, -0.15] + [0.0] * 3 + [1]),
        "high": np.array([0.15, 0.14, 0.15] + [0.0] * 3 + [1]),
    }

    starting_box = {
        "low": np.array([-0.15, -0.05, -0.15] + [0.0] * 3 + [1]),
        "high": np.array([0.15, 0.14, 0.15] + [0.0] * 3 + [1]),
    }

    end_effector_starting_pose = np.array((0.06, 0.06, 0.06) + (0.0, 0.0, 0.0, 1.0))
    visual_target_starting_pose = np.array((0.06, 0.06, 0.06) + (0.0, 0.0, 0.0, 1.0))
    visual_target_color = (0, 0, 255)

    ########
    # Meshes
    ########
    gripper_mesh_path = ROBOT_DIR / "dvrk_gripper.stl"
    psm_main_link_surface_mesh_path = ROBOT_DIR / "psm_main_link.stl"
    sphere_surface_mesh_path = MESH_DIR / "unit_sphere.stl"

    ###################
    # Common components
    ###################
    add_scene_header(
        root_node=root_node,
        plugin_list=PLUGIN_LIST,
        visual_style_flags=VISUAL_STYLES["debug"] if debug_rendering else VISUAL_STYLES["normal"],
        animation_loop=animation_loop,
        scene_has_collisions=False,
    )

    ###################
    # Camera and lights
    ###################
    root_node.addObject(
        "LightManager",
        listening=True,
        ambient=(
            0.4,
            0.4,
            0.4,
            0.4,
        ),
    )

    root_node.addObject("PositionalLight", position=camera_position)

    # Setting only orientation or lookAt gives absolutely wrong values for camera pose.
    # See https://github.com/sofa-framework/sofa/issues/2727
    # Hence, we additionally calculate a lookAt from the orientation.
    look_at = determine_look_at(np.array(camera_position), np.array(camera_orientation))

    # Approximation of nearest and furthest points from the camera
    # Roughly the distance from camera to the nearest corner of the workspace
    z_near = float(np.linalg.norm(camera_position[1:]) - np.linalg.norm(workspace["high"][1:3]))
    # Roughly the distance from camera to the furthest corner of the workspace plus 10%
    z_far = float(np.linalg.norm(camera_position[1:]) + np.linalg.norm(workspace["low"][1:3])) * 1.1

    camera = Camera(
        root_node=root_node,
        placement_kwargs={
            "position": camera_position,
            "orientation": camera_orientation,
            "lookAt": look_at,
        }
        if camera_placement_kwargs is None
        else camera_placement_kwargs,
        z_near=z_near,
        z_far=z_far,
        width_viewport=image_shape[0],
        height_viewport=image_shape[1],
        vertical_field_of_view=camera_field_of_view_vertical,
    )

    #####################################################
    # Dedicated scene node for the rest of the components
    #####################################################
    scene_node = root_node.addChild("scene")

    ##############
    # End effector
    ##############
    # all Sofa.Core.Controller objects have to be added to the scene graph to correctly receive events
    end_effector = scene_node.addObject(
        EndEffector(
            parent_node=scene_node,
            name="end_effector",
            pose=end_effector_starting_pose,
            randomize_starting_position=randomize_starting_position,
            starting_box=starting_box,
            visual_mesh_path_gripper=gripper_mesh_path,
            visual_mesh_path_main_link=psm_main_link_surface_mesh_path,
            remote_center_of_motion=remote_center_of_motion,
            animation_loop_type=animation_loop,
            workspace=workspace,
            show_object=False,
            show_object_scale=0.01,
            add_visual_marker=True,
            visual_marker_mesh_path=sphere_surface_mesh_path,
            visual_marker_scale=sphere_radius,
        )
    )

    ###############
    # Visual Target
    ###############
    visual_target_visual_func = partial(add_visual_model, color=visual_target_color)
    visual_target = ControllableRigidObject(
        parent_node=scene_node,
        name="visual_target",
        pose=visual_target_starting_pose,
        visual_mesh_path=sphere_surface_mesh_path,
        scale=sphere_radius,
        add_visual_model_func=visual_target_visual_func,
        show_object=False,
        show_object_scale=0.01,
    )

    if show_remote_center_of_motion:
        RigidObject(
            parent_node=scene_node,
            name="remote_center_of_motion",
            pose=remote_center_of_motion + (0, 0, 0, 1),
            visual_mesh_path=sphere_surface_mesh_path,
            scale=1.0,
            add_visual_model_func=visual_target_visual_func,
            show_object=True,
            show_object_scale=0.01,
        )

    ########################
    # Showing bounding boxes
    ########################
    if show_bounding_boxes:
        scene_node.addObject("MechanicalObject", template="Rigid3d", position=[0, 0, 0, 0, 0, 0, 1])
        add_bounding_box(scene_node, min=workspace["low"][:3], max=workspace["high"][:3], show_bounding_box=True, name="workspace")
        add_bounding_box(scene_node, min=starting_box["low"][:3], max=starting_box["high"][:3], show_bounding_box=True, name="starting_space")

    ###############
    # Returned data
    ###############
    scene_creation_result = {
        "camera": camera,
        "interactive_objects": {
            "end_effector": end_effector,
            "visual_target": visual_target,
        },
        "workspace": workspace,
    }

    return scene_creation_result
