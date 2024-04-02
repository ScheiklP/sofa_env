import numpy as np

import Sofa
import Sofa.Core

from pathlib import Path
from typing import Optional, Tuple, Dict
from functools import partial

from sofa_env.sofa_templates.motion_restriction import add_fixed_constraint_in_bounding_box, MOTION_RESTRICTION_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, ConstraintSolverType, IntersectionMethod, add_scene_header, VISUAL_STYLES, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.camera import Camera, CAMERA_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST
from sofa_env.utils.camera import determine_look_at

from sofa_env.scenes.bimanual_tissue_manipulation.sofa_objects.gripper import PivotizedGripper, Side, GRIPPER_PLUGIN_LIST
from sofa_env.scenes.bimanual_tissue_manipulation.sofa_objects.tissue import Tissue, TISSUE_PLUGIN_LIST


PLUGIN_LIST = ["SofaPython3"] + SCENE_HEADER_PLUGIN_LIST + CAMERA_PLUGIN_LIST + TISSUE_PLUGIN_LIST + GRIPPER_PLUGIN_LIST + VISUAL_PLUGIN_LIST + MOTION_RESTRICTION_PLUGIN_LIST

LENGTH_UNIT = "mm"
TIME_UNIT = "s"
WEIGHT_UNIT = "kg"

HERE = Path(__file__).resolve().parent
INSTRUMENT_MESH_DIR = HERE.parent.parent.parent / "assets/meshes/instruments"
TEXTURE_FILE_PATH = HERE / "textures" / "canvas.png"


def createScene(
    root_node: Sofa.Core.Node,
    debug_rendering: bool = False,
    animation_loop: AnimationLoopType = AnimationLoopType.DEFAULT,
    image_shape: Tuple[Optional[int], Optional[int]] = (400, 400),
    zero_gravity: bool = False,
    check_collision: bool = True,
    tissue_size: Tuple[float, float] = (80.0, 60.0),
    tissue_mass: float = 0.05,
    tissue_grid_resolution: int = 40,
    tissue_visual_resolution: int = 100,
) -> Dict:
    """Creates the scene of the BimanualTissueManipulationEnv.

    Args:
        root_node (Sofa.Core.Node): SET BY ENV. The simulation tree's root node.
        debug_rendering (bool): Whether to render more information for debugging purposes.
        animation_loop (AnimationLoopType): Animation loop of the simulation.
        image_shape (Tuple[Optional[int], Optional[int]]): SET BY ENV. Height and Width of the rendered images.
        zero_gravity (bool): Whether to simulate without gravity.
        check_collision (bool): Whether to create collision listener and check for collisions.
        tissue_size (Tuple[float, float]): Size of the tissue. Defaults to (80.0, 100.0).
        tissue_mass (float): Mass of the tissue. Defaults to 0.03.
        tissue_grid_resolution (int): Resolution of the tissue. Defaults to 20.
        tissue_visual_resolution (int): Visual resolution of the tissue . Defaults to 30.

    Returns:
        scene_creation_result = {
            "camera": camera,
            "tissue": tissue,
            "left_gripper": left_gripper,
            "right_gripper": right_gripper,
            "left_contact_listener": left_contact_listener,
            "right_contact_listener": right_contact_listener,
        }
    """

    gravity = (0.0, 0.0, 0.0) if zero_gravity else (0.0, -9.81, 0.0)
    contact_distance = 0.2
    alarm_distance = contact_distance / 5.0

    left_color = (0.0, 1.0, 0.0)
    right_color = (0.0, 0.0, 1.0)

    ###################
    # Common Components
    ###################
    add_scene_header(
        root_node=root_node,
        plugin_list=PLUGIN_LIST,
        visual_style_flags=VISUAL_STYLES["debug"] if debug_rendering else VISUAL_STYLES["normal"],
        collision_detection_method=IntersectionMethod.LOCALMIN,
        gravity=gravity,
        scene_has_collisions=check_collision,
        collision_detection_method_kwargs={
            "alarmDistance": alarm_distance,
            "contactDistance": contact_distance,
        },
        constraint_solver=ConstraintSolverType.LCP,
        constraint_solver_kwargs={
            "maxIt": 1000,
            "tolerance": 0.001,
        },
        animation_loop=animation_loop,
    )

    ###################
    # Camera and Lights
    ###################
    camera_pose = np.array([0.0, 50.0, 180.0, 0.0, 0.0, 0.0, 1.0])

    camera_config = {
        "placement_kwargs": {
            "position": camera_pose[:3],
            "orientation": camera_pose[3:],
        },
        "vertical_field_of_view": 50.0,
    }

    camera_config["placement_kwargs"]["lookAt"] = determine_look_at(
        camera_config["placement_kwargs"]["position"],
        camera_config["placement_kwargs"]["orientation"],
    )

    camera = Camera(
        root_node=root_node,
        placement_kwargs=camera_config["placement_kwargs"],
        z_near=0.01,
        z_far=450,
        width_viewport=image_shape[0],
        height_viewport=image_shape[1],
        with_light_source=True,
        show_object=debug_rendering,
        show_object_scale=5.0,
        vertical_field_of_view=camera_config["vertical_field_of_view"],
    )

    root_node.addObject(
        "LightManager",
        listening=True,
        ambient=(0.3, 0.3, 0.3, 0.1),
        shadows=True,
        softShadows="0",
    )

    root_node.addObject("DirectionalLight", direction=camera_config["placement_kwargs"]["lookAt"])

    #####################################################
    # Dedicated SCENE NODE for the rest of the components
    #####################################################
    scene_node = root_node.addChild("scene")

    ########
    # Tissue
    ########
    tissue = Tissue(
        parent_node=scene_node,
        name="tissue",
        show_debug_objects=debug_rendering,
        size=tissue_size,
        texture_file_path=TEXTURE_FILE_PATH,
        grid_resolution=tissue_grid_resolution,
        visual_resolution=tissue_visual_resolution,
        collision_group=1,
        total_mass=tissue_mass,
    )

    # Fix bottom edge
    add_fixed_constraint_in_bounding_box(
        attached_to=scene_node.rigidified_tissue.deformable,
        min=(-0.5 * tissue_size[0], -0.5 * tissue_size[1] - 0.1, -0.1),
        max=(0.5 * tissue_size[0], -0.5 * tissue_size[1] + 0.3, 0.1),
        show_bounding_box=debug_rendering,
        bounding_box_name="bb_bottom_edge",
    )

    #########
    # Gripper
    #########
    gripper_surface_mesh_path = INSTRUMENT_MESH_DIR / "single_action_forceps_shaft.stl"

    angle = np.rad2deg(np.arctan2(tissue_size[0], tissue_size[1]))
    # factor to place the rcm outside the tissue
    factor = 1.8
    left_rcm = np.array([-factor * tissue_size[0], factor * tissue_size[1], 0.0, -90.0, -angle, 0.0])
    right_rcm = np.array([factor * tissue_size[0], factor * tissue_size[1], 0.0, -90.0, angle, 0.0])
    halve_cloth_diagonal = np.sqrt((0.5 * tissue_size[0]) ** 2 + (0.5 * tissue_size[1]) ** 2)
    rcm_diagonal = np.sqrt((factor * tissue_size[0]) ** 2 + (factor * tissue_size[1]) ** 2)
    initial_depth = rcm_diagonal - halve_cloth_diagonal
    right_ptsd = np.array([0.0, 0.0, 0.0, -initial_depth])
    left_ptsd = np.array([0.0, 0.0, 0.0, -initial_depth])
    state_limits = {
        "low": np.array([-45.0, 0.0, 0.0, -initial_depth * 1.2]),
        "high": np.array([45.0, 0.0, 0.0, 0.0]),
    }

    right_gripper = PivotizedGripper(
        parent_node=scene_node,
        side=Side.RIGHT,
        name="right_gripper",
        visual_mesh_path=gripper_surface_mesh_path,
        visual_mesh_path_jaw=INSTRUMENT_MESH_DIR / "single_action_forceps_jaw.stl",
        rcm_pose=right_rcm,
        state_limits=state_limits,
        ptsd_state=right_ptsd,
        show_object=debug_rendering,
        add_visual_model_func=partial(
            add_visual_model,
            color=right_color,
            rotation=(180.0, 0.0, 0.0),
            translation=(0.0, 0.0, 17.0),
        ),
        collision_group=2,
        show_object_scale=10.0,
        scale=1.0,
    )

    left_gripper = PivotizedGripper(
        parent_node=scene_node,
        side=Side.LEFT,
        name="left_gripper",
        visual_mesh_path=gripper_surface_mesh_path,
        visual_mesh_path_jaw=INSTRUMENT_MESH_DIR / "single_action_forceps_jaw.stl",
        state_limits=state_limits,
        rcm_pose=left_rcm,
        ptsd_state=left_ptsd,
        show_object=debug_rendering,
        add_visual_model_func=partial(
            add_visual_model,
            color=left_color,
            rotation=(180.0, 0.0, 0.0),
            translation=(0.0, 0.0, 17.0),
        ),
        scale=1.0,
        collision_group=3,
        show_object_scale=10.0,
    )

    scene_node.addObject(right_gripper)
    scene_node.addObject(left_gripper)

    # Align rigid reference frame orientation with gripper orientation
    right_gripper_orientation = right_gripper.get_pose()[3:]
    left_gripper_orientation = left_gripper.get_pose()[3:]
    rigidified_tissue = tissue.rigidified_tissue_node
    with rigidified_tissue.rigid.MechanicalObject.position.writeable() as positions:
        positions[Side.LEFT.value][3:] = left_gripper_orientation
        positions[Side.RIGHT.value][3:] = right_gripper_orientation

    left_gripper.attach_to_tissue(rigidified_tissue)
    right_gripper.attach_to_tissue(rigidified_tissue)

    if check_collision:
        left_contact_listener = scene_node.addObject(
            "ContactListener",
            name="left_contact_listener",
            collisionModel1=left_gripper.shaft_collision_model_node.SphereCollisionModel.getLinkPath(),
            collisionModel2=tissue.collision_model.getLinkPath(),
        )
        right_contact_listener = scene_node.addObject(
            "ContactListener",
            name="right_contact_listener",
            collisionModel1=right_gripper.shaft_collision_model_node.SphereCollisionModel.getLinkPath(),
            collisionModel2=tissue.collision_model.getLinkPath(),
        )
    else:
        left_contact_listener = None
        right_contact_listener = None

    ###############
    # Returned Data
    ###############
    scene_creation_result = {
        "camera": camera,
        "tissue": tissue,
        "left_gripper": left_gripper,
        "right_gripper": right_gripper,
        "left_contact_listener": left_contact_listener,
        "right_contact_listener": right_contact_listener,
    }

    return scene_creation_result
