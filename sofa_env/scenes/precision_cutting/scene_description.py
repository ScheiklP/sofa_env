from typing import Optional, Tuple, Callable, Union, Dict
from functools import partial
from pathlib import Path

import numpy as np

import Sofa
import Sofa.Core

from sofa_env.sofa_templates.solver import add_solver, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.camera import Camera, CAMERA_PLUGIN_LIST
from sofa_env.sofa_templates.rigid import MechanicalBinding
from sofa_env.sofa_templates.scene_header import IntersectionMethod, AnimationLoopType, add_scene_header, VISUAL_STYLES, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST
from sofa_env.sofa_templates.mappings import MappingType, MAPPING_PLUGIN_LIST

from sofa_env.utils.camera import determine_look_at

from sofa_env.scenes.precision_cutting.sofa_objects.cloth import Cloth, CLOTH_PLUGIN_LIST
from sofa_env.scenes.precision_cutting.sofa_objects.grid_path_projection import GridPathProjection
from sofa_env.scenes.precision_cutting.sofa_objects.cloth_cut import sine_cut
from sofa_env.scenes.precision_cutting.sofa_objects.scissors import ArticulatedScissors, SCISSORS_PLUGIN_LIST
from sofa_env.scenes.precision_cutting.sofa_objects.gripper import ArticulatedGripper, GRIPPER_PLUGIN_LIST

PLUGIN_LIST = (
    [
        "SofaPython3",
    ]
    + SCENE_HEADER_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
    + CLOTH_PLUGIN_LIST
    + CAMERA_PLUGIN_LIST
    + SCISSORS_PLUGIN_LIST
    + GRIPPER_PLUGIN_LIST
    + MAPPING_PLUGIN_LIST
)


LENGTH_UNIT = "mm"
TIME_UNIT = "s"
WEIGHT_UNIT = "kg"

HERE = Path(__file__).resolve().parent
INSTRUMENT_MESH_DIR = HERE.parent.parent.parent / "assets/meshes/instruments"
TEXTURE_DIR = HERE.parent.parent.parent / "assets/textures"
MESH_DIR = HERE.parent.parent.parent / "assets/meshes/models"


def createScene(
    root_node: Sofa.Core.Node,
    image_shape: Tuple[Optional[int], Optional[int]] = (1024, 1024),
    debug_rendering: bool = False,
    animation_loop: AnimationLoopType = AnimationLoopType.DEFAULT,
    cloth_size: Tuple[float, float] = (75.0, 75.0),
    cloth_resolution: int = 40,
    cloth_cut_stroke_radius: float = 1.5,
    cloth_cutting_path_func: Callable[[], GridPathProjection] = partial(sine_cut, amplitude=30.0, frequency=1.5 / 75.0, position=0.66),
    scissors_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = np.array([30.0, 30.0, 30.0, 10.0]),
    show_closest_point_on_path: bool = False,
    with_gripper: bool = False,
) -> Dict:
    """
    Creates the scene of the PrecisionCuttingEnv.

    Args:
        root_node (Sofa.Core.Node): SET BY ENV. The simulation tree's root node.
        image_shape (Tuple[Optional[int], Optional[int]]): SET BY ENV. Height and Width of the rendered images.
        debug_rendering (bool): Whether to render more information for debugging purposes.
        animation_loop (AnimationLoopType): Animation loop of the simulation.
        cloth_size (Tuple[float, float]): Size of the cloth in mm.
        cloth_resolution (int): Resolution of the cloth in points.
        cloth_cut_stroke_radius (float): Radius of the cloth cut stroke in mm.
        cloth_cutting_path_func (Callable[[], GridPathProjection]): Function that returns a GridPathProjection object.
        scissors_reset_noise (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Noise added to the scissors' initial state on reset.
        show_closest_point_on_path (bool): Whether to show the closest point on the cutting path to the scissors.
        with_gripper (bool): SET BY ENV. Whether to add a gripper to the scene.

    Returns:
        scene_creation_result = {
            "camera": camera,
            "cloth": cloth,
            "scissors": scissors,
            "closest_path_point_mechanical_object": closest_path_point_mechanical_object,
        }
    """

    ###################
    # Common components
    ###################
    gravity = -981.0  # mm/s^2
    contact_distance = 0.2
    alarm_distance = contact_distance / 5.0

    add_scene_header(
        root_node=root_node,
        plugin_list=PLUGIN_LIST,
        visual_style_flags=VISUAL_STYLES["normal"],
        gravity=(0.0, 0.0, gravity),
        animation_loop=animation_loop,
        scene_has_collisions=True,
        collision_detection_method_kwargs={
            "alarmDistance": alarm_distance,
            "contactDistance": contact_distance,
        },
        collision_detection_method=IntersectionMethod.LOCALMIN,
    )

    ###################
    # Camera and lights
    ###################
    root_node.addObject(
        "LightManager",
        listening=True,
        ambient=(0.4, 0.4, 0.4, 0.4),
    )

    camera_pose = np.array([-63.24066124, -97.25488281, 89.39038112, 0.38739854, -0.12891536, -0.26941274, 0.87219262])
    camera_config = {
        "placement_kwargs": {
            "position": camera_pose[:3],
            "orientation": camera_pose[3:],
        },
        "vertical_field_of_view": 45.0,
    }

    camera_config["placement_kwargs"]["lookAt"] = determine_look_at(
        camera_config["placement_kwargs"]["position"],
        camera_config["placement_kwargs"]["orientation"],
    )

    camera = Camera(
        root_node=root_node,
        placement_kwargs=camera_config["placement_kwargs"],
        z_near=50.0,
        z_far=200.0,
        width_viewport=image_shape[0],
        height_viewport=image_shape[1],
        show_object=False,
        with_light_source=True,
        show_object_scale=1.0,
        vertical_field_of_view=camera_config["vertical_field_of_view"],
    )

    #####################################################
    # Dedicated scene node for the rest of the components
    #####################################################
    scene_node = root_node.addChild("scene")

    cloth_solvers = partial(
        add_solver,
        ode_solver_rayleigh_stiffness=0.1,
        ode_solver_rayleigh_mass=0.1,
    )

    cloth = Cloth(
        parent_node=scene_node,
        name="cloth",
        size=cloth_size,
        grid_resolution=cloth_resolution,
        visual_resolution=cloth_resolution,
        position=(0, 0, 0),
        orientation=(0, 0, 0),
        animation_loop_type=animation_loop,
        add_solver_func=cloth_solvers,
        texture_file_path=HERE / "meshes/canvas_texture_with_red.png",
        cutting_path=cloth_cutting_path_func(),
        cut_stroke_radius=cloth_cut_stroke_radius,
        show_debug_objects=debug_rendering,
    )
    scene_node.addObject(cloth)

    scissors_rcm = np.array([0.0, -60.0, 10.0, 180.0, 0.0, 0.0])
    cartesian_workspace = {
        "low": np.array(
            [
                scissors_rcm[0] - cloth_size[0],
                scissors_rcm[1] - 0.5 * cloth_size[1],
                scissors_rcm[2] - 50.0,
            ]
        ),
        "high": np.array(
            [
                scissors_rcm[0] + cloth_size[0],
                scissors_rcm[1] + 1.5 * cloth_size[1],
                scissors_rcm[2] + 50.0,
            ]
        ),
    }
    scissors = ArticulatedScissors(
        parent_node=scene_node,
        name="scissors",
        rcm_pose=scissors_rcm,
        ptsd_state=np.array([0.0, 80.0, 0.0, 2.0]),
        angle=30,
        visual_mesh_path_shaft=INSTRUMENT_MESH_DIR / "instrument_shaft.stl",
        visual_mesh_paths_jaws=[INSTRUMENT_MESH_DIR / "scissors_jaw_left.stl", INSTRUMENT_MESH_DIR / "scissors_jaw_right.stl"],
        add_visual_model_func=partial(add_visual_model, color=(0.0, 1.0, 0.0)),
        show_object=debug_rendering,
        show_remote_center_of_motion=debug_rendering,
        show_object_scale=10.0,
        mechanical_binding=MechanicalBinding.SPRING,
        cartesian_workspace=cartesian_workspace,
        state_limits={
            "low": np.array([-90.0, 60.0, -90.0, -10.0]),
            "high": np.array([90.0, 120.0, 90.0, 250.0]),
        },
        ptsd_reset_noise=scissors_reset_noise,
        animation_loop_type=animation_loop,
    )

    scene_node.addObject(scissors)

    if with_gripper:
        gripper = ArticulatedGripper(
            parent_node=scene_node,
            name="gripper",
            rcm_pose=np.array([-60.0, -60.0, 40.0, 180.0, 0.0, 0.0]),
            ptsd_state=np.array([30.0, 45.0, 0.0, 30.0]),
            visual_mesh_path_shaft=INSTRUMENT_MESH_DIR / "instrument_shaft.stl",
            visual_mesh_paths_jaws=[INSTRUMENT_MESH_DIR / "maryland_dissector_jaw_left.stl", INSTRUMENT_MESH_DIR / "maryland_dissector_jaw_right.stl"],
            angle=30,
            cloth_to_grasp=cloth,
            state_limits={
                "low": np.array([-90.0, 60.0, -90.0, -10.0]),
                "high": np.array([90.0, 120.0, 90.0, 250.0]),
            },
            add_visual_model_func=partial(add_visual_model, color=(0.0, 0.0, 1.0)),
        )
    else:
        gripper = None

    root_node.addObject(
        "CarvingManager",
        carvingDistance=0.2,
        active=True,
    )

    closest_path_point_node = scene_node.addChild("closest_point_on_path")
    closest_path_point_mechanical_object = closest_path_point_node.addObject(
        "MechanicalObject",
        template="Rigid3d",
        position=np.identity(7)[-1],
    )

    if show_closest_point_on_path:
        add_visual_model(
            attached_to=closest_path_point_node,
            surface_mesh_file_path=MESH_DIR / "unit_sphere.stl",
            scale=2.0,
            color=(1.0, 1.0, 0.0),
            mapping_type=MappingType.RIGID,
        )

    ###############
    # Returned data
    ###############
    scene_creation_result = {
        "camera": camera,
        "cloth": cloth,
        "scissors": scissors,
        "closest_path_point_mechanical_object": closest_path_point_mechanical_object,
        "gripper": gripper,
    }

    return scene_creation_result
