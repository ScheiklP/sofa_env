import numpy as np
from functools import partial
from pathlib import Path

from typing import Optional, Tuple, Dict, List

import Sofa
import Sofa.Core
import Sofa.SofaDeformable

from sofa_env.sofa_templates.camera import PhysicalCamera, CAMERA_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import VISUAL_STYLES, AnimationLoopType, IntersectionMethod, add_scene_header, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST

from sofa_env.utils.math_helper import distance_between_line_segments
from sofa_env.utils.camera import determine_look_at

from sofa_env.scenes.rope_cutting.sofa_objects.rope import CuttableRope, ROPE_PLUGIN_LIST
from sofa_env.scenes.tissue_dissection.sofa_objects.cauter import PivotizedCauter, CAUTER_PLUGIN_LIST


LENGTH_UNIT = "mm"
TIME_UNIT = "s"
WEIGHT_UNIT = "kg"


HERE = Path(__file__).resolve().parent
INSTRUMENT_MESH_DIR = HERE.parent.parent.parent / "assets/meshes/instruments"
TEXTURE_DIR = HERE.parent.parent.parent / "assets/textures"

PLUGIN_LIST = (
    [
        "Sofa.Component.Engine.Select",  # [BoxROI]
        "Sofa.Component.Topology.Container.Grid",  # [RegularGridTopology]
        "SofaCarving",  # [CarvingManager]
    ]
    + ROPE_PLUGIN_LIST
    + CAMERA_PLUGIN_LIST
    + CAUTER_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
)


def createScene(
    root_node: Sofa.Core.Node,
    debug_rendering: bool = False,
    image_shape: Tuple[Optional[int], Optional[int]] = (None, None),
    animation_loop: AnimationLoopType = AnimationLoopType.FREEMOTION,
    rope_stiffness: float = 1e4,
    rope_mass: float = 1e-2,
    num_ropes: int = 15,
    min_distance_between_ropes: float = 10.0,
    width: float = 100.0,
    height: float = 100.0,
    depth: float = 50.0,
) -> Dict:
    """
    Creates the scene of the RopeCuttingEnv.

    ``num_ropes`` are sampled between two walls that are ``width`` apart, ``height`` high and ``depth``deep.

    Args:
        root_node (Sofa.Core.Node): SET BY ENV. The simulation tree's root node.
        debug_rendering (bool): Whether to render more information for debugging purposes.
        image_shape (Tuple[Optional[int], Optional[int]]): SET BY ENV. Height and Width of the rendered images.
        animation_loop (AnimationLoopType): Animation loop of the simulation.
        rope_stiffness (float): Stiffness of the spring force field that model the ropes.
        rope_mass (float): Total mass of each rope.
        num_ropes (int): Number of ropes that are added to the scene.
        min_distance_between_ropes (float): Minimum distance between the ropes.
        width (float): Distance between the walls.
        height (float): Height of the walls.
        depth (float): Depth of the walls.

    Returns:
        scene_creation_result = {
            "camera": camera.sofa_object,
            "physical_camera": camera,
            "cauter": cauter,
            "ropes": ropes,
            "rope_creation_func": add_ropes_to_scene,
            "rope_creation_kwargs": {
                "num_ropes": num_ropes,
                "left_wall_box": left_wall_box,
                "right_wall_box": right_wall_box,
                "scene_node": scene_node,
                "rope_stiffness": rope_stiffness,
                "rope_mass": rope_mass,
                "min_distance_between_ropes": min_distance_between_ropes,
            },
        }
    """

    gravity = -981  # mm/s^2

    ###################
    # Common components
    ###################
    add_scene_header(
        root_node=root_node,
        plugin_list=PLUGIN_LIST,
        visual_style_flags=VISUAL_STYLES["debug"] if debug_rendering else VISUAL_STYLES["normal"],
        gravity=(0.0, 0.0, gravity),
        animation_loop=animation_loop,
        scene_has_collisions=True,
        collision_detection_method_kwargs={
            "alarmDistance": 3.0,
            "contactDistance": 0.9,
        },
        collision_detection_method=IntersectionMethod.LOCALMIN,
        contact_friction=1e-6,
    )

    cartesian_workspace = {
        "low": np.array([-width / 2, -50, -height / 2 - 10]),
        "high": np.array([width / 2, depth + 10, height / 2 + 10]),
    }

    ###################
    # Camera and lights
    ###################
    root_node.addObject("LightManager")
    root_node.addObject("DirectionalLight", direction=[0, 0, 1], color=[0.3, 0.3, 0.3])

    pose = [50.0, -66.0, 66.0, 0.50320104, 0.12986683, 0.1960476, 0.83155797]
    placement_kwargs = {
        "position": pose[:3],
        "orientation": pose[3:],
    }
    placement_kwargs["lookAt"] = determine_look_at(camera_position=placement_kwargs["position"], camera_orientation=placement_kwargs["orientation"]).tolist()
    camera = PhysicalCamera(
        root_node=root_node,
        placement_kwargs=placement_kwargs,
        with_light_source=True,
        show_object=debug_rendering,
        show_object_scale=10.0,
        width_viewport=image_shape[0],
        height_viewport=image_shape[1],
        z_near=2.0,
        z_far=float(np.linalg.norm(np.array([cartesian_workspace["low"][0], cartesian_workspace["high"][1], cartesian_workspace["low"][2]]) - pose[:3])) * 1.2,
        vertical_field_of_view=52.0,
    )

    #####################################################
    # Dedicated scene node for the rest of the components
    #####################################################
    scene_node = root_node.addChild("scene")

    ########################################
    # Left and right boundaries of the scene
    ########################################
    left_wall_box = {
        "low": np.array([-width / 2, 0.0, -height / 2]),
        "high": np.array([-width / 2, depth, height / 2]),
    }

    right_wall_box = {
        "low": np.array([width / 2, 0.0, -height / 2]),
        "high": np.array([width / 2, depth, height / 2]),
    }

    box_node = scene_node.addChild("box")
    box_node.addObject("MechanicalObject")
    box_node.addObject("BoxROI", box=[left_wall_box["low"], left_wall_box["high"]], name="left_wall_box", drawBoxes=debug_rendering)
    box_node.addObject("BoxROI", box=[right_wall_box["low"], right_wall_box["high"]], name="right_wall_box", drawBoxes=debug_rendering)

    left_wall_node = box_node.addChild("left_wall")
    left_wall_node.addObject("RegularGridTopology", n=[10, 10, 10], min=left_wall_box["low"] - np.array([10, 0, 0]), max=left_wall_box["high"])
    left_wall_node.addObject("OglModel")

    right_wall_node = box_node.addChild("right_wall")
    right_wall_node.addObject("RegularGridTopology", n=[10, 10, 10], min=right_wall_box["low"], max=right_wall_box["high"] + np.array([10, 0, 0]))
    right_wall_node.addObject("OglModel")

    ################
    # Cuttable ropes
    ################
    ropes = add_ropes_to_scene(
        num_ropes=num_ropes,
        left_wall_box=left_wall_box,
        right_wall_box=right_wall_box,
        scene_node=scene_node,
        rope_stiffness=rope_stiffness,
        rope_mass=rope_mass,
        min_distance_between_ropes=min_distance_between_ropes,
    )

    ########
    # Cauter
    ########
    add_visual_model_cauter = partial(add_visual_model, color=(0.0, 0.0, 1.0))
    cauter = PivotizedCauter(
        parent_node=scene_node,
        name="cauter",
        visual_mesh_path=INSTRUMENT_MESH_DIR / "dissection_electrode.stl",
        show_object=debug_rendering,
        show_object_scale=10,
        add_visual_model_func=add_visual_model_cauter,
        ptsd_state=np.array([45.0, 45.0, 0.0, 40.0]),
        rcm_pose=np.array([cartesian_workspace["low"][0], cartesian_workspace["low"][1], cartesian_workspace["high"][2], -180.0, 0.0, 0.0]),
        animation_loop_type=animation_loop,
        spring_stiffness=1e19,
        angular_spring_stiffness=1e19,
        total_mass=1e5,
        cartesian_workspace=cartesian_workspace,
        state_limits={
            "low": np.array([-90, -90, -90, 0]),
            "high": np.array([90, 90, 90, 300]),
        },
    )

    scene_node.addObject(cauter)

    carving_manager = root_node.addObject(
        "CarvingManager",
        carvingDistance=0.8,
        active=False,
    )
    cauter.set_carving_manager(carving_manager)

    ###############
    # Returned data
    ###############
    scene_creation_result = {
        "camera": camera.sofa_object,
        "physical_camera": camera,
        "cauter": cauter,
        "ropes": ropes,
        "rope_creation_func": add_ropes_to_scene,
        "rope_creation_kwargs": {
            "num_ropes": num_ropes,
            "left_wall_box": left_wall_box,
            "right_wall_box": right_wall_box,
            "scene_node": scene_node,
            "rope_stiffness": rope_stiffness,
            "rope_mass": rope_mass,
            "min_distance_between_ropes": min_distance_between_ropes,
        },
    }

    return scene_creation_result


def add_ropes_to_scene(
    num_ropes: int,
    left_wall_box: dict,
    right_wall_box: dict,
    scene_node: Sofa.Core.Node,
    rope_stiffness: float,
    rope_mass: float,
    min_distance_between_ropes: float,
) -> List[CuttableRope]:
    """Sample ``num_ropes`` between the walls with ``min_distance_between_ropes``.
    The ropes are modelled with 20 points, and a PlaneForceField that acts as a floor 10 millimeters below the walls.
    """
    start_positions = []
    end_positions = []

    # Sample start and end positions, until there are enough ropes that do not collide
    num_valid_ropes = 0
    attempts = 0
    while num_valid_ropes < num_ropes:
        start = np.random.uniform(left_wall_box["low"], left_wall_box["high"])
        end = np.random.uniform(right_wall_box["low"], right_wall_box["high"])

        distances_to_other_ropes_too_small = []
        for other_start, other_end in zip(start_positions, end_positions):
            distances_to_other_ropes_too_small.append(distance_between_line_segments(start, end, other_start, other_end, clamp_segments=True)[2] < min_distance_between_ropes)

        if not any(distances_to_other_ropes_too_small):
            start_positions.append(start)
            end_positions.append(end)
            num_valid_ropes += 1
        attempts += 1

        # If there is no valid combination after 1000 attempts of finding start and end positions, try again
        if attempts > 1000:
            start_positions = []
            end_positions = []
            attempts = 0
            num_valid_ropes = 0

    ropes = []
    for rope_number, (start, end) in enumerate(zip(start_positions, end_positions)):
        rope = CuttableRope(
            attached_to=scene_node,
            name=f"rope_{rope_number}",
            start_position=start,
            end_position=end,
            number_of_points=20,
            stiffness=rope_stiffness,
            total_mass=rope_mass,
            plane_height=left_wall_box["low"][2] - 10.0,
        )
        ropes.append(rope)

    return ropes
