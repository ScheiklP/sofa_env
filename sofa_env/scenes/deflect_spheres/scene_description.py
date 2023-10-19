from functools import partial
import numpy as np
from pathlib import Path

from typing import Optional, Tuple, Dict, List, Union

import Sofa
import Sofa.Core

from sofa_env.scenes.deflect_spheres.sofa_objects.post import Post, State, POST_PLUGIN_LIST
from sofa_env.scenes.tissue_dissection.sofa_objects.cauter import PivotizedCauter, CAUTER_PLUGIN_LIST

from sofa_env.sofa_templates.camera import PhysicalCamera, CAMERA_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import VISUAL_STYLES, AnimationLoopType, IntersectionMethod, add_scene_header, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST

from sofa_env.utils.math_helper import euler_to_rotation_matrix, rotation_matrix_to_quaternion
from sofa_env.utils.camera import determine_look_at


LENGTH_UNIT = "mm"
TIME_UNIT = "s"
WEIGHT_UNIT = "kg"


HERE = Path(__file__).resolve().parent
INSTRUMENT_MESH_DIR = HERE.parent.parent.parent / "assets/meshes/instruments"
TEXTURE_DIR = HERE.parent.parent.parent / "assets/textures"

PLUGIN_LIST = (
    [
        "Sofa.Component.Topology.Container.Grid",  # [RegularGridTopology]
    ]
    + CAMERA_PLUGIN_LIST
    + CAUTER_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
    + POST_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
)


def createScene(
    root_node: Sofa.Core.Node,
    debug_rendering: bool = False,
    image_shape: Tuple[Optional[int], Optional[int]] = (None, None),
    animation_loop: AnimationLoopType = AnimationLoopType.FREEMOTION,
    width: float = 100.0,
    height: float = 80.0,
    depth: float = 100.0,
    num_objects: int = 5,
    post_height_limits: Tuple[float, float] = (30.0, 60.0),
    min_post_distance: float = 20.0,
    stiffness_limits: Tuple[float, float] = (1e3, 1e3),
    single_agent: bool = True,
    cauter_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = np.array([30.0, 30.0, 30.0, 10.0]),
    no_textures: bool = False,
) -> Dict:
    """
    Creates the scene of the DeflectSpheresEnv.

    Args:
        root_node (Sofa.Core.Node): SET BY ENV. The simulation tree's root node.
        debug_rendering (bool): Whether to render more information for debugging purposes.
        image_shape (Tuple[Optional[int], Optional[int]]): SET BY ENV. Height and Width of the rendered images.
        animation_loop (AnimationLoopType): Animation loop of the simulation.
        width (float): Width of the board on which spheres are placed.
        height (float): Height of the box over the board on which spheres are placed.
        depth (float): Width of the board on which spheres are placed.
        num_objects (int): SET BY ENV. How many spheres to place in the scene.
        post_height_limits (Tuple[float, float]): Min and max z value in which spheres are placed over the board.
        min_post_distance (float): Minimum distance between spheres when sampling new positions.
        stiffness_limits (Tuple[float, float]): Min and max value of the spring stiffness that holds the spheres in place.
        single_agent (bool): SET BY ENV. Whether to create the scene with one, or two cauter instruments.
        cauter_reset_noise (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Limits to uniformly sample noise that is added to the cauter's initial state at reset.
        no_textures (bool): Whether to not load texture files. May be needed to run scene without GPU support.

    Returns:
        scene_creation_result = {
            "camera": camera.sofa_object,
            "physical_camera": camera,
            "right_cauter": right_cauter,
            "left_cauter": left_cauter,
            "posts": posts,
            "sample_positions_func": sample_positions,
            "contact_listener": contact_listener,
        }
    """

    gravity = -981  # mm/s^2

    if post_height_limits[0] < 20.0:
        raise ValueError(f"post_height_limits[0] should be >= 20.0. Otherwise, the sphere will be too close to the floor. Received {post_height_limits[0]}")

    if post_height_limits[1] > height - 20.0:
        raise ValueError(f"post_height_limits[1] should be <= height - 20.0. Otherwise, the sphere will be too close to the ceiling. Received {post_height_limits[1]}")

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
        contact_friction=1.0,
    )

    cartesian_workspace = {
        "low": np.array([-width / 2 - 10, -depth / 2 - 10, -10.0]),
        "high": np.array([width / 2 + 10, depth / 2 + 10, height + 10]),
    }

    ###################
    # Camera and lights
    ###################
    root_node.addObject("LightManager")
    root_node.addObject("DirectionalLight", direction=[-1, -1, 1], color=[1.3] * 3)

    pose = np.array([0.0, -depth - 40, height + 40, 0.0, 0.0, 0.0, 1.0])
    pose[3:] = rotation_matrix_to_quaternion(euler_to_rotation_matrix(np.array([58.0, 0.0, 0.0])))
    placement_kwargs = {
        "position": pose[:3],
        "orientation": pose[3:],
    }
    placement_kwargs["lookAt"] = determine_look_at(camera_position=placement_kwargs["position"], camera_orientation=placement_kwargs["orientation"])
    camera = PhysicalCamera(
        root_node=root_node,
        placement_kwargs=placement_kwargs,
        with_light_source=debug_rendering,
        show_object=debug_rendering,
        show_object_scale=10.0,
        width_viewport=image_shape[0],
        height_viewport=image_shape[1],
        z_near=5.0,
        z_far=float(np.linalg.norm(np.array([cartesian_workspace["low"][0], cartesian_workspace["high"][1], cartesian_workspace["low"][2]]) - pose[:3])) * 1.2,
        vertical_field_of_view=45.0,
    )

    #####################################################
    # Dedicated scene node for the rest of the components
    #####################################################
    scene_node = root_node.addChild("scene")

    board_node = scene_node.addChild("board")
    grid_shape = (10, 10, 2)
    board_node.addObject("RegularGridTopology", min=[-width / 2, -depth / 2, -5], max=[width / 2, depth / 2, 0], n=grid_shape)
    if no_textures:
        board_node.addObject("OglModel", color=[value / 255 for value in [246.0, 205.0, 139.0]])
    else:
        board_texture_path = TEXTURE_DIR / "wood_texture.png"
        board_node.addObject("OglModel", texturename=str(board_texture_path))
        board_node.init()
        # Fix the texture coordinates
        with board_node.OglModel.texcoords.writeable() as texcoords:
            for index, coordinates in enumerate(texcoords):
                x, y, _ = np.unravel_index(index, grid_shape, "F")
                coordinates[:] = [x / grid_shape[0], y / grid_shape[1]]

    def sample_positions(rng: np.random.Generator) -> List[np.ndarray]:
        num_valid_posts = 0
        positions: List[np.ndarray] = []
        while num_valid_posts < num_objects:
            height = rng.uniform(*post_height_limits)
            x = rng.uniform(low=-width / 2 + 5, high=width / 2 - 5)
            y = rng.uniform(low=-depth / 2 + 5, high=depth / 2 - 5)

            distances_to_other_posts_too_small = []
            for position in zip(positions):
                distances_to_other_posts_too_small.append(np.linalg.norm(position - np.array([x, y, height])) < min_post_distance)

            if not any(distances_to_other_posts_too_small):
                positions.append(np.array([x, y, height]))
                num_valid_posts += 1
        return positions

    rng = np.random.default_rng()
    positions = sample_positions(rng)
    posts = []
    for position in positions:
        stiffness = np.random.uniform(*stiffness_limits)
        post = Post(
            parent_node=scene_node,
            name=f"post_{len(posts)}",
            position=position[:2],
            animation_loop_type=animation_loop,
            height=position[2],
            stiffness=stiffness,
        )
        posts.append(post)

    ########
    # Cauter
    ########
    right_cauter_visual = partial(add_visual_model, color=posts[0].colors[State.ACTIVE_RIGHT])
    right_cauter = PivotizedCauter(
        parent_node=scene_node,
        name="right_cauter",
        visual_mesh_path=INSTRUMENT_MESH_DIR / "dissection_electrode.stl",
        add_visual_model_func=right_cauter_visual,
        show_object=debug_rendering,
        show_object_scale=10,
        ptsd_state=np.array([-44.0, 76.0, 0.0, 40.0]),
        rcm_pose=np.array([cartesian_workspace["high"][0], cartesian_workspace["low"][1], cartesian_workspace["high"][2], -180.0, 0.0, 0.0]),
        animation_loop_type=animation_loop,
        spring_stiffness=1e19,
        angular_spring_stiffness=1e19,
        total_mass=1e5,
        cartesian_workspace=cartesian_workspace,
        state_limits={
            "low": np.array([-90, -90, -90, 0]),
            "high": np.array([90, 90, 90, 300]),
        },
        ptsd_reset_noise=cauter_reset_noise,
        collision_group=0,
    )

    scene_node.addObject(right_cauter)

    if not single_agent:
        left_cauter_visual = partial(add_visual_model, color=posts[0].colors[State.ACTIVE_LEFT])
        left_cauter = PivotizedCauter(
            parent_node=scene_node,
            name="left_cauter",
            visual_mesh_path=INSTRUMENT_MESH_DIR / "dissection_electrode.stl",
            add_visual_model_func=left_cauter_visual,
            show_object=debug_rendering,
            show_object_scale=10,
            ptsd_state=np.array([44.0, 45.0, 0.0, 15.0]),
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
            ptsd_reset_noise=cauter_reset_noise,
            collision_group=1,
        )
    else:
        left_cauter = None

    contact_listener = {"right_cauter": [], "left_cauter": []}
    for post in posts:
        contact_listener["right_cauter"].append(
            scene_node.addObject(
                "ContactListener",
                name=f"contact_listener_right_cauter_sphere_{post.name}",
                collisionModel1=right_cauter.cutting_sphere_collision_model.getLinkPath(),
                collisionModel2=post.sphere_collision_model.getLinkPath(),
            )
        )
        if not single_agent:
            contact_listener["left_cauter"].append(
                scene_node.addObject(
                    "ContactListener",
                    name=f"contact_listener_left_cauter_sphere_{post.name}",
                    collisionModel1=left_cauter.cutting_sphere_collision_model.getLinkPath(),
                    collisionModel2=post.sphere_collision_model.getLinkPath(),
                )
            )

    ###############
    # Returned data
    ###############
    scene_creation_result = {
        "camera": camera.sofa_object,
        "physical_camera": camera,
        "right_cauter": right_cauter,
        "left_cauter": left_cauter,
        "posts": posts,
        "sample_positions_func": sample_positions,
        "contact_listener": contact_listener,
    }

    return scene_creation_result
