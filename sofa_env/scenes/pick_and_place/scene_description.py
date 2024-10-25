import Sofa
import Sofa.Core

import numpy as np

from functools import partial
from typing import Optional, Tuple, Dict
from pathlib import Path

from sofa_env.sofa_templates.collision import add_collision_model, CollisionModelType, COLLISION_PLUGIN_LIST
from sofa_env.sofa_templates.rigid import MechanicalBinding, RigidObject, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.rope import Rope, poses_for_circular_rope, ROPE_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, ConstraintSolverType, IntersectionMethod, add_scene_header, VISUAL_STYLES, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.camera import Camera, CAMERA_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, set_color, VISUAL_PLUGIN_LIST

from sofa_env.scenes.rope_threading.sofa_objects.gripper import ArticulatedGripper, GRIPPER_PLUGIN_LIST
from sofa_env.utils.camera import determine_look_at
from sofa_env.utils.math_helper import euler_to_rotation_matrix, rotation_matrix_to_quaternion
from sofa_env.utils.pivot_transform import sofa_orientation_to_camera_orientation


PLUGIN_LIST = (
    [
        "SofaPython3",
        "SofaMiscCollision",
        "Sofa.Component.MechanicalLoad",  # [PlaneForceField]
        "Sofa.Component.Topology.Container.Grid",  # [CylinderGridTopology]
    ]
    + SCENE_HEADER_PLUGIN_LIST
    + RIGID_PLUGIN_LIST
    + COLLISION_PLUGIN_LIST
    + CAMERA_PLUGIN_LIST
    + ROPE_PLUGIN_LIST
    + GRIPPER_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
)

LENGTH_UNIT = "mm"
TIME_UNIT = "s"
WEIGHT_UNIT = "kg"

HERE = Path(__file__).resolve().parent
INSTRUMENT_MESH_DIR = HERE.parent.parent.parent / "assets/meshes/instruments"
TEXTURE_DIR = HERE.parent.parent.parent / "assets/textures"


class Peg(RigidObject):
    def set_color(self, new_color: Tuple[int, int, int]):
        set_color(self.visual_model_node.OglModel, color=tuple(color / 255 for color in new_color))


def createScene(
    root_node: Sofa.Core.Node,
    image_shape: Tuple[Optional[int], Optional[int]] = (None, None),
    debug_rendering: bool = False,
    animation_loop: AnimationLoopType = AnimationLoopType.FREEMOTION,
    start_grasped: bool = True,
    gripper_randomization: Optional[dict] = {
        "angle_reset_noise": 10.0,
        "ptsd_reset_noise": np.array([10.0, 10.0, 10.0, 5.0]),
        "rcm_reset_noise": np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0]),
    },
    torus_parameters: Dict[str, float] = {
        "beam_radius": 2.0,
        "young_modulus": 1e3,
        "mechanical_damping": 1.0,
        "total_mass": 5.0,
        "poisson_ratio": 0.0,
    },
) -> Dict:
    """
    Creates the scene of the PickAndPlaceEnv.

    Args:
        root_node (Sofa.Core.Node): SET BY ENV. The simulation tree's root node.
        image_shape (Tuple[Optional[int], Optional[int]]): SET BY ENV. Height and Width of the rendered images.
        debug_rendering (bool): Whether to render more information for debugging purposes.
        animation_loop (AnimationLoopType): Animation loop of the simulation.
        start_grasped (bool): SET BY ENV. Whether the episode should start with the torus already grasped by the gripper.
        gripper_randomization (Optional[dict]): Parameters for the randomization of the gripper.
        torus_parameters (Dict[str, float]): Parameters for the mechanical behavior of the torus.

    Returns:
        scene_creation_result = {
            "root_node": root_node,
            "gripper": gripper,
            "camera": camera,
            "torus": torus,
            "pegs": pegs,
            "target_positions": target_positions,
            "contact_listeners": contact_listeners,
        }
    """

    gravity = -981.0  # mm/s^2

    ########
    # Meshes
    ########
    board_visual_surface_mesh_path = HERE / "meshes/board.obj"
    board_texture_path = TEXTURE_DIR / "wood_texture.png"

    peg_visual_surface_mesh_path = HERE / "meshes/peg.stl"
    peg_collision_surface_mesh_path = HERE / "meshes/peg_collision.stl"

    ###################
    # Common components
    ###################
    add_scene_header(
        root_node=root_node,
        plugin_list=PLUGIN_LIST,
        visual_style_flags=VISUAL_STYLES["debug"] if debug_rendering else VISUAL_STYLES["normal"],
        gravity=(0.0, gravity, 0.0),
        animation_loop=animation_loop,
        scene_has_collisions=True,
        collision_detection_method_kwargs={
            "alarmDistance": 3.0,
            "contactDistance": 1.0,
        },
        constraint_solver=ConstraintSolverType.GENERIC,
        constraint_solver_kwargs={
            "maxIt": 100,
            "tolerance": 0.0001,
        },
        collision_detection_method=IntersectionMethod.LOCALMIN,
        contact_friction=0.9,
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

    camera_position = (0.0, 150.0, -160.0)
    camera_orientation = sofa_orientation_to_camera_orientation(rotation_matrix_to_quaternion(euler_to_rotation_matrix(np.array([45.0, 0.0, 180.0]))))
    # Setting only orientation or lookAt gives absolutely wrong values for camera pose.
    # See https://github.com/sofa-framework/sofa/issues/2727
    # Hence, we additionally calculate a lookAt from the orientation.
    look_at = determine_look_at(np.array(camera_position), np.array(camera_orientation))

    camera = Camera(
        root_node=root_node,
        placement_kwargs={
            "position": camera_position,
            "orientation": camera_orientation,
            "lookAt": look_at,
        },
        z_near=1,
        z_far=350,
        show_object=debug_rendering,
        show_object_scale=15.0,
        vertical_field_of_view=60,
        width_viewport=image_shape[0],
        height_viewport=image_shape[1],
    )

    root_node.addObject("PositionalLight", position=camera_position)

    #####################################################
    # Dedicated scene node for the rest of the components
    #####################################################
    scene_node = root_node.addChild("scene")

    #######
    # Board
    #######
    world_zero = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    board_visual_function = partial(add_visual_model, texture_file_path=str(board_texture_path))
    RigidObject(
        parent_node=scene_node,
        name="board",
        pose=world_zero,
        visual_mesh_path=board_visual_surface_mesh_path,
        add_visual_model_func=board_visual_function,
        animation_loop_type=animation_loop,
        fixed_position=True,
        fixed_orientation=True,
        total_mass=1.0,
    )

    #######
    # Torus
    #######
    radius = 14.0
    num_points = 30

    if start_grasped:
        # Rope is grasped -> set the starting position such that the rope is between the gripper jaws on start
        rope_euler_angles = np.array([0.0, 0.0, 0.0])
        rope_start_position = np.array([0.0, 35.0, 20.0])
    else:
        # Rope is grasped -> set the starting position such that the rope is between the gripper jaws on start
        rope_euler_angles = np.array([90.0, 0.0, 0.0])
        rope_start_position = np.array([40.0, 40.0, -3.5])

    rope_poses = poses_for_circular_rope(radius=radius, num_points=num_points, start_position=rope_start_position, euler_angle_rotation=rope_euler_angles)

    collision_model_indices = list(range(0, len(rope_poses) - 2, 3))

    torus = Rope(
        parent_node=scene_node,
        name="torus",
        radius=3.5,
        poses=rope_poses,
        animation_loop_type=animation_loop,
        rope_color=(0, 0, 255),
        show_object=True,
        fix_start=False,
        fix_end=False,
        collision_group=2,
        check_self_collision=True,
        collision_model_indices=collision_model_indices,
        **torus_parameters,
    )
    torus.node.addObject(
        "PlaneForceField",
        normal=[0, 1, 0],
        d=torus.radius,
        stiffness=300,
        damping=1,
        showPlane=True,
        showPlaneSize=100,
    )
    torus.node.addObject(
        "BilateralLagrangianConstraint",
        template="Rigid3d",
        object1=torus.mechanical_object.getLinkPath(),
        object2=torus.mechanical_object.getLinkPath(),
        first_point=0,
        second_point=len(rope_poses) - 1,
    )

    ######
    # Pegs
    ######
    add_peg_collision = partial(
        add_collision_model,
        contact_stiffness=None,
        model_types=[CollisionModelType.LINE, CollisionModelType.TRIANGLE],
        is_static=True,
        collision_group=1,
    )
    pegs = []
    target_positions = []
    for x in (-80, 0, 80):
        for z in (-80, 0, 80):
            pegs.append(
                Peg(
                    parent_node=scene_node,
                    name=f"peg_{x}_{z}",
                    pose=(x, 0, z) + (0.0, 0.0, 0.0, 1.0),
                    visual_mesh_path=peg_visual_surface_mesh_path,
                    collision_mesh_path=peg_collision_surface_mesh_path,
                    animation_loop_type=animation_loop,
                    add_collision_model_func=add_peg_collision,
                    fixed_position=True,
                    fixed_orientation=True,
                    total_mass=1.0,
                )
            )
            target_positions.append(np.array((x, 8.1, z), dtype=np.float32))

    target_positions = np.asarray(target_positions)

    #########
    # Gripper
    #########
    if gripper_randomization is None:
        gripper_randomization = {}

    gripper = scene_node.addObject(
        ArticulatedGripper(
            parent_node=scene_node,
            name="gripper",
            rcm_pose=np.array([0.0, 115.0, 20.0, 90.0, 0.0, 0.0]),
            ptsd_state=np.array([0.0, 0.0, 180.0, 50.0]),
            visual_mesh_path_shaft=INSTRUMENT_MESH_DIR / "instrument_shaft.stl",
            visual_mesh_paths_jaws=[INSTRUMENT_MESH_DIR / "forceps_jaw_left.stl", INSTRUMENT_MESH_DIR / "forceps_jaw_right.stl"],
            mechanical_binding=MechanicalBinding.SPRING,
            total_mass=1e12,
            rope_to_grasp=torus,
            angle=11.0,
            animation_loop_type=animation_loop,
            show_object=debug_rendering,
            show_object_scale=10,
            show_remote_center_of_motion=debug_rendering,
            state_limits={
                "low": np.array([-90, -90, np.iinfo(np.int16).min, 0]),
                "high": np.array([90, 90, np.iinfo(np.int16).max, 150]),
            },
            cartesian_workspace={
                "low": np.array([-120.0, 0.0, -120.0]),
                "high": np.array([120.0, 100.0, 120.0]),
            },
            angle_to_grasp_threshold=18.0,
            angle_to_release_threshold=25.0,
            collision_group=3,
            deactivate_collision_while_grasped=True,
            start_grasped=start_grasped,
            spring_stiffness=1e29,
            angular_spring_stiffness=1e29,
            articulation_spring_stiffness=1e29,
            spring_stiffness_grasping=1e9,
            angular_spring_stiffness_grasping=1e9,
            grasp_index_pair=(6, 5),
            **gripper_randomization,
        )
    )

    contact_listener_node = scene_node.addChild("contact_listener")
    contact_listeners = []
    for peg in pegs:
        contact_listeners.append(
            contact_listener_node.addObject(
                "ContactListener",
                name=f"contact_listener_{peg.name}_gripper_jaw_0",
                collisionModel1=peg.collision_model_node.TriangleCollisionModel.getLinkPath(),
                collisionModel2=gripper.sphere_collisions_jaw_0.getLinkPath(),
            )
        )
        contact_listeners.append(
            contact_listener_node.addObject(
                "ContactListener",
                name=f"contact_listener_{peg.name}_gripper_jaw_1",
                collisionModel1=peg.collision_model_node.TriangleCollisionModel.getLinkPath(),
                collisionModel2=gripper.sphere_collisions_jaw_1.getLinkPath(),
            )
        )

    scene_creation_result = {
        "root_node": root_node,
        "gripper": gripper,
        "camera": camera,
        "torus": torus,
        "pegs": pegs,
        "target_positions": target_positions,
        "contact_listeners": contact_listeners,
    }

    return scene_creation_result
