import Sofa
import Sofa.Core

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

from sofa_env.scenes.rope_threading.sofa_objects.camera import ControllableCamera, CONTROLLABLE_CAMERA_KWARGS
from sofa_env.scenes.rope_threading.sofa_objects.gripper import ArticulatedGripper, GRIPPER_PLUGIN_LIST
from sofa_env.scenes.rope_threading.sofa_objects.eye import Eye, EYE_PLUGIN_LIST
from sofa_env.scenes.rope_threading.sofa_objects.transfer_rope import TransferRope, TRANSFER_ROPE_PLUGIN_LIST

from sofa_env.sofa_templates.scene_header import AnimationLoopType, ConstraintSolverType, IntersectionMethod, add_scene_header, VISUAL_STYLES, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.rope import RopeCollisionType, poses_for_linear_rope, ROPE_PLUGIN_LIST
from sofa_env.sofa_templates.rigid import MechanicalBinding, RIGID_PLUGIN_LIST

from sofa_env.utils.camera import determine_look_at
from sofa_env.utils.math_helper import euler_to_rotation_matrix, multiply_quaternions, rotation_matrix_to_quaternion

PLUGIN_LIST = (
    [
        "SofaPython3",
        "Sofa.Component.Constraint.Lagrangian.Solver",  # <- [GenericConstraintSolver]
        "Sofa.Component.Setting",  # <- [BackgroundSetting]
        "Sofa.Component.Visual",  # <- [VisualStyle]
        "Sofa.Component.Topology.Container.Grid",  # <- [RegularGridTopology]
        "Sofa.Component.MechanicalLoad",  # [PlaneForceField]
    ]
    + SCENE_HEADER_PLUGIN_LIST
    + ROPE_PLUGIN_LIST
    + TRANSFER_ROPE_PLUGIN_LIST
    + GRIPPER_PLUGIN_LIST
    + EYE_PLUGIN_LIST
    + RIGID_PLUGIN_LIST
    + CONTROLLABLE_CAMERA_KWARGS
)

LENGTH_UNIT = "mm"
TIME_UNIT = "s"
WEIGHT_UNIT = "kg"

HERE = Path(__file__).resolve().parent
INSTRUMENT_MESH_DIR = HERE.parent.parent.parent / "assets/meshes/instruments"
TEXTURE_DIR = HERE.parent.parent.parent / "assets/textures"


DEFAULT_EYE_CONFIG = [
    (60, 10, 0, 90),
    (10, 10, 0, 90),
    (10, 60, 0, -45),
    (60, 60, 0, 90),
]

DEFAULT_CARTESIAN_WORKSPACE_LIMITS = {
    "low": (-20.0, -20.0, 0.0),
    "high": (230.0, 180.0, 100.0),
}

def createScene(
    root_node: Sofa.Core.Node,
    image_shape: Tuple[Optional[int], Optional[int]] = (None, None),
    eye_config: List[Tuple[int, int, int, int]] = DEFAULT_EYE_CONFIG,
    eye_reset_noise: Union[None, Dict[str, np.ndarray], List[Dict[str, np.ndarray]]] = None,
    randomize_gripper: bool = False,
    randomize_grasp_index: bool = False,
    start_grasped: bool = True,
    debug_rendering: bool = False,
    positioning_camera: bool = False,
    animation_loop: AnimationLoopType = AnimationLoopType.FREEMOTION,
    camera_pose: np.ndarray = np.array([115.0, -60.0, 75.0, 0.42133736, 0.14723759, 0.29854957, 0.8436018]),
    # another sensible camera pose with broader view
    # [50.0, -75.0, 150.0, 0.3826834, 0.0, 0.0, 0.9238795]
    no_textures: bool = False,
    only_right_gripper: bool = False,
    gripper_and_rope_same_collision_group: bool = False,
    num_rope_points: int = 100,
    cartesian_workspace_limits: dict[str, tuple[float, float, float]] = DEFAULT_CARTESIAN_WORKSPACE_LIMITS,
) -> Dict:
    """
    Creates the scene of the RopeThreadingEnv.

    Args:
        root_node (Sofa.Core.Node): SET BY ENV. The simulation tree's root node.
        image_shape (Tuple[Optional[int], Optional[int]]): SET BY ENV. Height and Width of the rendered images.
        eye_config (List[Tuple[int, int, int, int]]): A list of [X, Y, Z, Angle] per eye to add to the board.
        eye_reset_noise (Union[None, Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]): Noise added to the eye pose when resetting the environment.
        randomize_gripper (bool): Whether to add noise to the grippers' pose on reset.
        randomize_grasp_index (bool): Whether to randomize where the gripper grasps the rope, if ``start_grasped=True``.
        start_grasped (bool): Whether the episode should start with the rope already grasped by the right gripper.
        debug_rendering (bool): Whether to render more information for debugging purposes.
        positioning_camera (bool): Whether the camera should be controllable from the ``runSofa`` binary.
        animation_loop (AnimationLoopType): Animation loop of the simulation.
        camera_pose (np.ndarray): Pose of the camera.
        no_textures (bool): Whether to not add texture to the visual models.
        only_right_gripper (bool): Whether to only add the right gripper.
        gripper_and_rope_same_collision_group (bool): Whether the gripper and rope should be in the same collision group.
        num_rope_points (int): Number of points to use for the rope.
        cartesian_workspace_limits (dict[str, tuple[float, float, float]]): The cartesian workspace limits imposed on the grippers.

    Returns:
        scene_creation_result = {
            "camera": camera,
            "interactive_objects": {
                "left_gripper": left_gripper,
                "right_gripper": right_gripper,
            },
            "rope": rope,
            "eyes": eyes,
            "contact_listeners": contact_listeners,
            "contact_listener_info": contact_listener_info,
        }
    """

    gravity = -981  # mm/s^2

    ###################
    # Common components
    ###################
    add_scene_header(
        root_node=root_node,
        plugin_list=PLUGIN_LIST,
        gravity=(0.0, 0.0, gravity),
        animation_loop=animation_loop,
        constraint_solver=ConstraintSolverType.GENERIC,
        visual_style_flags=VISUAL_STYLES["debug"] if debug_rendering else VISUAL_STYLES["normal"],
        scene_has_collisions=True,
        collision_detection_method=IntersectionMethod.MINPROXIMITY,
        collision_detection_method_kwargs={
            "alarmDistance": 1.0,
            "contactDistance": 0.05,
        },
        scene_has_cutting=False,
        contact_friction=0.0,
    )

    #####################################################
    # Dedicated scene node for the rest of the components
    #####################################################
    scene_node = root_node.addChild("scene")

    if eye_reset_noise is not None:
        if isinstance(eye_reset_noise, list):
            if not len(eye_reset_noise) == len(eye_config):
                raise ValueError(f"When specifying a per eye reset noise, please pass one dictionary per eye. Got {len(eye_reset_noise)}, expected {len(eye_config)}.")
        elif isinstance(eye_reset_noise, dict):
            eye_reset_noise = [eye_reset_noise for _ in range(len(eye_config))]
        else:
            raise TypeError(f"Expected Dict[str, np.ndarray] or list thereof. Got {type(eye_reset_noise)}.")
    else:
        eye_reset_noise = [None for _ in range(len(eye_config))]

    eyes = []
    render_index = len(eye_config) > 1
    for index, eye_placement in enumerate(eye_config):
        eyes.append(
            Eye(
                parent_node=scene_node,
                position=(eye_placement[0], eye_placement[1], eye_placement[2]),
                rotation=eye_placement[3],
                name=f"eye_{index}",
                index=index,
                animation_loop_type=animation_loop,
                surface_mesh_path=HERE / "meshes/eye.stl",
                show_object=debug_rendering,
                show_object_scale=10,
                position_reset_noise=eye_reset_noise[index],
                render_index=render_index,
            )
        )

    poses_for_rope = poses_for_linear_rope(length=200, num_points=num_rope_points, start_position=np.array([70.0, 0.0, 34.0]))
    # Rotate the rope such that y axis of gripper and rope align
    transformation_quaternion = rotation_matrix_to_quaternion(euler_to_rotation_matrix(np.array([180.0, 0.0, 0.0])))
    poses_for_rope = [np.append(pose[:3], multiply_quaternions(transformation_quaternion, pose[3:])) for pose in poses_for_rope]

    rope = TransferRope(
        parent_node=scene_node,
        name="rope",
        waypoints=[eye.center_pose[:3] for eye in eyes],
        radius=1.0,
        beam_radius=2.0,
        total_mass=5.0,
        poisson_ratio=0.0,
        young_modulus=5e4,
        fix_start=False,
        fix_end=False,
        collision_type=RopeCollisionType.SPHERES,
        animation_loop_type=animation_loop,
        mechanical_damping=1.0,
        show_object=debug_rendering,
        show_object_scale=3,
        poses=poses_for_rope,
        collision_group=1,
    )

    cartesian_workspace = {
        "low": np.array(cartesian_workspace_limits["low"]),
        "high": np.array(cartesian_workspace_limits["high"]),
    }

    board_limits = {
        "low": np.array([np.inf, np.inf, -5]),
        "high": np.array([-np.inf, -np.inf, 0]),
    }

    # Adapt the board size to the eyes
    for index, eye in enumerate(eyes):
        xyz_min = np.array(eye.position) - [5.0, 5.0, 0.0]
        xyz_max = np.array(eye.position) + [5.0, 5.0, 0.0]
        reset_noise = eye_reset_noise[index]
        if reset_noise is not None:
            xyz_min = xyz_min + reset_noise["low"][:3]
            xyz_max = xyz_max + reset_noise["high"][:3]

        board_limits["low"] = np.minimum(board_limits["low"], xyz_min)
        board_limits["high"] = np.maximum(board_limits["high"], xyz_max)

    grid_shape = (10, 10, 2)
    board_node = scene_node.addChild("board")

    # Center position of the board
    board_pose = np.zeros(7)
    board_pose[:3] = (board_limits["low"] + board_limits["high"]) / 2
    board_pose[-1] = 1.0

    # Add the mechanical object that holds the mechanical state of the rigid object
    board_mechanical_object = board_node.addObject(
        "MechanicalObject",
        template="Rigid3d",
        position=board_pose,
        showObject=debug_rendering,
        showObjectScale=10.0,
    )

    board_visual_node = board_node.addChild("visual")
    board_visual_node.addObject("RegularGridTopology", min=board_limits["low"], max=board_limits["high"], n=grid_shape)
    if no_textures:
        board_visual_node.addObject("OglModel", color=[value / 255 for value in [246.0, 205.0, 139.0]])
        board_visual_node.addObject("RigidMapping", globalToLocalCoords=True)
    else:
        board_texture_path = TEXTURE_DIR / "wood_texture.png"
        board_visual_node.addObject("OglModel", texturename=str(board_texture_path))
        board_visual_node.addObject("RigidMapping", globalToLocalCoords=True)
        board_node.init()
        # Fix the texture coordinates
        with board_visual_node.OglModel.texcoords.writeable() as texcoords:
            for index, coordinates in enumerate(texcoords):
                x, y, _ = np.unravel_index(index, grid_shape, "F")
                coordinates[:] = [x / grid_shape[0], y / grid_shape[1]]

    left_gripper = None
    if not only_right_gripper:
        left_gripper = ArticulatedGripper(
            parent_node=scene_node,
            name="left_gripper",
            visual_mesh_path_shaft=INSTRUMENT_MESH_DIR / "instrument_shaft.stl",
            visual_mesh_paths_jaws=[INSTRUMENT_MESH_DIR / "forceps_jaw_left.stl", INSTRUMENT_MESH_DIR / "forceps_jaw_right.stl"],
            rope_to_grasp=rope,
            ptsd_state=np.array([0.0, 0.0, 180.0, 50.0]),
            rcm_pose=np.array([0.0, 0.0, 100.0, 0.0, 180.0, 0.0]),
            collision_spheres_config={
                "positions": [[0, 0, 5 + i * 2] for i in range(10)],
                "backside": [[0, -1.5, 5 + i * 2] for i in range(10)],
                "radii": [1] * 10,
            },
            jaw_length=25,
            angle=20.0,
            angle_limits=(0.0, 60.0),
            total_mass=1e12,
            mechanical_binding=MechanicalBinding.SPRING,
            animation_loop_type=animation_loop,
            show_object=debug_rendering,
            show_object_scale=10,
            show_remote_center_of_motion=debug_rendering,
            state_limits={
                "low": np.array([-90.0, -90.0, np.finfo(np.float16).min, 0.0]),
                "high": np.array([90.0, 90.0, np.finfo(np.float16).max, 100.0]),
            },
            spring_stiffness=1e29,
            angular_spring_stiffness=1e29,
            articulation_spring_stiffness=1e29,
            spring_stiffness_grasping=1e9,
            angular_spring_stiffness_grasping=1e9,
            angle_to_grasp_threshold=10.0,
            angle_to_release_threshold=15.0,
            collision_group=1 if gripper_and_rope_same_collision_group else 0,
            collision_contact_stiffness=100,
            cartesian_workspace=cartesian_workspace,
            start_grasped=False,
            grasp_index_pair=(0, 0),
            ptsd_reset_noise=np.array([10.0, 10.0, 45.0, 10.0]) if randomize_gripper else None,
            angle_reset_noise=20.0 if randomize_gripper else None,
            deactivate_collision_while_grasped=False if gripper_and_rope_same_collision_group else True,
        )

    right_gripper = ArticulatedGripper(
        parent_node=scene_node,
        name="right_gripper",
        visual_mesh_path_shaft=INSTRUMENT_MESH_DIR / "instrument_shaft.stl",
        visual_mesh_paths_jaws=[INSTRUMENT_MESH_DIR / "forceps_jaw_left.stl", INSTRUMENT_MESH_DIR / "forceps_jaw_right.stl"],
        rope_to_grasp=rope,
        ptsd_state=np.array([0.0, 0.0, 180.0, 50.0]),
        rcm_pose=np.array([100.0, 0.0, 100.0, 0.0, 180.0, 0.0]),
        collision_spheres_config={
            "positions": [[0, 0, 5 + i * 2] for i in range(10)],
            "backside": [[0, -1.5, 5 + i * 2] for i in range(10)],
            "radii": [1] * 10,
        },
        jaw_length=25,
        angle=5.0,
        angle_limits=(0.0, 60.0),
        total_mass=1e12,
        mechanical_binding=MechanicalBinding.SPRING,
        animation_loop_type=animation_loop,
        show_object=debug_rendering,
        show_object_scale=10,
        show_remote_center_of_motion=debug_rendering,
        state_limits={
            "low": np.array([-90, -90, np.iinfo(np.int16).min, 0]),
            "high": np.array([90, 90, np.iinfo(np.int16).max, 100]),
        },
        spring_stiffness=1e29,
        angular_spring_stiffness=1e29,
        articulation_spring_stiffness=1e29,
        spring_stiffness_grasping=1e9,
        angular_spring_stiffness_grasping=1e9,
        angle_to_grasp_threshold=10.0,
        angle_to_release_threshold=15.0,
        collision_group=1 if gripper_and_rope_same_collision_group else 0,
        collision_contact_stiffness=100,
        cartesian_workspace=cartesian_workspace,
        start_grasped=start_grasped,
        grasp_index_pair=(5, 15),
        grasp_index_reset_noise={"low": -5, "high": 15} if randomize_grasp_index else None,
        ptsd_reset_noise=np.array([10.0, 10.0, 45.0, 10.0]) if randomize_gripper else None,
        angle_reset_noise=None,
        deactivate_collision_while_grasped=False if gripper_and_rope_same_collision_group else True,
    )

    if not positioning_camera:
        if not only_right_gripper:
            scene_node.addObject(left_gripper)
        scene_node.addObject(right_gripper)

    contact_listeners = {"left_gripper": [], "right_gripper": []}
    contact_listener_info = {"left_gripper": [], "right_gripper": []}

    for gripper_name in ("left_gripper", "right_gripper"):
        instrument = right_gripper if gripper_name == "right_gripper" else left_gripper
        if instrument is None:
            continue
        for jaw_name in ("jaw_0", "jaw_1"):
            for eye in eyes:
                collision_model = instrument.sphere_collisions_jaw_0 if jaw_name == "jaw_0" else instrument.sphere_collisions_jaw_1
                contact_listener = eye.rigid_object.node.addObject(
                    "ContactListener",
                    name=f"contact_listener_{eye.name}_{gripper_name}_{jaw_name}",
                    collisionModel1=eye.collision_node.SphereCollisionModel.getLinkPath(),
                    collisionModel2=collision_model.getLinkPath(),
                )

                contact_listeners[gripper_name].append(contact_listener)
                contact_listener_info[gripper_name].append(
                    {
                        "jaw": jaw_name,
                        "eye": eye.name,
                    }
                )

    rope.node.addObject(
        "PlaneForceField",
        normal=[0, 0, 1],
        d=rope.radius,
        stiffness=300,
        damping=1,
        showPlane=True,
        showPlaneSize=0.1,
    )

    ###################
    # Camera and lights
    ###################
    root_node.addObject(
        "LightManager",
        listening=True,
        ambient=(
            0.8,
            0.8,
            0.8,
            0.8,
        ),
    )

    placement_kwargs = {
        "position": camera_pose[:3],
        "orientation": camera_pose[3:],
    }
    placement_kwargs["lookAt"] = determine_look_at(camera_position=placement_kwargs["position"], camera_orientation=placement_kwargs["orientation"])
    light_source_kwargs = {
        "cutoff": 45.0 / 1.2,
        "color": [0.8] * 4,
        "attenuation": 0.0,
        "exponent": 1.0,
        "shadowsEnabled": False,
    }
    camera = ControllableCamera(
        root_node=root_node,
        placement_kwargs=placement_kwargs,
        with_light_source=True,
        show_object=debug_rendering,
        show_object_scale=10.0,
        light_source_kwargs=light_source_kwargs,
        width_viewport=image_shape[0],
        height_viewport=image_shape[1],
        z_near=1,
        z_far=float(np.linalg.norm(cartesian_workspace["high"] - cartesian_workspace["low"])) * 1.5,
    )

    if positioning_camera:
        root_node.addObject(camera)

    ###############
    # Returned data
    ###############
    scene_creation_result = {
        "camera": camera,
        "interactive_objects": {
            "left_gripper": left_gripper,
            "right_gripper": right_gripper,
        },
        "rope": rope,
        "eyes": eyes,
        "contact_listeners": contact_listeners,
        "contact_listener_info": contact_listener_info,
        "board_mo": board_mechanical_object,
    }

    return scene_creation_result
