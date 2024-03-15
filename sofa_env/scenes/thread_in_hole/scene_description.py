import Sofa
import Sofa.Core
import Sofa.Simulation

import numpy as np

from typing import Optional, Tuple, Union, Dict
from pathlib import Path
from sofa_env.sofa_templates.camera import Camera, CAMERA_PLUGIN_LIST
from sofa_env.sofa_templates.rigid import MechanicalBinding, RIGID_PLUGIN_LIST

from sofa_env.sofa_templates.rope import Rope, RopeCollisionType, poses_for_linear_rope, ROPE_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, IntersectionMethod, add_scene_header, VISUAL_STYLES, SCENE_HEADER_PLUGIN_LIST

from sofa_env.scenes.rope_threading.sofa_objects.gripper import ArticulatedGripper, GRIPPER_PLUGIN_LIST
from sofa_env.scenes.ligating_loop.sofa_objects.cavity import Cavity, CAVITY_PLUGIN_LIST
from sofa_env.utils.pivot_transform import ptsd_to_pose

PLUGIN_LIST = (
    [
        "Sofa.Component.Engine.Select",  # <- [BoxROI]
        "Sofa.Component.StateContainer",  # <- [MechanicalObject]
    ]
    + SCENE_HEADER_PLUGIN_LIST
    + GRIPPER_PLUGIN_LIST
    + ROPE_PLUGIN_LIST
    + CAMERA_PLUGIN_LIST
    + CAVITY_PLUGIN_LIST
    + RIGID_PLUGIN_LIST
)

LENGTH_UNIT = "mm"
TIME_UNIT = "s"
WEIGHT_UNIT = "kg"

HERE = Path(__file__).resolve().parent
INSTRUMENT_MESH_DIR = HERE.parent.parent.parent / "assets/meshes/instruments"


def createScene(
    root_node: Sofa.Core.Node,
    image_shape: Tuple[Optional[int], Optional[int]] = (None, None),
    debug_rendering: bool = False,
    animation_loop: AnimationLoopType = AnimationLoopType.FREEMOTION,
    randomize_gripper: bool = True,
    thread_config: dict = {
        "length": 50.0,
        "radius": 2.0,
        "total_mass": 1.0,
        "young_modulus": 4000.0,
        "poisson_ratio": 0.3,
        "beam_radius": 5.0,
        "mechanical_damping": 0.2,
    },
    hole_config: dict = {
        "inner_radius": 5.0,
        "outer_radius": 25.0,
        "height": 30.0,
        "young_modulus": 5000.0,
        "poisson_ratio": 0.3,
        "total_mass": 10.0,
    },
    gripper_config: dict = {
        "cartesian_workspace": {
            "low": np.array([-100.0] * 2 + [0.0]),
            "high": np.array([100.0] * 2 + [200.0]),
        },
        "state_reset_noise": np.array([15.0, 15.0, 0.0, 20.0]),
        "rcm_reset_noise": np.array([10.0, 10.0, 10.0, 5.0, 5.0, 5.0]),
        "gripper_ptsd_state": np.array([60.0, 0.0, 180.0, 90.0]),
        "gripper_rcm_pose": np.array([100.0, 0.0, 150.0, 0.0, 180.0, 0.0]),
    },
    camera_config: dict = {
        "placement_kwargs": {
            "position": [0.0, -135.0, 100.0],
            "lookAt": [0.0, 0.0, 45.0],
        },
        "vertical_field_of_view": 62.0,
    },
    create_shell: bool = False,
    hole_position_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    hole_rotation_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
) -> Dict:
    """
    Creates the scene of the ThreadInHoleEnv.

    Args:
        root_node (Sofa.Core.Node): SET BY ENV. The simulation tree's root node.
        image_shape (Tuple[Optional[int], Optional[int]]): SET BY ENV. Height and Width of the rendered images.
        debug_rendering (bool): Whether to render more information for debugging purposes.
        animation_loop (AnimationLoopType): Animation loop of the simulation.
        randomize_gripper (bool): Whether to randomize the gripper state at reset.
        thread_config (dict):
            "length": Length of the thread.
            "radius": Radius of the thread for visual and collision.
            "total_mass": The thread's total mass.
            "young_modulus": The thread's Young modulus.
            "poisson_ratio": The thread's poisson ratio.
            "beam_radius": Mechanical radius of the beam.
            "mechanical_damping": Uniform velocity damping.
        hole_config (dict):
            "inner_radius": The hole's inner radius.
            "outer_radius": The hole's outer radius.
            "height": The hole's height.
            "young_modulus": The hole's Young modulus.
            "poisson_ratio": The hole's poisson ratio.
            "total_mass": The hole's total mass.
        gripper_config (dict):
            "cartesian_workspace": Dictionary with Cartesian low and high values of the gripper's workspace.
            "state_reset_noise": Optional noise to uniformly sample from that is added to the PTSD state of the gripper.
            "rcm_reset_noise": Optional noise to uniformly sample from that is added to the XYZ position and Euler XYZ angles of the RCM.
            "gripper_ptsd_state": Initial PTSD state of the gripper.
            "gripper_rcm_pose": Initial PTSD state of the RCM.
        camera_config (dict):
            "placement_kwargs": Dictionary with camera placement information.
            "vertical_field_of_view": Vertical field of view in degrees.
        create_shell (bool): SET BY ENV. Whether to create a mechanical model of the outher shell of the hole for checking if the thread is inside the hole.
        hole_position_reset_noise (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): SET BY ENV. Optional noise to uniformly sample from that is added to the initial position of the hole.
        hole_rotation_reset_noise (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): SET BY ENV. Optional noise to uniformly sample from that is added to the XYZ Euler angle rotation of the hole.

    Returns:
        scene_creation_results = {
            "camera": camera,
            "gripper": gripper,
            "thread": gripper.rope,
            "hole": hole,
            "root": root_node,
            "contact_listeners": (contact_listener_jaw_0, contact_listener_jaw_1),
        }
    """

    gravity = -981  # mm/s^2

    ###################
    # Common components
    ###################
    contact_distance = 0.2
    add_scene_header(
        root_node=root_node,
        plugin_list=PLUGIN_LIST,
        visual_style_flags=VISUAL_STYLES["debug"] if debug_rendering else VISUAL_STYLES["normal"],
        gravity=(0.0, 0.0, gravity),
        animation_loop=animation_loop,
        scene_has_collisions=True,
        collision_detection_method_kwargs={
            "alarmDistance": max(hole_config["inner_radius"], thread_config["radius"]),
            "contactDistance": contact_distance,
        },
        collision_detection_method=IntersectionMethod.LOCALMIN,
        contact_friction=1e-6,
    )

    ###################
    # Camera and lights
    ###################
    root_node.addObject(
        "LightManager",
        listening=True,
        ambient=(
            0.3,
            0.3,
            0.3,
            0.3,
        ),
    )
    if not debug_rendering:
        root_node.addObject("DirectionalLight", name="light1", direction=(2, 1, -1), color=(0.8, 0.8, 0.8, 1.0))
        root_node.addObject("DirectionalLight", name="light2", direction=(0, -1, 0), color=(0.8, 0.8, 0.8, 1.0))
        root_node.addObject("OglShadowShader")

    camera = Camera(
        root_node=root_node,
        placement_kwargs=camera_config["placement_kwargs"],
        z_near=1.0,
        z_far=250.0,
        width_viewport=image_shape[0],
        height_viewport=image_shape[1],
        show_object=debug_rendering,
        show_object_scale=5.0,
        vertical_field_of_view=camera_config["vertical_field_of_view"],
    )

    #####################################################
    # Dedicated scene node for the rest of the components
    #####################################################
    scene_node = root_node.addChild("scene")

    # Visualize the Cartesian workspace
    if debug_rendering:
        scene_node.addObject("MechanicalObject")
        scene_node.addObject("BoxROI", box=gripper_config["cartesian_workspace"]["low"].tolist() + gripper_config["cartesian_workspace"]["high"].tolist(), drawBoxes=True, drawSize=3.0)

    #################
    # Hollow cylinder
    #################
    hole = Cavity(
        parent_node=scene_node,
        name="hole",
        inner_radius=hole_config["inner_radius"],
        outer_radius=hole_config["outer_radius"],
        height=hole_config["height"],
        total_mass=hole_config["total_mass"],
        young_modulus=hole_config["young_modulus"],
        poisson_ratio=hole_config["poisson_ratio"],
        animation_loop_type=animation_loop,
        show_object=debug_rendering,
        show_object_scale=10.0,
        collision_group=0,
        check_self_collision=False,
        discretization_radius=2,
        discretization_angle=10,
        discretization_height=10,
        create_shell=create_shell,
        color=(0.0, 1.0, 1.0),
        fixed_lenght=1.0,
        position_reset_noise=hole_position_reset_noise,
        rotation_reset_noise=hole_rotation_reset_noise,
    )

    ########
    # Thread
    ########
    gripper_ptsd_state = gripper_config["gripper_ptsd_state"]
    gripper_rcm_pose = gripper_config["gripper_rcm_pose"]
    gripper_pose = ptsd_to_pose(gripper_ptsd_state, gripper_rcm_pose)

    length = thread_config["length"]
    radius = thread_config["radius"]
    num_points = int(np.floor(length / (radius * 2)))
    rope_poses = poses_for_linear_rope(
        length=length,
        num_points=num_points,
        vector=np.array([0.0, 0.0, -1.0]),
        start_position=gripper_pose[:3],
    )

    rope_kwargs = {
        "parent_node": scene_node,
        "name": "thread",
        "radius": radius,
        "poses": rope_poses,
        "total_mass": thread_config["total_mass"],
        "young_modulus": thread_config["young_modulus"],
        "poisson_ratio": thread_config["poisson_ratio"],
        "beam_radius": thread_config["beam_radius"],
        "mechanical_damping": thread_config["mechanical_damping"],
        "collision_type": RopeCollisionType.SPHERES,
        "animation_loop_type": animation_loop,
        "collision_group": 1,
    }

    thread = Rope(**rope_kwargs)

    gripper = scene_node.addObject(
        ArticulatedGripper(
            parent_node=scene_node,
            name="gripper",
            visual_mesh_path_shaft=INSTRUMENT_MESH_DIR / "instrument_shaft.stl",
            visual_mesh_paths_jaws=[INSTRUMENT_MESH_DIR / "forceps_jaw_left.stl", INSTRUMENT_MESH_DIR / "forceps_jaw_right.stl"],
            rope_to_grasp=thread,
            ptsd_state=gripper_ptsd_state,
            rcm_pose=gripper_rcm_pose,
            collision_group=1,
            jaw_length=25,
            angle=5.0,
            angle_limits=(0.0, 60.0),
            total_mass=0.1,
            mechanical_binding=MechanicalBinding.ATTACH,
            animation_loop_type=animation_loop,
            show_object=debug_rendering,
            show_object_scale=10,
            show_remote_center_of_motion=debug_rendering,
            state_limits={
                "low": np.array([-90, -90, np.iinfo(np.int16).min, 0]),
                "high": np.array([90, 90, np.iinfo(np.int16).max, 200]),
            },
            spring_stiffness=1e8,
            angular_spring_stiffness=1e8,
            articulation_spring_stiffness=1e15,
            spring_stiffness_grasping=1e9,
            angular_spring_stiffness_grasping=1e9,
            angle_to_grasp_threshold=10.0,
            angle_to_release_threshold=np.inf,
            cartesian_workspace=gripper_config["cartesian_workspace"],
            start_grasped=True,
            grasp_index_pair=(5, 0),
            ptsd_reset_noise=gripper_config["state_reset_noise"] if randomize_gripper else None,
            rcm_reset_noise=gripper_config["rcm_reset_noise"] if randomize_gripper else None,
            angle_reset_noise=None,
            deactivate_collision_while_grasped=False,
            recalculate_orientation_offset=False,
        )
    )
    contact_listener_jaw_0 = scene_node.addObject(
        "ContactListener",
        name="contact_listener_cylinder_jaw_0",
        collisionModel1=hole.triangle_collision_model.getLinkPath(),
        collisionModel2=gripper.sphere_collisions_jaw_0.getLinkPath(),
    )
    contact_listener_jaw_1 = scene_node.addObject(
        "ContactListener",
        name="contact_listener_cylinder_jaw_1",
        collisionModel1=hole.triangle_collision_model.getLinkPath(),
        collisionModel2=gripper.sphere_collisions_jaw_1.getLinkPath(),
    )

    scene_creation_results = {
        "camera": camera,
        "gripper": gripper,
        "thread": gripper.rope,
        "hole": hole,
        "root": root_node,
        "contact_listeners": (contact_listener_jaw_0, contact_listener_jaw_1),
    }

    return scene_creation_results
