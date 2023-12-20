import Sofa
import Sofa.Core
import Sofa.Simulation

import numpy as np

from typing import Optional, Tuple, Dict
from pathlib import Path

from sofa_env.utils.camera import determine_look_at

from sofa_env.sofa_templates.rigid import MechanicalBinding, RIGID_PLUGIN_LIST

from sofa_env.scenes.rope_threading.sofa_objects.camera import ControllableCamera
from sofa_env.scenes.ligating_loop.sofa_objects.loop import LigatingLoop, LOOP_PLUGIN_LIST
from sofa_env.scenes.ligating_loop.sofa_objects.gripper import ArticulatedGripper, GRIPPER_PLUGIN_LIST
from sofa_env.scenes.ligating_loop.sofa_objects.cavity import Cavity, CAVITY_PLUGIN_LIST
from sofa_env.scenes.ligating_loop.ligating_loop_env import ActionType


from sofa_env.sofa_templates.scene_header import AnimationLoopType, ConstraintSolverType, IntersectionMethod, add_scene_header, VISUAL_STYLES, SCENE_HEADER_PLUGIN_LIST

PLUGIN_LIST = (
    [
        "SofaPython3",
        "Sofa.GL.Component.Rendering3D",  # <- [OglModel]
        "Sofa.Component.Topology.Container.Grid",  # <- [RegularGridTopology]
    ]
    + SCENE_HEADER_PLUGIN_LIST
    + RIGID_PLUGIN_LIST
    + LOOP_PLUGIN_LIST
    + GRIPPER_PLUGIN_LIST
    + CAVITY_PLUGIN_LIST
)

LENGTH_UNIT = "mm"
TIME_UNIT = "s"
WEIGHT_UNIT = "kg"

HERE = Path(__file__).resolve().parent
INSTRUMENT_MESH_DIR = HERE.parent.parent.parent / "assets/meshes/instruments"
TEXTURE_DIR = HERE.parent.parent.parent / "assets/textures"


def createScene(
    root_node: Sofa.Core.Node,
    debug_rendering: bool = False,
    image_shape: Tuple[Optional[int], Optional[int]] = (None, None),
    animation_loop: AnimationLoopType = AnimationLoopType.FREEMOTION,
    use_beam_adapter_plugin: bool = False,
    with_gripper: bool = False,
    positioning_camera: bool = False,
    stiff_loop: bool = False,
    num_rope_points: int = 50,
    loop_radius: float = 15.0,
    band_width: float = 6.0,
    loop_action_type: ActionType = ActionType.CONTINUOUS,
) -> Dict:
    """
    Creates the scene of the LigatingLoopEnv.

    Args:
        root_node (Sofa.Core.Node): SET BY ENV. The simulation tree's root node.
        debug_rendering (bool): Whether to render more information for debugging purposes.
        image_shape (Tuple[Optional[int], Optional[int]]): SET BY ENV. Height and Width of the rendered images.
        animation_loop (AnimationLoopType): Animation loop of the simulation.
        use_beam_adapter_plugin (bool): Whether to replace the loop components with components from the BeamAdapter Plugin from SOFA.
        with_gripper (bool): SET BY ENV. Whether to include a gripper as a second instrument.
        positioning_camera (bool): Whether to make the camera controllable with keyboard input for manual positioning.
        stiff_loop (bool): Whether to set the mechanical parameters of the loop such that the loop is keeping a round shape under gravity.
        loop_radius (float): Radius of the loop in millimeters.
        band_width (float): SET BY ENV. Width of the marking in millimeters.
        loop_action_type (ActionType): Determines, how actions are applied to the loop.

    Returns:
        scene_creation_result = {
            "camera": camera,
            "loop": ligating_loop,
            "gripper": gripper,
            "cavity": cavity,
            "contact_listeners": {
                "shaft": shaft_contact_listener,
                "loop": loop_contact_listener,
            },
        }
    """

    gravity = -981.0  # mm/s^2

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
            "contactDistance": 0.5,
        },
        collision_detection_method=IntersectionMethod.LOCALMIN,
        contact_friction=0.0,
        constraint_solver=ConstraintSolverType.GENERIC,
        constraint_solver_kwargs={
            "maxIterations": 1000,
            "tolerance": 0.001,
            "computeConstraintForces": False,
            "scaleTolerance": True,
            "multithreading": True,
        },
    )

    cartesian_workspace = {
        "low": np.array([-50, -50, 0]),
        "high": np.array([50, 50, 100]),
    }

    ###################
    # Camera and lights
    ###################
    root_node.addObject(
        "LightManager",
        ambient=(
            0.4,
            0.4,
            0.4,
            0.4,
        ),
    )

    pose = np.array([70, 70, 120, -0.277815948, 0.364971693, 0.115075120, 0.881119560])
    placement_kwargs = {
        "position": pose[:3],
        "orientation": pose[3:],
    }
    placement_kwargs["lookAt"] = determine_look_at(camera_position=placement_kwargs["position"], camera_orientation=placement_kwargs["orientation"])
    light_source_kwargs = {
        "cutoff": 60.0 / 2.0,
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
        z_near=20,
        z_far=220,
        vertical_field_of_view=38.0,
    )

    if positioning_camera:
        root_node.addObject(camera)

    #####################################################
    # Dedicated scene node for the rest of the components
    #####################################################
    scene_node = root_node.addChild("scene")

    cavity = Cavity(
        parent_node=scene_node,
        name="cavity",
        total_mass=0.1,
        young_modulus=2000.0,
        poisson_ratio=0.0,
        inner_radius=8.0,
        outer_radius=10.0,
        height=50.0,
        discretization_radius=2,
        discretization_angle=15,
        discretization_height=25,
        translation=np.array([0.0, 0.0, 0.0]),
        fixed_lenght=5.0,
        texture_path=TEXTURE_DIR / "simple_color_texture.png",
        animation_loop_type=animation_loop,
        collision_group=3,
        band_width=band_width,
    )
    cavity.node.init()
    cavity.color_band(start=30.0, end=34.0)

    ######
    # Loop
    ######
    if use_beam_adapter_plugin:
        mechanical_rope_parameters = {
            "beam_radius": 0.08,
            "radius": 0.3,
            "total_mass": 10.0,
            "poisson_ratio": 0.0,
            "young_modulus": 5e9,
            "mechanical_damping": 0.2,
            "use_beam_adapter_plugin": True,
        }
    else:
        mechanical_rope_parameters = {
            "beam_radius": 0.08,
            "radius": 1.0,
            "total_mass": 10.0,
            "poisson_ratio": 0.0,
            "young_modulus": 5e9 if not stiff_loop else 5e10,
            "mechanical_damping": 0.2,
            "use_beam_adapter_plugin": False,
        }

    ligating_loop = LigatingLoop(
        parent_node=scene_node,
        name="ligating_loop",
        num_rope_points=num_rope_points,
        loop_radius=loop_radius,
        ptsd_state=np.array([0.0, 0.0, 180.0, 30.0]),
        rcm_pose=np.array([-50.0, 0.0, 65.0, 0.0, 90.0, 0.0]),
        total_mass=1e10,
        animation_loop_type=animation_loop,
        show_object=debug_rendering,
        show_object_scale=5.0,
        collision_group=1,
        cartesian_workspace=cartesian_workspace,
        mechanical_rope_parameters=mechanical_rope_parameters,
        mechanical_binding=MechanicalBinding.SPRING,
        spring_stiffness=9e20,
        angular_spring_stiffness=9e20,
        action_type=loop_action_type,
    )

    if not positioning_camera:
        scene_node.addObject(ligating_loop)

    #########
    # Gripper
    #########
    if with_gripper:
        gripper = ArticulatedGripper(
            parent_node=scene_node,
            name="gripper",
            visual_mesh_path_shaft=INSTRUMENT_MESH_DIR / "instrument_shaft.stl",
            visual_mesh_paths_jaws=[INSTRUMENT_MESH_DIR / "forceps_jaw_left.stl", INSTRUMENT_MESH_DIR / "forceps_jaw_right.stl"],
            ptsd_state=np.array([0.0, 0.0, 0.0, 20.0]),
            rcm_pose=np.array([70.0, 0.0, 65.0, 0.0, -90.0, 0.0]),
            collision_spheres_config={
                "positions": [[0, 0, 5 + i * 2] for i in range(10)],
                "radii": [1] * 10,
            },
            angle=0.0,
            angle_limits=(0.0, 60.0),
            total_mass=5e4,
            animation_loop_type=animation_loop,
            show_object=debug_rendering,
            show_object_scale=5.0,
            mechanical_binding=MechanicalBinding.SPRING,
            spring_stiffness=1e19,
            angular_spring_stiffness=1e19,
            articulation_spring_stiffness=1e19,
            collision_group=2,
            cartesian_workspace=cartesian_workspace,
            ptsd_reset_noise=None,
            rcm_reset_noise=None,
            angle_reset_noise=None,
            show_remote_center_of_motion=debug_rendering,
        )
    else:
        gripper = None

    ##################
    # Contact Listener
    ##################
    shaft_contact_listener = scene_node.addObject(
        "ContactListener",
        name="shaft_cavity_listener",
        collisionModel1=cavity.triangle_collision_model.getLinkPath(),
        collisionModel2=ligating_loop.shell_collision_model.getLinkPath(),
    )

    loop_contact_listener = scene_node.addObject(
        "ContactListener",
        name="loop_cavity_listener",
        collisionModel1=cavity.triangle_collision_model.getLinkPath(),
        collisionModel2=ligating_loop.rope.sphere_collision_models.getLinkPath(),
    )

    #############
    # Visual wall
    #############
    wall_node = scene_node.addChild("wall")
    wall_node.addObject("RegularGridTopology", n=[2, 2, 2], min=[-50, -50, -5], max=[50, 50, 0])
    wall_node.addObject("OglModel")

    ###############
    # Returned data
    ###############
    scene_creation_result = {
        "camera": camera,
        "loop": ligating_loop,
        "gripper": gripper,
        "cavity": cavity,
        "contact_listeners": {
            "shaft": shaft_contact_listener,
            "loop": loop_contact_listener,
        },
    }

    return scene_creation_result
