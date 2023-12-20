from functools import partial
import numpy as np

from pathlib import Path
from typing import Optional, Tuple, Dict

import Sofa.Core

from sofa_env.sofa_templates.motion_restriction import add_bounding_box, MOTION_RESTRICTION_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import VISUAL_STYLES, AnimationLoopType, ConstraintSolverType, IntersectionMethod, add_scene_header, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.camera import Camera, CAMERA_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST

from sofa_env.utils.camera import determine_look_at

from sofa_env.scenes.grasp_lift_touch.sofa_objects.point_of_interest import PointOfInterest, POI_PLUGIN_LIST
from sofa_env.scenes.grasp_lift_touch.sofa_objects.cauter import Cauter, CAUTER_PLUGIN_LIST
from sofa_env.scenes.grasp_lift_touch.sofa_objects.gallbladder import Gallbladder, GALLBLADDER_PLUGIN_LIST
from sofa_env.scenes.grasp_lift_touch.sofa_objects.gripper import Gripper, GRIPPER_PLUGIN_LIST
from sofa_env.scenes.grasp_lift_touch.sofa_objects.liver import Liver, LIVER_PLUGIN_LIST
from sofa_env.scenes.grasp_lift_touch.sofa_objects.tool_controller import ToolController


HERE = Path(__file__).resolve().parent
INSTRUMENT_MESH_DIR = HERE.parent.parent.parent / "assets/meshes/instruments"
MODEL_MESH_DIR = HERE.parent.parent.parent / "assets/meshes/models"

PLUGIN_LIST = (
    [
        "SofaPython3",
        "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.Topology.Container.Dynamic",
    ]
    + SCENE_HEADER_PLUGIN_LIST
    + LIVER_PLUGIN_LIST
    + GALLBLADDER_PLUGIN_LIST
    + GRIPPER_PLUGIN_LIST
    + CAUTER_PLUGIN_LIST
    + POI_PLUGIN_LIST
    + CAMERA_PLUGIN_LIST
    + MOTION_RESTRICTION_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
)

GRAVITY = 9.810


def createScene(
    root_node: Sofa.Core.Node,
    image_shape: Tuple[Optional[int], Optional[int]] = (None, None),
    show_everything: bool = False,
    animation_loop: AnimationLoopType = AnimationLoopType.DEFAULT,
    randomize_poi_position: bool = True,
) -> Dict:

    """
    Creates the scene of the GraspLiftTouchEnv.

    Args:
        root_node (Sofa.Core.Node): SET BY ENV. The simulation tree's root node.
        image_shape (Tuple[Optional[int], Optional[int]]): SET BY ENV. Height and Width of the rendered images.
        show_everything (bool): Render additional information for debugging.
        animation_loop (AnimationLoopType): Animation loop of the simulation.
        randomize_poi_position (bool): Randomize the position of the point of interest.

    Returns:
        scene_creation_result = {
            "interactive_objects": {"gripper": gripper, "cauter": cauter},
            "contact_listener": {
                "cauter_gallbladder": cauter_gallbladder_listener,
                "cauter_liver": cauter_liver_listener,
                "gripper_liver": gripper_liver_listener,
            },
            "poi": poi,
            "camera": camera,
            "liver": liver,
            "gallbladder": gallbladder,
        }
    """

    add_scene_header(
        root_node=root_node,
        plugin_list=PLUGIN_LIST,
        visual_style_flags=VISUAL_STYLES["debug"] if show_everything else VISUAL_STYLES["normal"],
        gravity=(0.0, GRAVITY, 0.0),
        constraint_solver=ConstraintSolverType.GENERIC,
        constraint_solver_kwargs={
            "maxIt": 1000,
            "tolerance": 0.001,
        },
        animation_loop=animation_loop,
        scene_has_collisions=True,
        collision_detection_method_kwargs={
            "alarmDistance": 5.0,
            "contactDistance": 0.5,
        },
        collision_detection_method=IntersectionMethod.LOCALMIN,
    )

    root_node.addObject("LightManager")
    root_node.addObject("OglShadowShader")

    camera = Camera(
        root_node,
        {
            "position": np.array([-94.5, -113.6, -237.2]),
            "orientation": np.array([0.683409, -0.684564, 0.0405591, 0.25036]),
            "lookAt": determine_look_at(np.array([-94.5, -113.6, -237.2]), np.array([0.683409, -0.684564, 0.0405591, 0.25036])),
        },
        z_near=100,
        z_far=500,
        width_viewport=image_shape[0],
        height_viewport=image_shape[1],
    )

    root_node.addObject(
        "DirectionalLight",
        direction=[94.5, 113.6, 237.2],
    )

    scene_node = root_node.addChild("scene")

    cartesian_instrument_workspace = {
        "low": np.array((-200, -200, -200)),
        "high": np.array((130, 130, 130)),
    }
    for array in cartesian_instrument_workspace.values():
        array.flags.writeable = False

    if show_everything:
        scene_node.addObject("MechanicalObject")
        add_bounding_box(
            scene_node,
            min=cartesian_instrument_workspace["low"],
            max=cartesian_instrument_workspace["high"],
            show_bounding_box=True,
            show_bounding_box_scale=4,
        )

    liver = Liver(
        parent_node=scene_node,
        volume_mesh_path=HERE / "meshes/liver_volumetric.vtk",
        visual_mesh_path=HERE / "meshes/liver_visual.stl",
        collision_mesh_path=HERE / "meshes/liver_collision.stl",
        animation_loop=animation_loop,
    )

    gallbladder = Gallbladder(
        parent_node=scene_node,
        liver_mesh_path=HERE / "meshes/liver_collision.stl",
        volume_mesh_path=HERE / "meshes/gallbladder_volumetric.vtk",
        collision_mesh_path=HERE / "meshes/gallbladder_collision.stl",
        visual_mesh_path=HERE / "meshes/gallbladder_visual.stl",
        animation_loop=animation_loop,
    )

    poi = PointOfInterest(
        parent_node=scene_node,
        name="PointOfInterest",
        visual_mesh_path=MODEL_MESH_DIR / "unit_sphere.stl",
        scale=4.5,
        randomized_pose=randomize_poi_position,
    )

    gripper = Gripper(
        parent_node=scene_node,
        name="Gripper",
        visual_mesh_path_shaft=INSTRUMENT_MESH_DIR / "single_action_forceps_shaft.stl",
        visual_mesh_path_jaw=INSTRUMENT_MESH_DIR / "single_action_forceps_jaw.stl",
        gallbladder_to_grasp=gallbladder,
        ptsd_state=np.array([-15.0, 30.0, -20.0, 130.0]),
        angle=45,
        collision_spheres_config={
            "positions": [[0, 0, 5 + i * 2] for i in range(10)],
            "radii": [1] * 10,
        },
        rcm_pose=np.array([-160.0, 30.0, -200.0, 0.0, 20.0, 45.0]),
        animation_loop_type=animation_loop,
        cartesian_workspace=cartesian_instrument_workspace,
        state_limits={
            "low": np.array([-75.0, -40.0, -1000.0, 12.0]),
            "high": np.array([75.0, 75.0, 1000.0, 300.0]),
        },
        total_mass=1e5,
        ptsd_reset_noise=np.array([10.0, 10.0, 45.0, 10.0]),
        angle_reset_noise=20.0,
    )

    scene_node.addObject(gripper)

    cauter_visual_model_func = partial(add_visual_model, color=(0.0, 1.0, 0.0))
    cauter = Cauter(
        name="Cauter",
        parent_node=scene_node,
        visual_mesh_path=INSTRUMENT_MESH_DIR / "dissection_electrode.stl",
        add_visual_model_func=cauter_visual_model_func,
        poi_to_touch=poi,
        ptsd_state=np.array([-15.0, 45.0, 0.0, 150.0]),
        rcm_pose=np.array([-150.0, -200.0, 0.0, 0.0, 90.0, 180.0]),
        animation_loop_type=animation_loop,
        cartesian_workspace=cartesian_instrument_workspace,
        state_limits={
            "low": np.array([-75.0, -40.0, -1000.0, 12.0]),
            "high": np.array([75.0, 75.0, 1000.0, 300.0]),
        },
        total_mass=1e5,
        ptsd_reset_noise=np.array([10.0, 10.0, 45.0, 10.0]),
    )

    scene_node.addObject(cauter)

    scene_node.addObject(ToolController(gripper=gripper, cauter=cauter))

    cauter_collision_model = cauter.sphere_collision_model.getLinkPath()
    gripper_collision_model = gripper.physical_shaft_node.approximated_collision_shaft_jaw.SphereCollisionModel.getLinkPath()
    gallbladder_collision_model = gallbladder.collision_model_node.TriangleCollisionModel.getLinkPath()
    liver_collision_model = liver.collision_model_node.TriangleCollisionModel.getLinkPath()

    cauter_gallbladder_listener = scene_node.addObject(
        "ContactListener",
        name="ContactListenerCauterGallbladder",
        collisionModel1=cauter_collision_model,
        collisionModel2=gallbladder_collision_model,
    )

    cauter_liver_listener = scene_node.addObject(
        "ContactListener",
        name="ContactListenerCauterLiver",
        collisionModel1=cauter_collision_model,
        collisionModel2=liver_collision_model,
    )

    gripper_liver_listener = scene_node.addObject(
        "ContactListener",
        name="ContactListenerGripperLiver",
        collisionModel1=gripper_collision_model,
        collisionModel2=liver_collision_model,
    )

    return {
        "interactive_objects": {"gripper": gripper, "cauter": cauter},
        "contact_listener": {
            "cauter_gallbladder": cauter_gallbladder_listener,
            "cauter_liver": cauter_liver_listener,
            "gripper_liver": gripper_liver_listener,
        },
        "poi": poi,
        "camera": camera,
        "liver": liver,
        "gallbladder": gallbladder,
    }
