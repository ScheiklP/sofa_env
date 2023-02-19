import numpy as np

from functools import partial
from pathlib import Path
from typing import Optional, Tuple, List
from enum import Enum, unique

import Sofa
import Sofa.Core

from sofa_env.sofa_templates.camera import PivotizedCamera, CAMERA_PLUGIN_LIST
from sofa_env.sofa_templates.rigid import PivotizedArticulatedInstrument, RigidObject, RIGID_PLUGIN_LIST
from sofa_env.scenes.grasp_lift_touch.sofa_objects.cauter import Cauter
from sofa_env.sofa_templates.scene_header import AnimationLoopType, add_scene_header, SCENE_HEADER_PLUGIN_LIST, VISUAL_STYLES
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST

from sofa_env.scenes.search_for_point.sofa_objects.point_of_interest import PointOfInterest, POI_PLUGIN_LIST

HERE: Path = Path(__file__).resolve().parent
MESHES_PATH: Path = HERE.parent.parent.parent / "assets" / "meshes"
ENDOSCOPES_MESH_DIR: Path = MESHES_PATH / "endoscopes"
INSTRUMENT_MESH_DIR: Path = MESHES_PATH / "instruments"
OPEN_HELP_DIR: Path = MESHES_PATH / "models" / "OpenHELP"

POI_CAMERA_VISIBLE = HERE / "meshes" / "pois_for_camera.stl"
POI_CAMERA_VISIBLE_WITHOUT_ABDOMINAL_WALL = HERE / "meshes" / "pois_for_camera_without_abdominal_wall.stl"
POI_CAUTER_REACHABLE = HERE / "meshes" / "pois_for_cauter.npy"

COLLISON_MODEL = HERE / "meshes" / "collision_model_cauter.stl"
COLLISON_MODEL_WITHOUT_ABDOMINAL_WALL = HERE / "meshes" / "collision_model_cauter_without_abdominal_wall.stl"


@unique
class POIMode(Enum):
    """POIMode specifies in which area the POI can be"""

    CAMERA_VISIBLE = POI_CAMERA_VISIBLE
    CAMERA_VISIBLE_WITHOUT_ABDOMINAL_WALL = POI_CAMERA_VISIBLE_WITHOUT_ABDOMINAL_WALL
    CAUTER_REACHABLE = POI_CAUTER_REACHABLE


PLUGIN_LIST: List[str] = (
    [
        "SofaPython3",
    ]
    + SCENE_HEADER_PLUGIN_LIST
    + CAMERA_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
    + RIGID_PLUGIN_LIST
    + POI_PLUGIN_LIST
)

COLORS = dict(
    Liver=(144, 53, 55),
    Rectum_Lumen=(233, 137, 124),
    Bladder_half=(236, 77, 13),
    Trachea=(253, 217, 175),
    Spleen=(133, 53, 67),
    Kidney_right=(112, 41, 35),
    Kidney_left=(112, 41, 35),
    Heart=(198, 50, 50),
    Pelvis_Muscles=(148, 73, 57),
    Lung_right=(226, 109, 109),
    Lung_left=(226, 109, 109),
    Stomach=(254, 116, 92),
    Duodenum=(170, 89, 103),
    Pancreas=(241, 179, 107),
    Torso_Complete=(255, 222, 205),
    Abdominal_wall=(255, 222, 205),
    Camera=(202, 203, 207),
)
for organ, color in COLORS.items():
    COLORS[organ] = tuple(i / 255.0 for i in color)


def createScene(
    root_node: Sofa.Core.Node,
    image_shape: Tuple[Optional[int], Optional[int]] = (None, None),
    debug_rendering: bool = False,
    animation_loop: AnimationLoopType = AnimationLoopType.DEFAULT,
    without_abdominal_wall: bool = False,
    transparent_abdominal_wall: bool = True,
    use_spotlight: bool = True,
    render_poi_options: bool = False,
    check_collision: bool = False,
    hide_surgeon_gripper: bool = False,
    hide_assistant_gripper: bool = False,
    hide_cauter: bool = False,
    poi_mode: POIMode = POIMode.CAMERA_VISIBLE_WITHOUT_ABDOMINAL_WALL,
    extend_dict_for_observation_wrappers: bool = False,
    place_camera_outside: bool = False,
):
    """Creates the scene of the SearchForPointEnv.

    The abdominal cavity is modelled with the surface files of the OpenHELP phantom.

    Args:
        root_node (Sofa.Core.Node): SET BY ENV. The simulation tree's root node.
        image_shape (Tuple[Optional[int], Optional[int]]): SET BY ENV. Height and Width of the rendered images.
        debug_rendering (bool): Whether to render more information for debugging purposes.
        animation_loop (AnimationLoopType): Animation loop of the simulation.
        without_abdominal_wall (bool): Whether to hide the abdominal wall.
        transparent_abdominal_wall (bool): Whether to make the abdominal wall transparent.
        use_spotlight (bool): Whether to add a spotlight to the endoscope.
        render_poi_options (bool): Whether to render the parts of the surface that are valid for sampling point of interest positions.
        check_collision (bool): Whether to check for collisions to detect instrument collisions in the active vision case.
        hide_surgeon_gripper (bool): Whether to hide the surgeon's gripper.
        hide_assistant_gripper (bool): Whether to hide the assistant's gripper.
        hide_cauter (bool): Whether to hide the cauter.
        poi_mode (POIMode): In which area the point of interest can be.
        extend_dict_for_observation_wrappers (bool): Whether to add additional information to the returned dictionary for ``PointCloudObservationWrapper`` and ``SemanticSegmentationWrapper``.
        place_endoscope_outside (bool): If ``True``, a static endoscope outside of the abdomen is used.

    Returns:
        sofa_objects = {
            "endoscope": endoscope,
            "assistant_gripper": assistant_gripper,
            "surgeon_gripper": surgeon_gripper,
            "cauter": cauter,
            "poi": poi,
            "contact_listener": {}, <- has key "cauter" for active vision
            **open_help_objects,
        }
    """

    # In case you want to change the trocar positions, you might need to change the collision model as well
    trocar_positions = dict(
        assistant_gripper=np.array([-96.35, 5.73, 229.6]),
        surgeon_gripper=np.array([-96.27, -84.37, 211.83]),
        cauter=np.array([117.2, 2.29, 233.62]),
        endoscope=np.array([10.0, -100.8, 281.82]),
    )

    trocar_rotations = dict(
        assistant_gripper=np.array([180.0, 30.0, 100.0]),
        surgeon_gripper=np.array([-150.0, 30.0, 100.0]),
        cauter=np.array([-170.0, -30.0, 0.0]),
        endoscope=np.array([-155.0, 0.0, 0.0]),
    )

    state_limits = dict(
        assistant_gripper=dict(
            low=np.array([-90.0, -90.0, -90.0, -100.0]),
            high=np.array([90.0, 90.0, 90.0, 200.0]),
        ),
        surgeon_gripper=dict(
            low=np.array([-90.0, -90.0, -90.0, -100.0]),
            high=np.array([90.0, 90.0, 90.0, 200.0]),
        ),
        cauter=dict(
            low=np.array([-60.0, -60.0, -90.0, 0.0]),
            high=np.array([35.0, 60.0, 90.0, 250.0]),
        ),
        endoscope=dict(
            low=np.array([-40.0, -50.0, -180.0, 10.0]),
            high=np.array([40.0, 40.0, 180.0, 150.0]),
        ),
    )

    initial_states = dict(
        assistant_gripper=np.array([0.0, 0.0, 0.0, 50.0]),
        surgeon_gripper=np.array([0.0, 0.0, 0.0, 30.0]),
        cauter=np.array([0.0, 0.0, 0.0, 50.0]),
        endoscope=np.array([0.0, 0.0, 0.0, 30.0]),
    )

    state_reset_noise = dict(
        assistant_gripper=np.array([10.0, 10.0, 90.0, 50.0]),
        surgeon_gripper=np.array([10.0, 10.0, 90.0, 50.0]),
        cauter=np.array([15.0, 15.0, 30.0, 20.0]),
        endoscope=np.array([15.0, 15.0, 15.0, 15.0]),
    )

    ###################
    # Common components
    ###################
    add_scene_header(
        root_node=root_node,
        plugin_list=PLUGIN_LIST,
        visual_style_flags=VISUAL_STYLES["debug"] if debug_rendering else VISUAL_STYLES["normal"],
        animation_loop=animation_loop,
        scene_has_collisions=True if check_collision else False,
        gravity=(0.0, 0.0, -981.0),
    )

    ###################
    # Lights
    ###################
    root_node.addObject("LightManager", ambient=(0.3, 0.3, 0.3, 0.3))
    if debug_rendering:
        root_node.addObject("DirectionalLight", direction=(2, 1, -1), color=(0.8, 0.8, 0.8, 1.0))
        root_node.addObject("DirectionalLight", direction=(0, -1, 0), color=(0.8, 0.8, 0.8, 1.0))
    if use_spotlight:
        root_node.addObject("OglShadowShader")

    #################
    # SOFA components
    #################

    endoscope_visual_function = partial(add_visual_model, color=COLORS["Camera"])
    camera_kwargs = dict(
        root_node=root_node,
        animation_loop_type=animation_loop,
        visual_mesh_path=ENDOSCOPES_MESH_DIR / "laparoscope_optics.stl",
        add_visual_model_func=endoscope_visual_function,
        show_object=debug_rendering,
        show_object_scale=10.0,
        show_remote_center_of_motion=debug_rendering,
        rcm_pose=np.concatenate([trocar_positions["endoscope"], trocar_rotations["endoscope"]]),
        width_viewport=image_shape[0],
        height_viewport=image_shape[1],
        z_near=2.0,
    )
    light_source_kwargs = dict(
        color=[1.0, 1.0, 1.0, 1.3],
        cutoff=60.0,
        attenuation=0.0,
        exponent=1.0,
        shadowsEnabled=use_spotlight,
    )

    if place_camera_outside:
        endoscope = PivotizedCamera(
            ptsd_state=np.array([0.0, 0.0, 0.0, -500.0]),
            z_far=1000.0,
            **camera_kwargs,
        )
    else:
        endoscope = PivotizedCamera(
            ptsd_state=initial_states["endoscope"],
            state_limits=state_limits["endoscope"],
            z_far=300.0,
            vertical_field_of_view=45.0,
            ptsd_reset_noise=state_reset_noise["endoscope"],
            rcm_reset_noise=None,
            with_light_source=use_spotlight,
            light_source_kwargs=light_source_kwargs,
            oblique_viewing_angle=30.0,
            **camera_kwargs,
        )

    scene_node = root_node.addChild("scene")

    if not hide_assistant_gripper:
        assistant_gripper = PivotizedArticulatedInstrument(
            parent_node=scene_node,
            name="assistant_gripper",
            visual_mesh_path_shaft=None if hide_assistant_gripper else INSTRUMENT_MESH_DIR / "instrument_shaft.stl",
            visual_mesh_paths_jaws=None if hide_assistant_gripper else [INSTRUMENT_MESH_DIR / "forceps_jaw_left.stl", INSTRUMENT_MESH_DIR / "forceps_jaw_right.stl"],
            animation_loop_type=animation_loop,
            add_visual_model_func=partial(add_visual_model, color=(1.0, 1.0, 0.0)),
            show_object=debug_rendering,
            show_object_scale=10.0,
            show_remote_center_of_motion=debug_rendering,
            ptsd_state=initial_states["assistant_gripper"],
            rcm_pose=np.concatenate([trocar_positions["assistant_gripper"], trocar_rotations["assistant_gripper"]]),
            state_limits=state_limits["assistant_gripper"],
            angle_limits=(0.0, 60.0),
            ptsd_reset_noise=state_reset_noise["assistant_gripper"],
            rcm_reset_noise=None,
            angle_reset_noise=None,
        )
    else:
        assistant_gripper = None

    if not hide_surgeon_gripper:
        surgeon_gripper = PivotizedArticulatedInstrument(
            parent_node=scene_node,
            name="surgeon_gripper",
            visual_mesh_path_shaft=None if hide_surgeon_gripper else INSTRUMENT_MESH_DIR / "instrument_shaft.stl",
            visual_mesh_paths_jaws=None if hide_surgeon_gripper else [INSTRUMENT_MESH_DIR / "forceps_jaw_left.stl", INSTRUMENT_MESH_DIR / "forceps_jaw_right.stl"],
            animation_loop_type=animation_loop,
            add_visual_model_func=partial(add_visual_model, color=(1.0, 0.0, 0.0)),
            show_object=debug_rendering,
            show_object_scale=10.0,
            show_remote_center_of_motion=debug_rendering,
            ptsd_state=initial_states["surgeon_gripper"],
            rcm_pose=np.concatenate([trocar_positions["surgeon_gripper"], trocar_rotations["surgeon_gripper"]]),
            state_limits=state_limits["surgeon_gripper"],
            angle_limits=(0.0, 60.0),
            ptsd_reset_noise=state_reset_noise["surgeon_gripper"],
            rcm_reset_noise=None,
            angle_reset_noise=None,
        )
    else:
        surgeon_gripper = None

    poi = PointOfInterest(
        parent_node=scene_node,
        name="PointOfInterest",
        visual_mesh_path=MESHES_PATH / "models" / "unit_sphere.stl",
        poi_mesh_path=poi_mode.value,
        radius=5.0,
    )

    if not hide_cauter:
        cauter = scene_node.addObject(
            Cauter(
                parent_node=scene_node,
                name="cauter",
                visual_mesh_path=None if hide_cauter else INSTRUMENT_MESH_DIR / "dissection_electrode.stl",
                add_visual_model_func=partial(add_visual_model, color=(0.0, 0.0, 1.0)),
                ptsd_state=initial_states["cauter"],
                rcm_pose=np.concatenate([trocar_positions["cauter"], trocar_rotations["cauter"]]),
                show_object=debug_rendering,
                show_object_scale=10.0,
                state_limits=state_limits["cauter"],
                poi_to_touch=poi,
                ptsd_reset_noise=state_reset_noise["cauter"],
                rcm_reset_noise=None,
                animation_loop_type=animation_loop,
            )
        )
    else:
        cauter = None

    # Add open help object to the scene
    open_help_objects = {}
    open_help_files = [file_path for file_path in OPEN_HELP_DIR.iterdir() if file_path.suffix == ".stl"]
    abdominal_wall_mesh_path = None
    for file in open_help_files:
        objectname = file.stem
        if objectname == "Abdominal_wall":
            abdominal_wall_mesh_path = file
        else:
            objectcolor = COLORS[objectname] if objectname in COLORS else (1.0, 0.0, 0.0)
            open_help_objects[objectname] = RigidObject(
                parent_node=scene_node,
                name=objectname,
                visual_mesh_path=file,
                add_visual_model_func=partial(add_visual_model, color=objectcolor),
                add_solver_func=lambda attached_to: (None, None),
                pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            )

    if not without_abdominal_wall:
        if abdominal_wall_mesh_path is None:
            raise FileNotFoundError("Abdominal wall mesh not found.")

        abdominal_wall = RigidObject(
            parent_node=scene_node,
            name="Abdominal_wall",
            visual_mesh_path=abdominal_wall_mesh_path,
            add_visual_model_func=partial(add_visual_model, color=COLORS["Abdominal_wall"], transparency=0.6 if transparent_abdominal_wall else 0.0),
            add_solver_func=lambda attached_to: (None, None),
            pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
        open_help_objects["Abdominal_wall"] = abdominal_wall

    if render_poi_options:
        if poi_mode == POIMode.CAUTER_REACHABLE:
            reachable_node = scene_node.addChild("reachable")
            reachable_node.addObject(
                "MechanicalObject",
                template="Vec3d",
                position=np.asarray(poi.possible_target_poses)[:, :3],
                showObject=True,
                showObjectScale=6.0,
                showColor=[1.0, 0.0, 0.0],
            )

        else:
            RigidObject(
                parent_node=root_node,
                name="visible",
                visual_mesh_path=poi_mode.value,
                pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                add_visual_model_func=partial(add_visual_model, color=(1.0, 0.0, 0.0)),
            )

    sofa_objects = {
        "camera": endoscope,
        "assistant_gripper": assistant_gripper,
        "surgeon_gripper": surgeon_gripper,
        "cauter": cauter,
        "poi": poi,
        "contact_listener": {},
        **open_help_objects,
    }

    if check_collision and cauter is not None:
        openhelp_collision_model = RigidObject(
            parent_node=scene_node,
            name="openhelp_collision",
            pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            collision_mesh_path=COLLISON_MODEL_WITHOUT_ABDOMINAL_WALL if without_abdominal_wall else COLLISON_MODEL,
        )
        cauter_collision_model = cauter.sphere_collision_model.getLinkPath()
        surface_collision_model = openhelp_collision_model.collision_model_node.TriangleCollisionModel.getLinkPath()
        cauter_listener = scene_node.addObject(
            "ContactListener",
            name="ContactListenerCauter",
            collisionModel1=cauter_collision_model,
            collisionModel2=surface_collision_model,
        )
        sofa_objects["contact_listener"]["cauter"] = cauter_listener

    if extend_dict_for_observation_wrappers:
        sofa_objects["pointcloud_objects"] = {}
        sofa_objects["pointcloud_objects"]["position_containers"] = []
        sofa_objects["pointcloud_objects"]["triangle_containers"] = []
        if cauter is not None:
            sofa_objects["pointcloud_objects"]["position_containers"].append(cauter.visual_model_node.OglModel)
            sofa_objects["pointcloud_objects"]["triangle_containers"].append(cauter.visual_model_node.OglModel)
        if assistant_gripper is not None:
            sofa_objects["pointcloud_objects"]["position_containers"].append(assistant_gripper.shaft_visual_model_node.OglModel)
            sofa_objects["pointcloud_objects"]["triangle_containers"].append(assistant_gripper.shaft_visual_model_node.OglModel)
            sofa_objects["pointcloud_objects"]["position_containers"].append(assistant_gripper.jaw_visual_model_nodes[0].OglModel)
            sofa_objects["pointcloud_objects"]["triangle_containers"].append(assistant_gripper.jaw_visual_model_nodes[0].OglModel)
            sofa_objects["pointcloud_objects"]["position_containers"].append(assistant_gripper.jaw_visual_model_nodes[1].OglModel)
            sofa_objects["pointcloud_objects"]["triangle_containers"].append(assistant_gripper.jaw_visual_model_nodes[1].OglModel)
        if surgeon_gripper is not None:
            sofa_objects["pointcloud_objects"]["position_containers"].append(surgeon_gripper.shaft_visual_model_node.OglModel)
            sofa_objects["pointcloud_objects"]["triangle_containers"].append(surgeon_gripper.shaft_visual_model_node.OglModel)
            sofa_objects["pointcloud_objects"]["position_containers"].append(surgeon_gripper.jaw_visual_model_nodes[0].OglModel)
            sofa_objects["pointcloud_objects"]["triangle_containers"].append(surgeon_gripper.jaw_visual_model_nodes[0].OglModel)
            sofa_objects["pointcloud_objects"]["position_containers"].append(surgeon_gripper.jaw_visual_model_nodes[1].OglModel)
            sofa_objects["pointcloud_objects"]["triangle_containers"].append(surgeon_gripper.jaw_visual_model_nodes[1].OglModel)

        for obj in open_help_objects:
            sofa_objects["pointcloud_objects"]["position_containers"].append(open_help_objects[obj].visual_model_node.OglModel)
            sofa_objects["pointcloud_objects"]["triangle_containers"].append(open_help_objects[obj].visual_model_node.OglModel)

        sofa_objects["semantic_segmentation_objects"] = sofa_objects["pointcloud_objects"]

    return sofa_objects
