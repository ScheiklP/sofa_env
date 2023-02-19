import Sofa
import Sofa.Core
import numpy as np

from functools import partial
from typing import Optional, Tuple, Union, Dict
from pathlib import Path

from sofa_env.sofa_templates.loader import add_loader, LOADER_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, ConstraintSolverType, add_scene_header, VISUAL_STYLES, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.materials import Material, ConstitutiveModel, MATERIALS_PLUGIN_LIST
from sofa_env.sofa_templates.camera import Camera, CAMERA_PLUGIN_LIST
from sofa_env.sofa_templates.motion_restriction import add_bounding_box, add_fixed_constraint_in_bounding_box, MOTION_RESTRICTION_PLUGIN_LIST
from sofa_env.sofa_templates.solver import ConstraintCorrectionType, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST

from sofa_env.utils.camera import determine_look_at

from sofa_env.scenes.tissue_retraction.sofa_objects import Tissue, EndEffector, TISSUE_PLUGIN_LIST, END_EFFECTOR_PLUGIN_LIST

PLUGIN_LIST = (
    [
        "SofaPython3",
        "Sofa.Component.MechanicalLoad",  # <- [PlaneForceField]
    ]
    + TISSUE_PLUGIN_LIST
    + END_EFFECTOR_PLUGIN_LIST
    + MATERIALS_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
    + CAMERA_PLUGIN_LIST
    + MOTION_RESTRICTION_PLUGIN_LIST
    + LOADER_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
)

LENGTH_UNIT = "m"
TIME_UNIT = "s"
WEIGHT_UNIT = "kg"

HERE = Path(__file__).resolve().parent


def createScene(
    root_node: Sofa.Core.Node,
    image_shape: Tuple[Optional[int], Optional[int]] = (None, None),
    show_bounding_boxes: bool = False,
    randomize_starting_position: bool = True,
    show_fixed_points: bool = False,
    debug_rendering: bool = False,
    texture_objects: bool = False,
    show_floor: bool = True,
    show_realsense: bool = False,
    animation_loop: AnimationLoopType = AnimationLoopType.FREEMOTION,
    remote_center_of_motion: Union[np.ndarray, Tuple] = (0.09036183, 0.15260103, 0.01567807),
    camera_position: Union[np.ndarray, Tuple] = (0.0, 0.135365, 0.233074),
    camera_orientation: Union[np.ndarray, Tuple] = (-0.24634687, -0.0, -0.0, 0.96918173),
    camera_field_of_view_vertical: int = 42,
    camera_placement_kwargs: Optional[dict] = None,
    workspace_height: Optional[float] = None,
    workspace_width: Optional[float] = None,
    workspace_depth: Optional[float] = None,
    workspace_floor: Optional[float] = None,
    add_background_model: bool = True,
) -> Dict:
    """
    Creates the scene of the TissueRetractionEnv

    Args:
        root_node (Sofa.Core.Node): SET BY ENV. The simulation tree's root node.
        image_shape (Tuple[Optional[int], Optional[int]]): SET BY ENV. Height and Width of the rendered images.
        show_bounding_boxes (bool): If True, the bounding boxes of workspace and starting positions are shown.
        randomize_starting_position (bool): If True, the starting position of the end-effector is randomized.
        show_fixed_points (bool): If True, the fixed points of the tissue are shown.
        debug_rendering (bool): Whether to render more information for debugging purposes.
        texture_objects (bool): Whether to add texture rendering the objects.
        show_floor (bool): Whether to render a visual model of the floor.
        show_realsense (bool): Whether to render a visual model of the realsense camera.
        animation_loop (AnimationLoopType): Animation loop of the simulation.
        remote_center_of_motion (Union[np.ndarray, Tuple]): Position of the robot's remote center of motion.
        camera_position (Union[np.ndarray, Tuple]): Position of the camera.
        camera_orientation (Union[np.ndarray, Tuple]): Orientation of the camera as a quaternion.
        camera_field_of_view_vertical (int): Vertical field of view of the camera.
        camera_placement_kwargs (Optional[dict]): Additional keyword arguments for the camera placement.
        workspace_height (Optional[float]): Height of the workspace.
        workspace_width (Optional[float]): Width of the workspace.
        workspace_depth (Optional[float]): Depth of the workspace.
        workspace_floor (Optional[float]): Height of the floor.
        add_background_model (bool): Whether to add a visual model for the background.

    Returns:
        scene_creation_result = {
            "camera": camera,
            "interactive_objects": {
                "end_effector": end_effector,
            },
            "workspace": workspace,
            "tissue_box": tissue_box,
        }
    """

    gravity = -9.81
    tissue_mass = 0.123264

    # Offsets to match the real world setup at ALTAIR.
    # the WORLD coordinate system is placed in the barycenter of the calibration triangle of the setup.
    # Gridboard and tissue are offset in relation to this WORLD coordinate system.
    # The y value of the tissue center is set to 0, since the tissue stl is already offset in y direction.
    tissue_center = (0.0, 0.0, -0.013)
    grid_board_center = (0.0, 0.0, -0.016)
    visual_target_center = (grid_board_center[0] - 0.024, 0.0055, grid_board_center[2] + 0.039)

    ###############################
    # Workspace of the end effector
    ###############################

    ox = grid_board_center[0]
    oz = grid_board_center[2]
    workspace_height = 0.085 if workspace_height is None else workspace_height
    workspace_width = 0.075 if workspace_width is None else workspace_width
    workspace_depth = 0.075 if workspace_depth is None else workspace_depth
    workspace_floor = 0.005 if workspace_floor is None else workspace_floor
    workspace = {
        "low": np.array([-workspace_width + ox, workspace_floor, -workspace_depth + oz] + [0.0] * 3 + [1]),
        "high": np.array([workspace_width + ox, workspace_height, workspace_depth + oz] + [0.0] * 3 + [1]),
    }

    starting_box = {
        "low": np.array([-workspace_width + ox, 0.04, -workspace_depth + oz] + [0.0] * 3 + [1]),
        "high": np.array([workspace_width + ox, workspace_height, workspace_depth + oz] + [0.0] * 3 + [1]),
    }

    ox = tissue_center[0]
    oz = tissue_center[2]
    tissue_box = {
        "low": np.array([-0.06 + ox, 0.005, -0.06 + oz] + [0.0] * 3 + [1]),
        "high": np.array([0.06 + ox, 0.013, 0.06 + oz] + [0.0] * 3 + [1]),
    }

    end_effector_starting_pose = np.array((0.06, 0.06, 0.06) + (0.0, 0.0, 0.0, 1.0))

    ########
    # Meshes
    ########
    tissue_surface_mesh_path = HERE / "meshes/tissue_surface.obj"
    tissue_surface_texture_path = HERE / "meshes/tissue_surface_texture.png" if texture_objects else None
    tissue_volume_mesh_path = HERE / "meshes/tissue_volume.msh"

    floor_surface_mesh_path = HERE / "meshes/floor.obj"
    floor_surface_texture_path = HERE / "meshes/floor_texture.png" if texture_objects else None

    background_surface_mesh_path = HERE / "meshes/background.obj"
    background_surface_texture_path = HERE / "meshes/background_texture.png" if texture_objects else None

    grid_board_surface_mesh_path = HERE / "meshes/grid_board.stl"

    gripper_open_mesh_path = HERE / "meshes/gripper_open.stl"
    gripper_closed_mesh_path = HERE / "meshes/gripper_closed.stl"

    psm_main_link_surface_mesh_path = HERE / "meshes/psm_main_link.stl"

    visual_target_surface_mesh_path = HERE / "meshes/visual_target.stl"

    realsense_surface_mesh = HERE / "meshes/realsense.stl"
    realsense_holder_surface_mesh = HERE / "meshes/realsense_holder.stl"

    ###################
    # Common components
    ###################
    add_scene_header(
        root_node=root_node,
        plugin_list=PLUGIN_LIST,
        visual_style_flags=VISUAL_STYLES["debug"] if debug_rendering else VISUAL_STYLES["normal"],
        gravity=(0.0, gravity, 0.0),
        constraint_solver=ConstraintSolverType.LCP,
        constraint_solver_kwargs={
            "maxIt": 1000,
            "tolerance": 0.001,
        },
        animation_loop=animation_loop,
        scene_has_collisions=False,
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

    root_node.addObject("PositionalLight", position=camera_position)

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
        }
        if camera_placement_kwargs is None
        else camera_placement_kwargs,
        z_near=0.0001,
        z_far=1,
        width_viewport=image_shape[0],
        height_viewport=image_shape[1],
        vertical_field_of_view=camera_field_of_view_vertical,
    )

    #####################################################
    # Dedicated scene node for the rest of the components
    #####################################################
    scene_node = root_node.addChild("scene")

    ########
    # Tissue
    ########
    # All Sofa.Core.Controller objects have to be added to the scene graph to correctly receive events
    if texture_objects:
        add_visual_tissue_model = partial(add_visual_model, texture_file_path=str(tissue_surface_texture_path))
    else:
        add_visual_tissue_model = partial(add_visual_model, color=(230 / 255, 220 / 255, 70 / 255))

    # Tissue parameters from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3966138/ without thinner
    tissue = scene_node.addObject(
        Tissue(
            animation_loop_type=animation_loop,
            constraint_correction_type=ConstraintCorrectionType.PRECOMPUTED,
            parent_node=scene_node,
            volume_mesh_path=tissue_volume_mesh_path,
            total_mass=tissue_mass,
            visual_mesh_path=tissue_surface_mesh_path,
            translation=tissue_center,
            material=Material(
                constitutive_model=ConstitutiveModel.COROTATED,
                poisson_ratio=0.4287,
                young_modulus=27040,
            ),
            add_visual_model_func=add_visual_tissue_model,
            show_object=False,
        )
    )

    # Fix the top right corner of the tissue
    tissue.deformable_object.fix_indices_in_bounding_box(
        min=(-0.0068, 0.004, -0.0663),
        max=(0.0532, 0.013, -0.0263),
        fixture_func=add_fixed_constraint_in_bounding_box,
    )

    if show_fixed_points:
        # Initialize the node, so we can read the indices from the BoxROI. Before that, the mechanical object has nothing loaded.
        tissue.deformable_object.node.init()
        tissue.deformable_object.node.BoxROI.drawPoints = True
        tissue.deformable_object.node.BoxROI.drawSize = 8

    #################################
    # Visual target on the grid board
    #################################
    visual_target_node = scene_node.addChild("visual_target")
    loader_object = add_loader(
        attached_to=visual_target_node,
        file_path=visual_target_surface_mesh_path,
        loader_kwargs={"triangulate": True, "translation": visual_target_center},
    )
    ogl_kwargs = {}
    ogl_kwargs["color"] = (150 / 255, 0 / 255, 150 / 255)
    visual_target_node.addObject(
        "OglModel",
        src=loader_object.getLinkPath(),
        **ogl_kwargs,
    )

    ##############
    # End Effector
    ##############
    # all Sofa.Core.Controller objects have to be added to the scene graph to correctly receive events
    end_effector = scene_node.addObject(
        EndEffector(
            parent_node=scene_node,
            animation_loop_type=animation_loop,
            name="end_effector",
            pose=end_effector_starting_pose,
            graspable_tissue=tissue,
            number_of_grasping_points=1,
            grasping_distance=0.009,
            remote_center_of_motion=remote_center_of_motion,
            workspace=workspace,
            randomize_starting_position=randomize_starting_position,
            starting_box=starting_box,
            show_object=False,
            visual_mesh_path_gripper_open=gripper_open_mesh_path,
            visual_mesh_path_gripper_closed=gripper_closed_mesh_path,
            visual_mesh_path_main_link=psm_main_link_surface_mesh_path,
        )
    )

    #######
    # Floor
    #######
    tissue.deformable_object.node.addObject(
        "PlaneForceField",
        normal=[0, 1, 0],
        d=0.0045,
        stiffness=30,
        damping=1,
        showPlane=True,
        showPlaneSize=0.1,
    )

    floor_node = scene_node.addChild("floor")
    loader_object = add_loader(
        attached_to=floor_node,
        file_path=floor_surface_mesh_path,
        loader_kwargs={"triangulate": True},
    )
    ogl_kwargs = {}

    if texture_objects:
        ogl_kwargs["texturename"] = str(floor_surface_texture_path)
    else:
        ogl_kwargs["color"] = (152 / 255, 189 / 255, 207 / 255)

    if show_floor:
        floor_node.addObject(
            "OglModel",
            src=loader_object.getLinkPath(),
            **ogl_kwargs,
        )

    ############
    # Background
    ############

    if add_background_model:
        background_node = scene_node.addChild("background")
        loader_object = add_loader(
            attached_to=background_node,
            file_path=background_surface_mesh_path,
            loader_kwargs={
                "triangulate": True,
                "translation": [0, 0, -0.38],
                "rotation": [90, 0, 0],
            },
        )
        ogl_kwargs = {}
        if texture_objects:
            ogl_kwargs["texturename"] = str(background_surface_texture_path)
        else:
            ogl_kwargs["color"] = (0, 0, 0)

        background_node.addObject(
            "OglModel",
            src=loader_object.getLinkPath(),
            **ogl_kwargs,
        )

    ############
    # Grid board
    ############
    grid_board_node = scene_node.addChild("grid_board")
    loader_object = add_loader(
        attached_to=grid_board_node,
        file_path=grid_board_surface_mesh_path,
        loader_kwargs={"triangulate": True, "translation": grid_board_center},
    )
    ogl_kwargs = {}
    if show_floor:
        ogl_kwargs["color"] = (150 / 255, 150 / 255, 150 / 255)
    else:
        ogl_kwargs["color"] = (0 / 255, 100 / 255, 0 / 255)

    grid_board_node.addObject(
        "OglModel",
        src=loader_object.getLinkPath(),
        **ogl_kwargs,
    )

    ##############
    # camera model
    ##############
    if show_realsense:
        realsense_node = scene_node.addChild("realsense")
        loader_object = add_loader(
            attached_to=realsense_node,
            file_path=realsense_surface_mesh,
            loader_kwargs={"triangulate": True, "translation": [0, 0.005, 0.0]},
            scale=0.001,
        )
        ogl_kwargs = {}
        ogl_kwargs["color"] = (0 / 255, 255 / 255, 0 / 255)

        realsense_node.addObject(
            "OglModel",
            src=loader_object.getLinkPath(),
            **ogl_kwargs,
        )

        holder_node = scene_node.addChild("realsense_holder")
        loader_object = add_loader(
            attached_to=holder_node,
            file_path=realsense_holder_surface_mesh,
            loader_kwargs={"triangulate": True, "translation": [0, 0.005, 0.0]},
            scale=0.001,
        )
        ogl_kwargs = {}
        ogl_kwargs["color"] = (30 / 255, 30 / 255, 255 / 255)

        holder_node.addObject(
            "OglModel",
            src=loader_object.getLinkPath(),
            **ogl_kwargs,
        )

    ########################
    # Showing bounding boxes
    ########################
    if show_bounding_boxes:
        scene_node.addObject("MechanicalObject", template="Rigid3d", position=[0, 0, 0, 0, 0, 0, 1])
        add_bounding_box(scene_node, min=workspace["low"][:3], max=workspace["high"][:3], show_bounding_box=True)
        add_bounding_box(scene_node, min=starting_box["low"][:3], max=starting_box["high"][:3], show_bounding_box=True)
        add_bounding_box(scene_node, min=tissue_box["low"][:3], max=tissue_box["high"][:3], show_bounding_box=True)

    ###############
    # Returned data
    ###############
    scene_creation_result = {
        "camera": camera,
        "interactive_objects": {
            "end_effector": end_effector,
        },
        "workspace": workspace,
        "tissue_box": tissue_box,
    }

    return scene_creation_result
