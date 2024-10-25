import Sofa
import Sofa.Core
import numpy as np

from pathlib import Path
from functools import partial
from typing import Optional, Tuple, Union, Dict

from sofa_env.sofa_templates.mappings import MappingType, MAPPING_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, ConstraintSolverType, IntersectionMethod, add_scene_header, VISUAL_STYLES, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.materials import Material, ConstitutiveModel, MATERIALS_PLUGIN_LIST
from sofa_env.sofa_templates.camera import Camera, CAMERA_PLUGIN_LIST
from sofa_env.sofa_templates.rigid import RigidObject, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.deformable import rigidify, DEFORMABLE_PLUGIN_LIST
from sofa_env.sofa_templates.motion_restriction import add_bounding_box, MOTION_RESTRICTION_PLUGIN_LIST
from sofa_env.sofa_templates.solver import ConstraintCorrectionType, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST
from sofa_env.sofa_templates.collision import CollisionModelType, add_collision_model, COLLISION_PLUGIN_LIST

from sofa_env.scenes.tissue_manipulation.sofa_robot_functions import Workspace, WorkspaceType
from sofa_env.scenes.tissue_manipulation.sofa_objects.rigidified_tissue import Tissue, TISSUE_PLUGIN_LIST
from sofa_env.scenes.tissue_manipulation.sofa_objects.gripper import AttachedGripper, GRIPPER_PLUGIN_LIST
from sofa_env.scenes.tissue_manipulation.sofa_objects.visual_target import VisualTarget, VISUAL_TARGET_PLUGIN_LIST


PLUGIN_LIST = ["SofaPython3"] + TISSUE_PLUGIN_LIST + GRIPPER_PLUGIN_LIST + VISUAL_TARGET_PLUGIN_LIST + MATERIALS_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST + CAMERA_PLUGIN_LIST + MOTION_RESTRICTION_PLUGIN_LIST + RIGID_PLUGIN_LIST + DEFORMABLE_PLUGIN_LIST + SOLVER_PLUGIN_LIST + VISUAL_PLUGIN_LIST + COLLISION_PLUGIN_LIST + MAPPING_PLUGIN_LIST

LENGTH_UNIT = "m"
TIME_UNIT = "s"
WEIGHT_UNIT = "kg"

COLOR_FLOOR = (55 / 255.0, 130 / 255.0, 97 / 255.0)
COLOR_TISSUE = (220 / 255.0, 180 / 255.0, 50 / 255.0)
COLOR_LIVER = (215 / 255.0, 5 / 255.0, 25 / 255.0)
COLOR_BACKGROUND = (160 / 255.0, 165 / 255.0, 160 / 255.0)  # grey
COLOR_METAL = (40 / 255.0, 40 / 255.0, 35 / 255.0)
COLOR_CORNERS = (10 / 255.0, 10 / 255.0, 10 / 255.0)
COLOR_GRIPPER = (127 / 255.0, 127 / 255.0, 128 / 255.0)  # steel gray
COLOR_MAN_TARGET = (0 / 255.0, 0 / 255.0, 0 / 255.0)  # black
COLOR_VIS_TARGET = (51 / 255.0, 223 / 255.0, 255 / 255.0)  # turquoise

HERE = Path(__file__).resolve().parent
MESHES = HERE / "meshes"


def createScene(
    root_node: Sofa.Core.Node,
    image_shape: Tuple[Optional[int], Optional[int]] = (400, 400),
    show_everything: bool = False,
    show_fixed_points: bool = False,
    show_grasping_point: bool = False,
    show_workspace_bounding_box: bool = False,
    with_gravity: bool = True,
    with_manipulation_target: bool = True,
    with_visual_target: bool = True,
    with_collision_models: bool = False,
    randomize_grasping_point: bool = False,
    randomize_manipulation_target: bool = False,
    workspace_kwargs: Optional[dict] = None,
    animation_loop: AnimationLoopType = AnimationLoopType.FREEMOTION,
    camera_position: Union[np.ndarray, Tuple] = (0.0, -0.21, 0.11),
    camera_look_at: Union[np.ndarray, Tuple] = (0.0, 0.0, 0.11),
    camera_field_of_view_vertical: int = 57,  # zed2
) -> Dict:
    """
    Creates the scene of the TissueManipulationEnv.

    Args:
        root_node (Sofa.Core.Node): SET BY ENV. The simulation tree's root node.
        image_shape (Tuple[Optional[int], Optional[int]]): SET BY ENV. Height and Width of the rendered images.
        show_everything (bool): Whether to render additional information such as the collision model surfaces and the frames of the gripper.
        show_fixed_points (bool): Whether to render the fixed and rigidified points of the tissue.
        show_grasping_point (bool): Whether to render the grasping point of the gripper.
        show_workspace_bounding_box (bool): Whether to render the bounding box of the workspace.
        show_sampling_bounding_box (bool): Whether to render the bounding box of the sampling area.
        with_gravity (bool): Whether to apply gravity to the scene.
        with_manipulation_target (bool): Whether to add the visual manipulation target to the tissue.
        with_visual_target (bool): Whether to add the visual goal target to the scene.
        with_collision_models (bool): Whether to add collision models to the scene.
        randomize_grasping_point (bool): Whether to randomize the grasping point of the gripper.
        randomize_manipulation_target (bool): Whether to randomize the position of the manipulation target on the tissue.
        workspace_kwargs (Optional[dict]): Optional arguments for the workspace class.
        target_margin (Optional[list]): Minimum distance to the workspace boundaries when sampling a new visual goal target.
        animation_loop (AnimationLoopType): Animation loop of the simulation.
        camera_position (Union[np.ndarray, Tuple]): Position of the camera.
        camera_look_at (Union[np.ndarray, Tuple]): Position the camera is looking at.
        camera_field_of_view_vertical (int): Vertical field of view of the camera.

    Returns:
        scene_creation_result = {
            "camera": camera,
            "interactive_objects": {
                "gripper": gripper,
                "tissue": tissue,
                "target": target,
            },
            "workspace": workspace,
        }
    """

    ########
    # Meshes
    ########
    gripper_surface_mesh_path = MESHES / "gripper_15deg_visual.stl"
    gripper_collision_mesh_path = MESHES / "gripper_15deg_collision.stl"

    visual_target_surface_mesh_path = MESHES / "visual_target_surface.stl"
    floor_surface_mesh_path = MESHES / "floor.stl"
    block_surface_mesh_path = MESHES / "metal_block.stl"
    corners_surface_mesh_path = MESHES / "corners.stl"

    tissue_surface_mesh_path = MESHES / "tissue_visual.stl"
    tissue_volume_mesh_path = MESHES / "tissue_volumetric.msh"
    tissue_collision_mesh_path = MESHES / "tissue_collision.stl"

    liver_surface_mesh_path = MESHES / "liver_visual.stl"
    liver_collision_mesh_path = MESHES / "liver_collision.stl"

    ############
    # Parameters
    ############
    tissue_mass = 1.07 * 44.367 / 1000.0  # 1.07 g/cm³ - Volume (Onshape) 44366,9681 mm³ = 44,367 cm³ - mass in kg
    liver_mass = 3.792  # mass of steel block

    gravity = (0.0, 0.0, -9.81) if with_gravity else (0.0, 0.0, 0.0)
    # smaller poisson ratio --> better simulation
    tissue_poisson_ratio = 0.47  # 0.47 - 0.49 https://www.azom.com/properties.aspx?ArticleID=920
    tissue_young_modulus = 80000  # 0.08 N / mm² -> 80.000 N / m² tech sheet (https://www.kaupo.de/shop/out/media/ECOFLEX_SERIE.pdf)

    tissue_rotation = (0.0, 0.0, 0.0)
    tissue_translation = (0.02, 0.0, 0.045)
    liver_rotation = (0.0, 0.0, 0.0)
    liver_translation = (0.0, 0.0, 0.0)
    gripper_translation = (0.0215, -0.005, 0.147)

    collision_group_gripper = 1
    collision_group_liver = 1
    collision_group_tissue = 1

    world_zero = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    ##########################
    # Workspace of the Gripper
    ##########################
    if workspace_kwargs is None:
        workspace_kwargs = {
            "bounds": np.asarray([-0.055, 0.045, 0.07, 0.125]),
            "translation": tissue_translation,
        }
    elif "bounds" in workspace_kwargs:
        workspace_kwargs["translation"] = tissue_translation

    if "workspace_type" not in workspace_kwargs.keys():
        workspace_kwargs["workspace_type"] = WorkspaceType.TISSUE_ALIGNED

    workspace = Workspace(**workspace_kwargs)

    ###################
    # Common Components
    ###################
    add_scene_header(
        root_node=root_node,
        plugin_list=PLUGIN_LIST,
        visual_style_flags=VISUAL_STYLES["debug"] if show_everything else VISUAL_STYLES["normal"],
        collision_detection_method=IntersectionMethod.MINPROXIMITY,
        gravity=gravity,
        scene_has_collisions=with_collision_models,
        collision_detection_method_kwargs={
            "alarmDistance": 0.005,
            "contactDistance": 0.001,
        },
        constraint_solver=ConstraintSolverType.LCP,
        constraint_solver_kwargs={
            "maxIt": 1000,
            "tolerance": 0.001,
        },
        animation_loop=animation_loop,
    )

    ###################
    # Camera and Lights
    ###################
    root_node.addObject(
        "LightManager",
        listening=True,
        ambient=(
            0.6,
            0.6,
            0.6,
            0.6,
        ),
    )

    root_node.addObject(
        "DirectionalLight",
        name="light1",  # name is required for domain randomization
        color=[1.0, 1.0, 1.0, 1.0],
        direction=[0.0, -0.3, 1.0],
    )

    placement_kwargs = {
        "distance": float(np.linalg.norm(np.asarray(camera_position) - np.asarray(camera_look_at))),  # is recalculated in SOFA
        "pivot": 2,  # custom manipulation settings are required for this scene due to scaling issues with the meshes
        "zoomSpeed": 25.0,
        "panSpeed": 0.5,
        "position": camera_position,
        "lookAt": camera_look_at,
        "orientation": (0.707, 0.0, 0.0, 0.707),  # fix orientation to perpendicular to tissue and horizontally aligned
    }

    camera = Camera(
        root_node=root_node,
        placement_kwargs=placement_kwargs,
        z_near=0.01,
        z_far=0.5,
        width_viewport=image_shape[0],
        height_viewport=image_shape[1],
        vertical_field_of_view=camera_field_of_view_vertical,
    )

    #####################################################
    # Dedicated SCENE NODE for the rest of the components
    #####################################################
    scene_node = root_node.addChild("SceneNode")

    ########
    # Tissue
    ########
    add_visual_tissue_model = partial(add_visual_model, color=COLOR_TISSUE)
    add_collision_tissue_model = partial(
        add_collision_model,
        mapping_type=MappingType.BARYCENTRIC,
        model_types=[CollisionModelType.POINT] if with_collision_models else [],
        collision_group=collision_group_tissue,
    )

    tissue = scene_node.addObject(
        Tissue(
            scale=0.001,  # because MSH volume mesh is in units meters - since gmsh 4.2 units are harder to convert
            rotation=tissue_rotation,
            translation=tissue_translation,
            animation_loop_type=animation_loop,
            constraint_correction_type=ConstraintCorrectionType.PRECOMPUTED,
            parent_node=scene_node,
            volume_mesh_path=tissue_volume_mesh_path,
            total_mass=tissue_mass,
            visual_mesh_path=tissue_surface_mesh_path,
            collision_mesh_path=tissue_collision_mesh_path,
            material=Material(
                constitutive_model=ConstitutiveModel.COROTATED,
                poisson_ratio=tissue_poisson_ratio,
                young_modulus=tissue_young_modulus,
            ),
            add_visual_model_func=add_visual_tissue_model,
            add_collision_model_func=add_collision_tissue_model,
            with_manipulation_target=with_manipulation_target,
            manipulation_target_color=COLOR_MAN_TARGET,
            manipulation_target_mesh_path=visual_target_surface_mesh_path,
            randomize_manipulation_target=randomize_manipulation_target,
        )
    )

    # Initialize the node, so we can read the indices from the BoxROI. Before that, the mechanical object has nothing loaded.
    tissue.node.init()
    tissue.init()

    # Rigidification of the elasticobject for given indices with given frameOrientations.
    bb_rigid = add_bounding_box(
        attached_to=tissue.node,
        min=(0.01, -0.015, 0.09 + tissue_translation[-1]),
        max=(0.03, 0.005, 0.1075 + tissue_translation[-1]),
        show_bounding_box=False,
    )
    bb_rigid.init()

    # Rigidify Tissue
    rigidified_tissue = rigidify(
        tissue,
        rigidification_indices=[item for sublist in bb_rigid.indices.toList() for item in sublist],
    )

    # Fix Tissue at ground
    bb_fixed = add_bounding_box(
        attached_to=rigidified_tissue.deformable,
        min=(-0.06, -0.02, 0.0),
        max=(0.06, 0.02, 0.025 + tissue_translation[-1]),
        show_bounding_box=False,
    )
    bb_fixed.init()
    rigidified_tissue.deformable.addObject("FixedProjectiveConstraint", indices=f"{bb_fixed.getLinkPath()}.indices")

    # Align rigid reference frame orientation with gripper orientation
    with rigidified_tissue.rigid.MechanicalObject.position.writeable() as positions:
        positions[0][3:] = [0.0, 0.0, 0.0, 1.0]

    # Visualization
    if show_fixed_points:
        tissue.deformable_object.node.BoxROI.drawPoints = True
        tissue.deformable_object.node.BoxROI.drawSize = 8
        tissue.deformable_object.node.RigidificationBoxROI.drawPoints = True
        tissue.deformable_object.node.RigidificationBoxROI.drawSize = 8

    #######
    # Liver
    #######
    RigidObject(
        name="liver",
        parent_node=scene_node,
        total_mass=liver_mass,
        pose=world_zero,
        fixed_position=True,
        fixed_orientation=True,
        visual_mesh_path=liver_surface_mesh_path,
        add_visual_model_func=partial(
            add_visual_model,
            rotation=liver_rotation,
            translation=liver_translation,
            color=COLOR_LIVER,
        ),
        collision_mesh_path=liver_collision_mesh_path if with_collision_models else None,
        add_collision_model_func=(
            partial(
                add_collision_model,
                rotation=liver_rotation,
                translation=liver_translation,
                collision_group=collision_group_liver,
            )
            if with_collision_models
            else add_collision_model
        ),
    )

    ###############################################################################################
    # Visual Target - Target CAN BE re-sampled in ENV to be in workspace around manipulation target
    ###############################################################################################
    if with_visual_target:
        target = VisualTarget(
            parent_node=scene_node,
            name="visual_target",
            total_mass=0.0,
            pose=(0.0,) * 6 + (1.0,),
            scale=1.0,
            visual_mesh_path=visual_target_surface_mesh_path,
            add_visual_model_func=partial(add_visual_model, color=COLOR_VIS_TARGET),
        )
    else:
        target = None

    #########
    # Gripper
    #########
    gripper = scene_node.addObject(
        AttachedGripper(
            name="gripper",
            parent_node=scene_node,
            rigidified_tissue=rigidified_tissue,
            grasping_active=True,
            randomize_grasping_point=randomize_grasping_point,
            show_grasping_point=show_grasping_point,
            workspace=workspace,
            visual_offset=np.asarray([0.0, -1.0 * 1e-3, 0.0] + [0.0] * 4),  # move grippe 1mm outside of tissue
            scale=1.0,
            position=gripper_translation,
            orientation=(0.0, 0.0, 0.0),  # deg, "zyx"
            total_mass=0.0,  # not affected by gravity
            visual_mesh_path=gripper_surface_mesh_path,
            collision_mesh_path=gripper_collision_mesh_path,
            add_collision_model_func=partial(
                add_collision_model,
                collision_group=collision_group_gripper,
                model_types=[CollisionModelType.POINT] if with_collision_models else [],
            ),
        )
    )
    if randomize_grasping_point:
        gripper.reset()

    ################
    # Visual Objects
    ################
    add_visual_floor = partial(add_visual_model, rotation=(90.0, 0.0, 0.0), translation=(0.0, 0.0, 0.029), color=COLOR_FLOOR)  # in mm - scale!
    # Visual floor board
    RigidObject(
        parent_node=scene_node,
        name="floor",
        scale=4,
        pose=world_zero,
        visual_mesh_path=floor_surface_mesh_path,
        add_visual_model_func=add_visual_floor,
    )

    # Visual background board
    add_visual_background = partial(add_visual_model, translation=(0.0, 0.1, 0.250), color=COLOR_BACKGROUND)  # in mm - scale!
    RigidObject(
        parent_node=scene_node,
        name="background",
        scale=4,
        pose=world_zero,
        visual_mesh_path=floor_surface_mesh_path,
        add_visual_model_func=add_visual_background,
    )

    # Metal block on which the liver is fixed
    RigidObject(
        parent_node=scene_node,
        name="metal_piece",
        scale=0.001,
        pose=world_zero,
        visual_mesh_path=block_surface_mesh_path,
        add_visual_model_func=partial(add_visual_model, color=COLOR_METAL),
    )

    # Corner pieces that hold the metal block
    RigidObject(
        parent_node=scene_node,
        name="corners",
        scale=0.001,
        pose=world_zero,
        visual_mesh_path=corners_surface_mesh_path,
        add_visual_model_func=partial(add_visual_model, color=COLOR_CORNERS),
    )

    ###############
    # Visualization
    ###############
    scene_node.addObject("MechanicalObject", template="Rigid3d", position=world_zero)
    if show_workspace_bounding_box:
        # adding offset for y-value since they are not allowed to be the same
        add_bounding_box(scene_node, min=workspace.get_low() - np.asarray([0, 1e-3, 0]), max=workspace.get_high(), show_bounding_box=show_workspace_bounding_box)

    ###############
    # Returned Data
    ###############
    scene_creation_result = {
        "camera": camera,
        "interactive_objects": {
            "gripper": gripper,
            "tissue": tissue,
            "target": target,
        },
        "workspace": workspace,
    }

    return scene_creation_result
