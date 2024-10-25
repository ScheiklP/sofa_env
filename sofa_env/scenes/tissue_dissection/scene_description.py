import Sofa
import Sofa.Core

import numpy as np

from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Dict, Union

from sofa_env.sofa_templates.deformable import SimpleDeformableObject, DEFORMABLE_PLUGIN_LIST
from sofa_env.sofa_templates.camera import Camera, CAMERA_PLUGIN_LIST
from sofa_env.sofa_templates.mappings import MappingType, MAPPING_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, IntersectionMethod, add_scene_header, VISUAL_STYLES, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.topology import create_initialized_grid, TOPOLOGY_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST

from sofa_env.utils.camera import determine_look_at

from sofa_env.scenes.tissue_dissection.sofa_objects.cauter import PivotizedCauter, CAUTER_PLUGIN_LIST

LENGTH_UNIT = "mm"
TIME_UNIT = "s"
WEIGHT_UNIT = "kg"

PLUGIN_LIST = (
    [
        "SofaPython3",
        "Sofa.Component.Constraint.Lagrangian.Model",  # [BilateralLagrangianConstraint]
        "Sofa.Component.MechanicalLoad",  # [ConstantForceField, PlaneForceField, UniformVelocityDampingForceField]
        "Sofa.Component.Mass",  # [UniformMass]
        "Sofa.Component.StateContainer",  # [MechanicalObject]
        "Sofa.Component.Mapping.NonLinear",  # [RigidMapping]
        "SofaCarving",  # [CarvingManager]
    ]
    + SCENE_HEADER_PLUGIN_LIST
    + TOPOLOGY_PLUGIN_LIST
    + DEFORMABLE_PLUGIN_LIST
    + CAUTER_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
    + CAUTER_PLUGIN_LIST
    + CAMERA_PLUGIN_LIST
    + MAPPING_PLUGIN_LIST
)

HERE = Path(__file__).resolve().parent
INSTRUMENT_MESH_DIR = HERE.parent.parent.parent / "assets/meshes/instruments"
MESH_DIR = HERE.parent.parent.parent / "assets/meshes/models"


def createScene(
    root_node: Sofa.Core.Node,
    image_shape: Tuple[Optional[int], Optional[int]] = (None, None),
    debug_rendering: bool = False,
    animation_loop: AnimationLoopType = AnimationLoopType.FREEMOTION,
    with_board_collision: bool = True,
    rows_to_cut: int = 3,
    show_border_point: bool = True,
    cauter_reset_noise: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = np.array([30.0, 30.0, 30.0, 10.0]),
) -> Dict:
    """
    Creates the scene of the TissueDissectionEnv.

    Args:
        root_node (Sofa.Core.Node): SET BY ENV. The simulation tree's root node.
        image_shape (Tuple[Optional[int], Optional[int]]): SET BY ENV. Height and Width of the rendered images.
        debug_rendering (bool): Whether to render more information for debugging purposes.
        animation_loop (AnimationLoopType): Animation loop of the simulation.
        with_board_collision (bool): SET BY ENV. Whether to add a collision model to the board. This will influence the behavior of the cauter whenn touching the board.
        With a collision model, the remote center of motion will be violated, when the cauter tries to move into the board and the cauter will be deflected.
        rows_to_cut (int): SET BY ENV. How many rows along the tissue should be simulated by cuttable connective tissue.
        The tissue is 10 rows long and the first 2 rows are not attached to the board. The rest of the tissue is connected to the board.
        ``rows_to_cut`` controls how many of the remaining rows are connected by deformable, cuttable connective tissue. The rest is fixed to the board with springs.
        show_border_point (bool): Whether to render the point on the connective tissue that is closest to the cauter tip.
        cauter_reset_noise (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]): Limits to uniformly sample noise that is added to the cauter's initial state at reset.

    Returns:
        scene_creation_results = {
            "camera": camera,
            "border_point_mechanical_object": border_point_mechanical_object,
            "cauter": cauter,
            "tissue": tissue,
            "connective_tissue": connective_tissue,
            "root": root_node,
            "contact_listener": board_cauter_contact_listener,
            "retraction_force": retraction_force,
            "topology_info": topology_info,
        }
    """

    discretization_x = 6
    discretization_y = 10
    discretization_z = 2

    points_offset_in_y = 2
    if rows_to_cut > (discretization_y - points_offset_in_y) or rows_to_cut <= 1:
        raise ValueError("rows_to_cut must be between 2 and {discretization_y - points_offset_in_y}")

    width_x = 60.0
    height_z = 2.0
    length_y = 100.0

    ###################
    # Common components
    ###################
    contact_distance = 0.2
    alarm_distance = contact_distance / 5.0
    gravity = -981  # mm/s^2
    add_scene_header(
        root_node=root_node,
        plugin_list=PLUGIN_LIST,
        visual_style_flags=VISUAL_STYLES["debug"] if debug_rendering else VISUAL_STYLES["normal"],
        gravity=(0.0, 0.0, gravity),
        animation_loop=animation_loop,
        scene_has_collisions=True,
        collision_detection_method_kwargs={
            "alarmDistance": alarm_distance,
            "contactDistance": contact_distance,
        },
        collision_detection_method=IntersectionMethod.LOCALMIN,
        contact_friction=0.0,
    )

    ###################
    # Camera and lights
    ###################
    root_node.addObject(
        "LightManager",
        listening=True,
        ambient=(0.5, 0.5, 0.5, 0.5),
    )
    camera_config = {
        "placement_kwargs": {
            "position": [width_x, length_y * 1.27, 34.0],
            "orientation": [0.2238795, 0.54386986, 0.75630178, 0.28651555],
        },
        "vertical_field_of_view": 62.0,
    }

    camera_config["placement_kwargs"]["lookAt"] = determine_look_at(
        camera_config["placement_kwargs"]["position"],
        camera_config["placement_kwargs"]["orientation"],
    )

    camera = Camera(
        root_node=root_node,
        placement_kwargs=camera_config["placement_kwargs"],
        z_near=1.0,
        z_far=250.0,
        width_viewport=image_shape[0],
        height_viewport=image_shape[1],
        show_object=False,
        with_light_source=True,
        show_object_scale=5.0,
        vertical_field_of_view=camera_config["vertical_field_of_view"],
    )

    scene_node = root_node.addChild("scene")

    tissue_grid_topology, _, tissue_tetrahedron_topology = create_initialized_grid(
        attached_to=scene_node,
        name="tissue_topology",
        xmin=-width_x / 2.0,
        xmax=width_x / 2.0,
        ymin=0.0,
        ymax=length_y,
        zmin=0.1,
        zmax=height_z + 0.1,
        num_x=discretization_x,
        num_y=discretization_y,
        num_z=discretization_z,
    )

    point_distance_y = length_y / discretization_y
    connective_tissue_grid_topology, _, connective_tissue_tetrahedron_topology = create_initialized_grid(
        attached_to=scene_node,
        name="connective_tissue_topology",
        num_x=discretization_x,
        num_y=rows_to_cut,
        num_z=discretization_z,
        xmin=-width_x / 2.0,
        xmax=width_x / 2.0,
        ymin=point_distance_y * (discretization_y - points_offset_in_y - rows_to_cut),
        ymax=length_y - point_distance_y * points_offset_in_y,
        zmin=0.0,
        zmax=0.1,
    )

    #######
    # Board
    #######
    # Rigid Base
    board_node = scene_node.addChild("board")
    board_node.addObject("EulerImplicitSolver")
    board_node.addObject("CGLinearSolver")
    board_node.addObject("MechanicalObject", template="Rigid3d", showObject=debug_rendering, showObjectScale=1.0)
    board_node.addObject("UniformMass", totalMass=100.0)
    board_node.addObject("FixedProjectiveConstraint")

    # Mapped Grid
    grid_node = board_node.addChild("grid")
    grid_node.addObject(
        "RegularGridTopology",
        nx=discretization_x,
        ny=discretization_y,
        nz=discretization_z,
        xmin=-width_x / 2.0,
        xmax=width_x / 2.0,
        ymin=0.0,
        ymax=length_y,
        zmin=-2 * height_z,
        zmax=0.0,
    )

    collision_node = grid_node.addChild("collision")
    collision_node.addObject(
        "RegularGridTopology",
        nx=discretization_x,
        ny=discretization_y,
        nz=1,
        xmin=-width_x / 2.0,
        xmax=width_x / 2.0,
        ymin=0.0,
        ymax=length_y,
        zmin=-1.0,
        zmax=-1.0,
    )

    if with_board_collision:
        collision_node.addObject("MechanicalObject", template="Vec3d", showObject=debug_rendering, showObjectScale=1.0)
        board_collision_model = collision_node.addObject("TriangleCollisionModel", group=0)
        collision_node.addObject("RigidMapping")
    else:
        board_collision_model = None

    grid_node.addObject("OglModel")
    grid_node.addObject("RigidMapping")

    ########
    # Tissue
    ########
    tissue = SimpleDeformableObject(
        parent_node=scene_node,
        name="tissue",
        positions=tissue_grid_topology.position.array().copy(),
        tetrahedra=tissue_tetrahedron_topology.tetrahedra.array().copy(),
        young_modulus=5e2,
        poisson_ratio=0.0,
        total_mass=100.0,
        show_object=debug_rendering,
        show_object_scale=2.0,
        cuttable=True,
        color=(1.0, 0.0, 0.0),
        collision_group=0,
        animation_loop_type=animation_loop,
    )
    tissue.node.addObject("UniformVelocityDampingForceField", dampingCoefficient=0.8, implicit=True, rayleighStiffness=0.0)
    tissue.node.addObject(
        "PlaneForceField",
        normal=[0, 0, 1],
        d=0.0,
        stiffness=3e5,
        damping=1,
        showPlane=True,
        showPlaneSize=60,
    )

    grasp_box = tissue.node.addObject("BoxROI", box=(6, length_y + 5, 5) + (-6, length_y - 5, -1), name="grasp_box")
    retraction_force = tissue.node.addObject("ConstantForceField", totalForce=[0, -1e5, 1e5], indices=grasp_box.indices.getLinkPath(), showArrowSize=1e-3)

    ############################################
    # Connective Tissue between Tissue and Board
    ############################################
    connective_tissue = SimpleDeformableObject(
        parent_node=scene_node,
        name="connective_tissue",
        positions=connective_tissue_grid_topology.position.array().copy(),
        tetrahedra=connective_tissue_tetrahedron_topology.tetrahedra.array().copy(),
        young_modulus=5.0,
        poisson_ratio=0.0,
        total_mass=0.5,
        show_object=debug_rendering,
        show_object_scale=2.0,
        cuttable=True,
        color=(0.0, 0.0, 1.0),
        collision_group=0,
        animation_loop_type=animation_loop,
    )

    indices_connective_tissue_to_board = list(range(int(discretization_z * discretization_x * rows_to_cut / 2)))
    indices_connective_tissue_to_tissue = list(range(indices_connective_tissue_to_board[-1] + 1, int(discretization_z * discretization_x * rows_to_cut)))
    num_constraint_points = len(indices_connective_tissue_to_tissue)

    relevant_tissue_indices = list(range(int(discretization_z * discretization_x * (discretization_y - points_offset_in_y) / 2)))
    indices_tissue_to_board = relevant_tissue_indices[:-num_constraint_points]
    indices_tissue_to_connective_tissue = relevant_tissue_indices[-num_constraint_points:]

    connective_tissue.node.addObject("FixedProjectiveConstraint", indices=indices_connective_tissue_to_board)
    tissue.node.addObject("RestShapeSpringsForceField", stiffness=1e3, points=indices_tissue_to_board)

    topology_info = {
        "discretization": (discretization_x, discretization_y, discretization_z),
        "size": (width_x, length_y, height_z),
        "points_offset_in_y": points_offset_in_y,
        "rows_to_cut": rows_to_cut,
        "indices_connective_tissue_to_board": indices_connective_tissue_to_board,
        "indices_connective_tissue_to_tissue": indices_connective_tissue_to_tissue,
        "indices_tissue_to_board": indices_tissue_to_board,
        "indices_tissue_to_connective_tissue": indices_tissue_to_connective_tissue,
    }

    connective_tissue.node.addObject(
        "BilateralLagrangianConstraint",
        template="Vec3d",
        object1=tissue.node.MechanicalObject.getLinkPath(),
        object2=connective_tissue.node.MechanicalObject.getLinkPath(),
        topology1=tissue.topology_container.getLinkPath(),
        topology2=connective_tissue.topology_container.getLinkPath(),
        first_point=indices_tissue_to_connective_tissue,
        second_point=indices_connective_tissue_to_tissue,
    )

    ########
    # Cauter
    ########
    add_visual_model_cauter = partial(add_visual_model, color=(0.0, 1.0, 0.0))
    cartesian_workspace = {
        "low": np.array([-width_x * 0.8, 0.0, -10.0]),
        "high": np.array([width_x * 0.8, length_y + 10.0, length_y * 0.8]),
    }
    cauter = PivotizedCauter(
        parent_node=scene_node,
        name="cauter",
        visual_mesh_path=INSTRUMENT_MESH_DIR / "dissection_electrode.stl",
        show_object=debug_rendering,
        show_object_scale=10,
        add_visual_model_func=add_visual_model_cauter,
        ptsd_state=np.array([0.0, 0.0, 0.0, 2.0]),
        rcm_pose=np.array([-width_x * (2 / 6), cartesian_workspace["high"][1], 30.0, 135.0, 0.0, 0.0]),
        animation_loop_type=animation_loop,
        spring_stiffness=1e20,
        angular_spring_stiffness=1e20,
        total_mass=1e3,
        cartesian_workspace=cartesian_workspace,
        state_limits={
            "low": np.array([-90, -90, -90, 0]),
            "high": np.array([90, 90, 90, 300]),
        },
        ptsd_reset_noise=cauter_reset_noise,
    )

    scene_node.addObject(cauter)

    carving_manager = root_node.addObject(
        "CarvingManager",
        carvingDistance=contact_distance,
        active=False,
    )
    cauter.set_carving_manager(carving_manager)

    if with_board_collision:
        if board_collision_model is None:
            raise ValueError("board_collision_model is None")
        board_cauter_contact_listener = scene_node.addObject(
            "ContactListener",
            name="contact_listener_cauter_board",
            collisionModel1=cauter.cutting_sphere_collision_model.getLinkPath(),
            collisionModel2=board_collision_model.getLinkPath(),
        )
    else:
        board_cauter_contact_listener = None

    border_point_node = scene_node.addChild("border_point")
    border_point_mechanical_object = border_point_node.addObject(
        "MechanicalObject",
        template="Rigid3d",
        position=[0.0] * 6 + [1.0],
    )

    if show_border_point:
        add_visual_model(attached_to=border_point_node, surface_mesh_file_path=MESH_DIR / "unit_sphere.stl", scale=2.0, color=(1.0, 1.0, 0.0), mapping_type=MappingType.RIGID)

    scene_creation_results = {
        "camera": camera,
        "border_point_mechanical_object": border_point_mechanical_object,
        "cauter": cauter,
        "tissue": tissue,
        "connective_tissue": connective_tissue,
        "root": root_node,
        "contact_listener": board_cauter_contact_listener,
        "retraction_force": retraction_force,
        "topology_info": topology_info,
    }

    return scene_creation_results
