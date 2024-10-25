import Sofa.Core
import inspect

from enum import Enum
from functools import partial
from pathlib import Path
from typing import Tuple, Union, Optional, List, Callable

from sofa_env.sofa_templates.loader import add_loader, LOADER_PLUGIN_LIST
from sofa_env.sofa_templates.mappings import MappingType, MAPPING_PLUGIN_LIST

COLLISION_PLUGIN_LIST = (
    [
        "Sofa.Component.Topology.Container.Constant",  # <- [MeshTopology]
        "Sofa.Component.Collision.Geometry",  # <- [LineCollisionModel, PointCollisionModel, SphereCollisionModel, TriangleCollisionModel]
    ]
    + LOADER_PLUGIN_LIST
    + MAPPING_PLUGIN_LIST
)


class CollisionModelType(Enum):
    """SOFA names for collision models represented as points, lines, and triangles."""

    POINT = "PointCollisionModel"
    LINE = "LineCollisionModel"
    TRIANGLE = "TriangleCollisionModel"


def is_default_collision_model_function(function: Callable) -> bool:
    """Check if ``function`` is the same as or a partial of ``add_collision_model``

    Args:
        function (Callable): The function to check.

    Returns:
        True, if ``function`` is the same or a partial of ``add_collision_model``
    """
    return function is add_collision_model if not isinstance(function, partial) else function.func is add_collision_model


def match_collision_model_kwargs(function: Callable, locals: dict) -> dict:
    """Check a dictionary of local variables for keyword matches of a custom collision model function.

    The function should at least accept ``attached_to`` as an argument to know where to add the SOFA objects.
    If the function expects parameters that are not in the locals, an Error is raised.

    Args:
        function (Callable): The custom collision model function
        locals (dict): The dictionary of local variables

    Returns:
        function_kwargs (dict): A subset of the local variables, that match the signature of ``function``.
    """
    function_kwargs = {}
    expected_parameters = inspect.signature(function).parameters
    unmatched_kwargs = []
    for parameter in expected_parameters:
        if parameter in locals:
            function_kwargs[parameter] = locals[parameter]
        else:
            unmatched_kwargs.append(parameter)

    if "attached_to" not in expected_parameters:
        raise ValueError(f"Custom collision model function {function} should accept a Sofa.Core.Node as parameter attached_to.")
    else:
        unmatched_kwargs.remove("attached_to")

    if unmatched_kwargs:
        raise ValueError(f"Custom collision model function {function} expected parameters {unmatched_kwargs} that were not found in the local variables of the objects __init__ function.")

    return function_kwargs


def add_collision_model(
    attached_to: Sofa.Core.Node,
    surface_mesh_file_path: Union[str, Path],
    scale: Union[float, Tuple[float, float, float]] = 1.0,
    name: str = "collision",
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    model_types: List[CollisionModelType] = [
        CollisionModelType.POINT,
        CollisionModelType.LINE,
        CollisionModelType.TRIANGLE,
    ],
    collision_group: Optional[int] = None,
    contact_stiffness: Optional[Union[float, int]] = None,
    is_static: bool = False,
    check_self_collision: Optional[bool] = None,
    mapping_type: MappingType = MappingType.BARYCENTRIC,
    mapping_kwargs: dict = {},
    triangulate: bool = True,
    is_carving_tool: bool = False,
) -> Sofa.Core.Node:
    """Adds a collision model to a node.

    Without collision models, objects do not interact on touch.
    For more details see https://www.sofa-framework.org/community/doc/components/collisions/collisionmodels/.

    Args:
        attached_to (Sofa.Core.Node): Parent node of the collision model.
        surface_mesh_file_path (Union[Path, str]): Path to the surface mesh that is to be used as a collision surface.
        scale: (Union[float, Tuple[float, float, float]]): Scaling factor for the imported mesh.
        name (str): Name of the collision model node.
        rotation (Tuple[float, float, float]): RPY rotations in degrees of the collision model in relation to the parent node. Order is XYZ.
        translation (Tuple[float, float, float]): XYZ translation of the collision model in relation to the parent node.
        model_types ([List[CollisionModelType]]): Types of models in the mesh to be used for collision checking.
        collision_group (int): Add the model to a collision group to disable collision checking between those models.
        contact_stiffness (Optional[Union[float, int]]): How `strong` should the surface repel the collision before `giving in`?
        is_static (bool): Should only be set for rigid, immobile objects. The object does not move in the scene (e.g. floor, wall) but reacts to collision. From the official SOFA documentation: Usually, two colliding objects having simulated being false are not considered in collision detection. Self-collision is not considered if simulated is false. If one of two colliding objects has simulated being false, the contact response is created as a child of the other.
        check_self_collision (bool): Whether to check for self collision in the model.
        mapping_type (MappingType): What mapping is to be used between parent and child topology? E.g. ``"BarycentricMapping"`` for mapping a mesh to a mesh, ``"RigidMapping"`` for mapping a mesh to a Rigid3 body (1 pose), ``"IdentityMapping"`` for mapping two identical meshes.
        mapping_kwargs (dict): Additional keyword arguments for the ``mapping_type``. For example ``{"input": node.MechanicalObject.getLinkPath(), index: 1}`` for a rigid mapping to the second index of a ``"MechanicalObject"``.
        triangulate (bool): Divide all polygons of the mesh into triangles.
        is_carving_tool (bool): If set to True, will add a ``"CarvingTool"`` tag to the collision models. Requires the SofaCarving plugin to be compiled.

    Returns:
        collision_model_node (Sofa.Core.Node): Sofa node with collision objects.
    """

    assert Path(surface_mesh_file_path).suffix in (
        ".stl",
        ".obj",
    ), f"Can only create a collision model with .stl or .obj files. Got {Path(surface_mesh_file_path).suffix}"

    collision_model_node = attached_to.addChild(name)

    loader_object = add_loader(
        attached_to=collision_model_node,
        file_path=surface_mesh_file_path,
        scale=scale,
        loader_kwargs={"rotation": rotation, "translation": translation, "triangulate": triangulate},
    )

    # dictionary of relevant keyword arguments for the collision model object
    collision_model_kwargs = {}

    if collision_group is not None:
        collision_model_kwargs["group"] = collision_group

    if contact_stiffness is not None:
        collision_model_kwargs["contactStiffness"] = contact_stiffness

    if check_self_collision is not None:
        collision_model_kwargs["selfCollision"] = check_self_collision

    if is_static:
        collision_model_kwargs["moving"] = False
        collision_model_kwargs["simulated"] = False
    else:
        collision_model_kwargs["moving"] = True
        collision_model_kwargs["simulated"] = True

    if is_carving_tool:
        collision_model_kwargs["tags"] = "CarvingTool"

    collision_model_node.addObject("MeshTopology", src=loader_object.getLinkPath())
    collision_model_node.addObject("MechanicalObject", template="Vec3d")

    for collision_model_type in model_types:
        collision_model_node.addObject(collision_model_type.value, **collision_model_kwargs)

    collision_model_node.addObject(mapping_type.value, **mapping_kwargs)

    return collision_model_node
