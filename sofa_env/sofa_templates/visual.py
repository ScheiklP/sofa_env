import Sofa.Core

import numpy as np

from pathlib import Path
from typing import Tuple, Union, Optional, List

from sofa_env.sofa_templates.loader import add_loader, LOADER_PLUGIN_LIST
from sofa_env.sofa_templates.mappings import MappingType, MAPPING_PLUGIN_LIST


VISUAL_PLUGIN_LIST = (
    [
        "Sofa.GL.Component.Rendering3D",  # [OglModel]
        "Sofa.GL.Component.Shader",  # [LightManager, PositionalLight]
        "Sofa.Component.Topology.Container.Constant",  # [MeshTopology]
        "Sofa.Component.Mapping.Linear",  # [BarycentricMapping]
    ]
    + LOADER_PLUGIN_LIST
    + MAPPING_PLUGIN_LIST
)


def get_ogl_models(node: Sofa.Core.Node, model_list: Optional[List[Sofa.Core.Object]] = None) -> List[Sofa.Core.Object]:
    """Recursively searches for all ``"OglModel"`` objects in a node and its children.

    Args:
        node (Sofa.Core.Node): The node to search for ``"OglModel"`` objects.
        model_list (Optional[List[Sofa.Core.Object]]): List of ``"OglModel"`` objects.

    Returns:
        model_list (List[Sofa.Core.Object]): List of ``"OglModel"`` objects.

    Example:
        >>> get_ogl_models(root_node)
    """

    if model_list is None:
        model_list = []

    for obj in node.objects:
        if hasattr(obj, "getAsACreateObjectParameter") and (obj_path := obj.getAsACreateObjectParameter()) is not None:
            if obj_path.endswith("OglModel"):
                model_list.append(obj)

    for child in node.children:
        get_ogl_models(child, model_list)

    return model_list


def add_visual_model(
    attached_to: Sofa.Core.Node,
    surface_mesh_file_path: Union[str, Path],
    scale: Union[float, Tuple[float, float, float]] = 1.0,
    name: str = "visual",
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    color: Optional[Tuple[float, float, float]] = None,
    transparency: Optional[float] = None,
    mapping_type: MappingType = MappingType.BARYCENTRIC,
    mapping_kwargs: dict = {},
    triangulate: bool = True,
    texture_file_path: Optional[str] = None,
    handle_seams: bool = True,
) -> Sofa.Core.Node:
    """Adds a visual model to a node.

    Warning:
        Do not use with SofaCarving plugin.

    Args:
        attached_to (Sofa.Core.Node): Parent node of the visual model.
        surface_mesh_file_path (Union[Path, str]): Path to the surface mesh that is to be used as a visual surface.
        scale: (Union[float, Tuple[float, float, float]]): Scaling factor for the imported mesh.
        name (str): Name of the visual model node.
        rotation (Tuple[float, float, float]): RPY rotations in degrees of the visual model in relation to the parent node. Order is X*Y*Z.
        translation (Tuple[float, float, float]): XYZ translation of the visual model in relation to the parent node.
        color (Optional[Tuple[float, float, float]]): RGB values between 0 and 1 for the mesh.
        transparency (Optional[float]): Transparency of the mesh between 0 and 1.
        mapping_type (MappingType): What mapping is to be used between parent and child topology? E.g. ``"BarycentricMapping"`` for mapping a mesh to a mesh, ``"RigidMapping"`` for mapping a mesh to a Rigid3 body (1 pose), ``"IdentityMapping"`` for mapping two identical meshes.
        mapping_kwargs (dict): Additional keyword arguments for the ``mapping_type``. For example ``{"input": node.MechanicalObject.getLinkPath(), index: 1}`` for a rigid mapping to the second index of a ``"MechanicalObject"``.
        triangulate (bool): Divide all polygons of the mesh into triangles.
        texture_file_path (Optional[Union[Path, str]]): Path to the texture file that is to be used as a texture for the visual surface.
        handle_seams (bool): duplicate vertices when texture coordinates are present (as it is possible that one vertex has multiple texture coordinates).

    Returns:
        visual_model_node (Sofa.Core.Node): Sofa node with ``"OglModel"``.

    Example:
        >>> rigid_node = root_node.addChild("rigid")
        >>> rigid_node.addObject("MechanicalObject", template="Rigid3d", position=[0.0] * 6 + [1.0])
        >>> add_visual_model(attached_to=rigid_node, surface_mesh_file_path=<path_to_mesh.stl>, mapping_type=MappingType.RIGID)
    """

    assert Path(surface_mesh_file_path).suffix in (
        ".stl",
        ".obj",
    ), f"Can only create a visual model with stl or obj files. Got {Path(surface_mesh_file_path).suffix}"

    if texture_file_path and color:
        raise Exception("Can only set color or texture, not both at the same time.")

    visual_model_node = attached_to.addChild(name)

    loader_object = add_loader(
        attached_to=visual_model_node,
        file_path=surface_mesh_file_path,
        loader_kwargs={"triangulate": triangulate, "handleSeams": handle_seams if texture_file_path is not None else None},
    )

    ogl_model_kwargs = {}
    if isinstance(scale, tuple):
        ogl_model_kwargs["scale3d"] = scale
    else:
        ogl_model_kwargs["scale"] = scale

    if texture_file_path is not None:
        visual_model_node.addObject(
            "OglModel",
            texturename=texture_file_path,
            src=loader_object.getLinkPath(),
            rotation=rotation,
            translation=translation,
            **ogl_model_kwargs,
        )
    else:
        if transparency is not None and transparency != 0.0:
            if color is None:
                color = (202 / 255, 203 / 255, 207 / 255)
            material = f"Transparent Diffuse 1 {color[0]} {color[1]} {color[2]} {1 - transparency} Ambient {1 if transparency else 0} {color[0]} {color[1]} {color[2]} 1 Specular {1 if transparency else 0} 1 1 1 1 Emissive 0 0 0 0 1 Shininess {1 if transparency else 0} 100"
            ogl_model_kwargs["material"] = material
        elif color is not None:
            ogl_model_kwargs["color"] = color

        visual_model_node.addObject(
            "OglModel",
            src=loader_object.getLinkPath(),
            rotation=rotation,
            translation=translation,
            **ogl_model_kwargs,
        )

    visual_model_node.addObject(mapping_type.value, **mapping_kwargs)

    return visual_model_node


def set_color(ogl_model: Sofa.Core.Object, color: Union[np.ndarray, Tuple[float, float, float]]) -> None:
    """Sets the color of an ogl model.

    Args:
        ogl_model (Sofa.Core.Object): The ``"OglModel"`` you want to change.
        color (Union[np.ndarray, Tuple[float, float, float]]): RGB values of the new color in [0, 1].

    Example:
        >>> object = RigidObject(...)
        >>> set_color(ogl_model=object.visual_model_node.OglModel, color=(1.0, 1.0, 0.0))
    """

    # Create some short-hands
    a = ogl_model.material.value
    r = color[0]
    g = color[1]
    b = color[2]

    # For some reason SOFA sets the ambient color as 1/5 of color
    rm = r / 5
    gm = g / 5
    bm = b / 5
    s = a.split()

    # Assemble the string that SOFA uses to set the material properties of an object
    new_color_string = f"{s[0]} {s[1]} {s[2]} {r} {g} {b} {s[6]} {s[7]} {s[8]} {rm} {gm} {bm} {s[12]} {s[13]} {s[14]} {r} {g} {b} {s[18]} {s[19]} {s[20]} {r} {g} {b} {s[24]} {s[25]} {s[26]} {s[27]}"
    ogl_model.material.value = new_color_string
