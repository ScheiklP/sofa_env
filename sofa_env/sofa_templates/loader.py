import Sofa.Core
from typing import Union, Tuple
from pathlib import Path, PosixPath

LOADER_PLUGIN_LIST = ["Sofa.Component.IO.Mesh"]

LOADER_INFOS = {
    ".obj": "MeshOBJLoader",
    ".stl": "MeshSTLLoader",
    ".vtk": "MeshVTKLoader",
    ".msh": "MeshGmshLoader",
    ".gidmsh": "GIDMeshLoader",
}


def check_file_path(file_path: Union[Path, str]) -> Path:
    """Checks if file exists either as relative or absolute path and returns the valid one."""
    assert type(file_path) in (
        PosixPath,
        str,
    ), f"Please pass the path to the file as str or pathlib.Path. Received <<{type(file_path)}>>."

    file_path = Path(file_path)

    if not file_path.is_file():
        assert file_path.absolute().is_file(), f"Could not find file under either relative ({file_path}) or absolute path ({file_path.absolute()})."
        file_path = file_path.absolute()

    return file_path


def loader_for(file_path: Union[str, Path]) -> str:
    """Look up the correct loader for a given filepath.

    Args:
        file_path (Union[str, Path]): Path to the file that is to be loaded.

    Returns:
        The name of the correct SOFA object to load the file.

    Examples:
        >>> model.addObject(loader_for("liver.stl"), filename="liver.stl")

    """
    assert type(file_path) in (
        PosixPath,
        str,
    ), f"Please pass the path to the file as str or pathlib.Path. Received <<{type(file_path)}>>."

    file_path = check_file_path(file_path)
    file_type = file_path.suffix

    assert file_type in LOADER_INFOS, f"No loader found for {file_path} of type {file_type}"

    return LOADER_INFOS[file_type]


def add_loader(
    attached_to: Sofa.Core.Node,
    file_path: Union[str, Path],
    name: str = "loader",
    scale: Union[float, Tuple[float, float, float]] = 1.0,
    loader_kwargs: dict = {}
) -> Sofa.Core.Object:
    """Adds a loader object to a node.

    Args:
        attached_to (Sofa.Core.Node): Parent node of the loader object.
        file_path (Union[str, Path]): Path to the file that is to be loaded.
        name (str): Name that is assigned to the loader.
        loader_kwargs (dict): Optional keyword arguments for the loader.

    Returns:
        loader_object (Sofa.Core.Object): The loader object.

    Examples:
        >>> loader_for("liver.stl")
        "MeshSTLLoader"
    """

    assert type(file_path) in (
        PosixPath,
        str,
    ), f"Please pass the path to the file as str or pathlib.Path. Received <<{type(file_path)}>>."

    file_path = check_file_path(file_path)
    file_type = file_path.suffix

    assert file_type in LOADER_INFOS, f"No loader found for {file_path} of type {file_type}"

    if LOADER_INFOS[file_type] in ("MeshVTKLoader", "MeshSTLLoader"):
        loader_kwargs.pop("handleSeams", None)

    if isinstance(scale, tuple):
        loader_kwargs.pop("scale", None)
        loader_kwargs["scale3d"] = scale
    else:
        loader_kwargs.pop("scale3d", None)
        loader_kwargs["scale"] = scale

    loader_object = attached_to.addObject(
        LOADER_INFOS[file_type],
        filename=str(file_path),
        name=name,
        **loader_kwargs,
    )

    return loader_object
