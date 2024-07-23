from pathlib import Path
from typing import Union
import open3d as o3d
import numpy as np
import os

HERE = Path(__file__).resolve().parent
ASSET_DIR = HERE.parent.parent / "assets"


def get_asset_dir() -> Path:
    """Get the asset directory.

    Returns:
        Path: The asset directory.
    """
    return ASSET_DIR


class PointCloudWriter:
    """Class to write point clouds to a ply file.

    Args:
        log_dir (Path): The directory to save the ``.ply`` files to.
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        overwrite: bool = False,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.overwrite = overwrite

        if self.overwrite:
            self.counter = 0
        else:
            existing_files = [filename for filename in self.log_dir.iterdir() if filename.suffix == ".ply"]
            existing_files.sort()
            if existing_files:
                self.counter = int(existing_files[-1].stem) + 1
                print(f"Found {len(existing_files)} existing files. Starting from {self.counter}.")
            else:
                self.counter = 0

    def write(self, point_cloud: Union[np.ndarray, o3d.geometry.PointCloud]) -> None:
        """Write a point cloud to a ``.ply`` file.

        Args:
            point_cloud (Union[np.ndarray, o3d.geometry.PointCloud]): The point cloud to export.
        """

        if isinstance(point_cloud, np.ndarray):
            o3d_point_cloud = o3d.geometry.PointCloud()
            o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud)
        elif isinstance(point_cloud, o3d.geometry.PointCloud):
            o3d_point_cloud = point_cloud
        else:
            raise ValueError(f"Unsupported point cloud type: {type(point_cloud)}. Supports np.ndarray and o3d.geometry.PointCloud.")

        file_path = self.log_dir / Path(f"{self.counter:06d}.ply")
        o3d.io.write_point_cloud(str(file_path), o3d_point_cloud)
        self.counter += 1


class SuppressOutput:
    def __init__(self, suppress_stdout=False, suppress_stderr=False):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr

    def __enter__(self):

        devnull = os.open(os.devnull, os.O_WRONLY)

        if self.suppress_stdout:
            self.saved_stdout = os.dup(1)
            os.dup2(devnull, 1)

        if self.suppress_stderr:
            self.saved_stderr = os.dup(2)
            os.dup2(devnull, 2)

        os.close(devnull)

    def __exit__(self, *args, **kwargs):
        if self.suppress_stdout:
            os.dup2(self.saved_stdout, 1)
            os.close(self.saved_stdout)

        if self.suppress_stderr:
            os.dup2(self.saved_stderr, 2)
            os.close(self.saved_stderr)
