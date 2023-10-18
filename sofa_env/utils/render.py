import numpy as np
import open3d as o3d
from typing import Sequence


class PointCloudRenderer:
    """Non-blocking rendering window for point clouds with Open3D.

    TODO:
        # If the env's RenderMode is not NONE -> Error in create_window
        # X Error of failed request:  BadAccess (attempt to access private resource denied)
        # Major opcode of failed request:  151 (GLX)
        # Minor opcode of failed request:  5 (X_GLXMakeCurrent)
        # Serial number of failed request:  157
        # Current serial number in output stream:  157
    """

    def __init__(self, initial_point_cloud: np.ndarray) -> None:
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(initial_point_cloud)
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window()
        self.visualizer.add_geometry(self.point_cloud)

    def render(self, point_cloud: np.ndarray) -> None:
        self.point_cloud.points = o3d.utility.Vector3dVector(point_cloud)
        self.visualizer.update_geometry(self.point_cloud)
        self.visualizer.poll_events()
        self.visualizer.update_renderer()


# Adapted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/base_vec_env.py
def tile_images(images_nhwc: Sequence[np.ndarray]) -> np.ndarray:  # pragma: no cover
    """Tile N images into one big PxQ image

    (P,Q) are chosen to be as close as possible, and if N is square, then P=Q.

    Args:
        images_nhwc (Sequence[np.ndarray]): (n_images, height, width, n_channels)

    Returns:
        np.ndarray: img_HWc, ndim=3

    """
    img_nhwc = np.asarray(images_nhwc)
    n_images, height, width, n_channels = img_nhwc.shape
    # new_height was named H before
    new_height = int(np.ceil(np.sqrt(n_images)))
    # new_width was named W before
    new_width = int(np.ceil(float(n_images) / new_height))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)])
    # img_HWhwc
    out_image = img_nhwc.reshape((new_height, new_width, height, width, n_channels))
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape((new_height * height, new_width * width, n_channels))
    return out_image
