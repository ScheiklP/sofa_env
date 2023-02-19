import numpy as np
import open3d as o3d


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
