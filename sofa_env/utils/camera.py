import numpy as np
from typing import Tuple

from sofa_env.utils.math_helper import rotated_z_axis

import Sofa.Core


def determine_look_at(camera_position: np.ndarray, camera_orientation: np.ndarray) -> np.ndarray:
    """Given position and orientation of a camera, returns the position in cartesian space the camara looks at.

    This is done by rotating the z axis with the cameras orientation and adding it to the camera's position.

    Args:
        camera_position (np.ndarray): XYZ position of the camera
        camera_orientation (np.ndarray): Camera orientation as quaternion

    Returns:
        look_at (np.ndarray): XYZ position of the point the camera looks at


    TODO:
        - determine the scale (0.1) with the absolute size of the camera position (to determine scale based on meters or millimeters) np.linalg.norm(camera_position) * transformed_z_vector
    """

    transformed_z_vector = rotated_z_axis(camera_orientation)
    look_at = np.array(camera_position) - 0.1 * transformed_z_vector

    return look_at


def world_to_pixel_coordinates(coordinates: np.ndarray, camera_object: Sofa.Core.Object) -> Tuple[int, int]:
    """Calculates the pixel coordinates of a Cartesian point in world coordinates.

    Args:
        coordinates (np.ndarray): XYZ position in world coordinates.
        camera_object (Sofa.Core.Object): ``"Camera"`` or ``"InteractiveCamera"`` SOFA object.

    Returns:
        row, column (int, int): Pixel coordinates of the XYZ position in reference to the top left corner of the screen.

    Notes:
        Returns pixel coordinates from the top left like a numpy array (row, column).

    Examples:
        >>> world_to_pixel_coordinates(np.array(0.0, 0.0, 4.0), sofa_camera)
        (400, 400)
    """

    homogeneous_coordinates = np.ones(4, dtype=np.float64)
    homogeneous_coordinates[:3] = coordinates

    screen_width_in_pixels = camera_object.widthViewport.value
    screen_height_in_pixels = camera_object.heightViewport.value

    P = camera_object.getOpenGLProjectionMatrix()
    P = np.asarray(P).reshape((4, 4)).transpose()
    MV = camera_object.getOpenGLModelViewMatrix()
    MV = np.asarray(MV).reshape((4, 4)).transpose()

    MVP = P @ MV

    q_point = MVP @ homogeneous_coordinates
    q_point /= q_point[3]

    point = np.zeros(3, dtype=np.float64)
    point[0] = q_point[0] / 2 + 0.5  # Transform from [-1,1] to [0,1]
    point[1] = q_point[1] / 2 + 0.5  # Transform from [-1,1] to [0,1]
    point[2] = 0  # Projection onto XY-plane

    pixel_position_x = int(point[0] * screen_width_in_pixels)
    pixel_position_y = int(point[1] * screen_height_in_pixels)

    row = screen_height_in_pixels - pixel_position_y
    column = pixel_position_x

    return row, column


def vertical_to_horizontal_fov(fov_vertical: float, width: int, height: int) -> float:
    """Calculates the horizontal field of view from a given vertical field of view and the image ratio.

    Args:
        fov_vertical (float): Vertical field of view of a camera in degrees.
        width (int): Horizontal render resolution in pixels.
        height (int): Vertical render resolution in pixels.

    Returns:
        fov_horizontal (float): Horizontal field of view of a camera in degrees.

    Notes:
        See: https://en.wikipedia.org/wiki/Field_of_view_in_video_games

    Examples:
        >>> vertical_to_horizontal(fov_vertical=12.0, width=12, height=3)
        45.0
    """

    fov_horizontal = 2 * np.arctan(np.tan(np.deg2rad(fov_vertical) / 2) * width / height)

    return np.rad2deg(fov_horizontal)


def get_focal_length(camera_object: Sofa.Core.Object, width: int, height: int) -> Tuple[float, float]:
    """Returns the focal length from a given ``sofa_camera``.

    Args:
        camera_object (Sofa.Core.Object): ``"Camera"`` or ``"InteractiveCamera"`` SOFA object.
        width (int): Horizontal render resolution in pixels.
        height (int): Vertical render resolution in pixels.

    Returns:
        fx (float): X-axis focal length.
        fy (float): Y-axis focal length.

    Notes:
        The camera object contains a projection matrix, from which the focal length is calculated.
        See: http://www.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche0092.html
    """

    fx = camera_object.projectionMatrix.array()[0] * width / 2.0
    fy = camera_object.projectionMatrix.array()[5] * height / 2.0

    return fx, fy
