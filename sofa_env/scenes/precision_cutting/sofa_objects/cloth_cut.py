from typing import Callable, Literal

import numpy as np
import numpy.typing as npt

from sofa_env.scenes.precision_cutting.sofa_objects.grid_path_projection import GridPathProjection, LengthUnit, RunningAxis


def make_cut(
    path: Callable[[npt.ArrayLike], npt.ArrayLike],
    position: float = 0.5,
    depth: float = 0.9,
    rotation: float = 0.0,
) -> GridPathProjection:
    """Create a cutting path from the specified function that is suitable for projection onto a ``Cloth``'s y axis.

    Args:
        path (Callable[[npt.ArrayLike], npt.ArrayLike]): The function specifying the cutting path.
        position (float): A value within the interval [0,1] that specifies the position of the cut relative to the ``Cloth``'s x axis.
        depth (float): A value within the interval [0,1] that specifies depth of the cut relative to the ``Cloth``'s y axis.
        rotation (float): The angle in degrees to rotate the cutting path about when projected.
    """
    assert 0.0 <= position and position <= 1.0
    assert 0.0 <= depth and depth <= 1.0
    return GridPathProjection(
        f=path,
        translation=(position, 0.0),
        rotation=rotation,
        y_range=(0.0, depth),
        running_axis=RunningAxis.Y,
        translation_length_unit=LengthUnit.RELATIVE,
        range_length_unit=LengthUnit.RELATIVE,
    )


def interpolated_cut(
    waypoints: np.ndarray,
    interpolation_kind: Literal["linear", "cubic"] = "linear",
    position: float = 0.5,
    depth: float = 1.0,
    rotation: float = 0.0,
) -> GridPathProjection:
    """Create a cutting path from the specified waypoints that is suitable for projection onto a ``Cloth``'s y axis.

    Args:
        waypoints (np.ndarray): A Nx2 array of [u, v] coordinates of waypoints.
        interpolation_kind (Literal["linear", "cubic"]): The kind of interpolation to use, either linear or cubic spline.
        position (float): A value within the interval [0,1] that specifies the position of the cut relative to the ``Cloth``'s x axis.
        depth (float): A value within the interval [0,1] that specifies depth of the cut relative to the ``Cloth``'s y axis.
        rotation (float): The angle in degrees to rotate the cutting path about when projected.
    """
    assert 0.0 <= position and position <= 1.0
    assert 0.0 <= depth and depth <= 1.0
    return GridPathProjection.from_points(
        waypoints=waypoints,
        interpolation_kind=interpolation_kind,
        translation=(position, 0.0),
        rotation=rotation,
        y_range=(0.0, depth),
        running_axis=RunningAxis.Y,
        translation_length_unit=LengthUnit.RELATIVE,
        range_length_unit=LengthUnit.RELATIVE,
    )


def linear_cut(
    slope: float = 0.5,
    position: float = 0.5,
    depth: float = 0.9,
) -> GridPathProjection:
    """Create a linear cutting path suitable for projection onto a ``Cloth``'s y axis.

    Args:
        slope (float): The slope of the line.
        position (float): A value within the interval [0,1] that specifies the position of the cut relative to the ``Cloth``'s x axis.
        depth (float): A value within the interval [0,1] that specifies depth of the cut relative to the ``Cloth``'s y axis.
    """
    # in order for the the position argument to work as expected the line's intercept is zero
    line = lambda x: slope * x
    return make_cut(path=line, position=position, depth=depth)


def sine_cut(
    amplitude: float = 10.0,
    frequency: float = 1.0,
    position: float = 0.5,
    depth: float = 1.0,
    rotation: float = 0.0,
) -> GridPathProjection:
    """Create a sinusiodal cutting path suitable for projection onto a ``Cloth``'s y axis.

    Args:
        amplitude (float): The peak deviation of the function from zero.
        frequency (float): The number of oscillation cycles per unit length.
        position (float): A value within the interval [0,1] that specifies the position of the cut relative to the ``Cloth``'s x axis.
        depth (float): A value within the interval [0,1] that specifies depth of the cut relative to the ``Cloth``'s y axis.
        rotation (float): The angle in degrees to rotate the sine wave about when projected.
    """
    sine = lambda x: amplitude * np.sin(2.0 * np.pi * frequency * x)
    return make_cut(path=sine, position=position, depth=depth, rotation=rotation)
