from enum import Enum
from typing import Callable, Optional, Union, Tuple, Literal

import numpy as np
import numpy.typing as npt

from sofa_env.utils.math_helper import euler_to_rotation_matrix, cubic_interp1d


class RunningAxis(Enum):
    """Indicate which axis of the grid corresponds to the x axis of the path. Think 'rise [f(x)] over run [x]'."""

    X = "x"
    Y = "y"


class LengthUnit(Enum):
    ABSOLUTE = "abs"
    RELATIVE = "rel"


class GridPathProjection:
    """GridPathProjection

    This class may be used to project path samples {[u, f(u)]} to grid points {[x,y]} on a 2-dimensional ``"RegularGridTopology"``.

    Args:
        f (Callable[[npt.ArrayLike], npt.ArrayLike]): The function to project onto the grid.
        translation (Union[Tuple[float, float], np.ndarray]): The 2D coordinates [tx, ty] used to translate the (rotated) path waypoints. Whether the translation coordinates are absolute or relative with respect to the grid's size may be specified by the argument ``translation_length_unit``. If ``translation_length_unit = LengthUnit.RELATIVE`` then it is mandatory for ``translation[0], translation[1]`` to lay within the interval [0,1].
        rotation (float): The rotation angle in degrees used for the path projection.
        x_range (Optional[Tuple[float, float]]): If ``range_length_unit = "abs"`` this specifies the allowed coordinate range along the grid's x axis as follows: ``x_range[0] <= x <= x_range[1]``. If ``range_length_unit = LengthUnit.RELATIVE`` this specifies the allowed coordinate range along the grid's x axis as follows: ``x_range[0]*grid_min_x <= x <= x_range[1]*grid_max_x`` where ``x_range[0], x_range[1]`` in [0,1] is mandatory.
        y_range (Optional[Tuple[float, float]]): If ``range_length_unit = "abs"`` this specifies the allowed coordinate range along the grid's y axis as follows: ``y_range[0] <= y <= y_range[1]``. If ``range_length_unit = LengthUnit.RELATIVE`` this specifies the allowed coordinate range along the grid's y axis as follows: ``y_range[0]*grid_min_y <= y <= y_range[1]*grid_max_y`` where ``y_range[0], y_range[1]`` in [0,1] is mandatory.
        running_axis (RunningAxis): The axis of the grid that corresponds to the x axis of the path. By default a path sample [u,f(u)] corresponds to the grid point [u,f(u)]. Selecting the y axis as ``running_axis`` means that a path sample [u,f(u)] corresponds to the grid point [f(u),u]. Note that this is different from rotating the path, but may be replicated with ``rotation`` by modifying ``f`` beforehand.
        translation_length_unit (LengthUnit): Whether to interpret the ``translation`` as absolute or relative with respect to the grid's size.
        range_length_unit (LengthUnit): Whether to interpret ``x_range`` and ``y_range`` as absolute or relative with respect to the grid's size.
    """

    def __init__(
        self,
        f: Callable[[npt.ArrayLike], npt.ArrayLike],
        translation: Union[Tuple[float, float], np.ndarray] = (0.0, 0.0),
        rotation: float = 0.0,
        x_range: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        y_range: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        running_axis: RunningAxis = RunningAxis.X,
        translation_length_unit: LengthUnit = LengthUnit.ABSOLUTE,
        range_length_unit: LengthUnit = LengthUnit.ABSOLUTE,
    ) -> None:
        # vectorize path function
        self.f = np.vectorize(lambda x: f(x))
        self.running_axis = running_axis

        # path range limits
        def check_range(range, length_unit) -> Optional[np.ndarray]:
            if range is None:
                return None
            r = np.array(range)
            assert r.ndim == 1 and r.shape[0] == 2
            assert r[0] < r[1]
            if length_unit == LengthUnit.RELATIVE:
                assert 0.0 <= r[0] and r[1] <= 1.0
            return r

        self.range_length_unit = range_length_unit
        self.x_range = check_range(x_range, self.range_length_unit)
        self.y_range = check_range(y_range, self.range_length_unit)
        self.translation = np.array(translation)
        assert self.translation.ndim == 1 and self.translation.shape[0] == 2
        self.translation_length_unit = translation_length_unit
        if self.translation_length_unit == LengthUnit.RELATIVE:
            assert np.all(0.0 <= self.translation) and np.all(self.translation <= 1.0)
        self.rotation = rotation

        def make_transforms() -> Tuple[Callable, Callable]:
            R = euler_to_rotation_matrix(np.array([0.0, 0.0, self.rotation]))[:2, :2]
            R_inv = euler_to_rotation_matrix(np.array([0.0, 0.0, -self.rotation]))[:2, :2]

            def t(grid_extend: Optional[np.ndarray]) -> np.ndarray:
                if self.translation_length_unit == LengthUnit.RELATIVE:
                    assert isinstance(grid_extend, np.ndarray)
                    return -0.5 * grid_extend + self.translation * grid_extend
                else:
                    return self.translation

            def transform(v: np.ndarray, grid_extend: Optional[np.ndarray]) -> np.ndarray:
                return (R @ v.reshape(-1, 2, 1)).reshape(-1, 2) + t(grid_extend)

            def inverse(v: np.ndarray, grid_extend: Optional[np.ndarray]) -> np.ndarray:
                return (R_inv @ (v - t(grid_extend)).reshape(-1, 2, 1)).reshape(-1, 2)

            return transform, inverse

        self.path_to_grid, self.grid_to_path = make_transforms()

    @classmethod
    def from_points(
        cls,
        waypoints: np.ndarray,
        interpolation_kind: Literal["linear", "cubic"] = "linear",
        translation: Union[Tuple[float, float], np.ndarray] = (0.0, 0.0),
        rotation: float = 0.0,
        x_range: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        y_range: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        running_axis: RunningAxis = RunningAxis.X,
        translation_length_unit: LengthUnit = LengthUnit.ABSOLUTE,
        range_length_unit: LengthUnit = LengthUnit.ABSOLUTE,
    ) -> "GridPathProjection":
        """
        Create a ``GridPathProjection`` where the projected path is an interploation of the specified ``waypoints`` {[u,v]}.

        Args:
            waypoints (np.ndarray): A Nx2 array of [u, v] coordinates of waypoints.
            interpolation_kind (Literal["linear", "cubic"]): The kind of interpolation to use, either linear or cubic spline.
            translation (Union[Tuple[float, float], np.ndarray]): The 2D coordinates [tx, ty] used to translate the (rotated) path waypoints. Whether the translation coordinates are absolute or relative with respect to the grid's size may be specified by the argument ``translation_length_unit``. If ``translation_length_unit = LengthUnit.RELATIVE`` then it is mandatory for ``translation[0], translation[1]`` to lay within the interval [0,1].
            rotation (float): The rotation angle in degrees used for the path projection.
            x_range (Optional[Tuple[float, float]]): If ``range_length_unit = "abs"`` this specifies the allowed coordinate range along the grid's x axis as follows: ``x_range[0] <= x <= x_range[1]``. If ``range_length_unit = LengthUnit.RELATIVE`` this specifies the allowed coordinate range along the grid's x axis as follows: ``x_range[0]*grid_min_x <= x <= x_range[1]*grid_max_x`` where ``x_range[0], x_range[1]`` in [0,1] is mandatory.
            y_range (Optional[Tuple[float, float]]): If ``range_length_unit = "abs"`` this specifies the allowed coordinate range along the grid's y axis as follows: ``y_range[0] <= y <= y_range[1]``. If ``range_length_unit = LengthUnit.RELATIVE`` this specifies the allowed coordinate range along the grid's y axis as follows: ``y_range[0]*grid_min_y <= y <= y_range[1]*grid_max_y`` where ``y_range[0], y_range[1]`` in [0,1] is mandatory.
            running_axis (RunningAxis): The axis of the grid that corresponds to the x axis of the waypoints. By default a waypoint [u,v] corresponds to the grid point [u,v]. Selecting the y axis as ``running_axis`` means that a waypoint [u,v] corresponds to the grid point [v,u]. Note that this is different from rotating the path, but may be replicated with ``rotation`` by modifying ``waypoints`` beforehand.
            translation_length_unit (LengthUnit): Whether to interpret the ``translation`` as absolute or relative with respect to the grid's size.
            range_length_unit (LengthUnit): Whether to interpret ``x_range`` and ``y_range`` as absolute or relative with respect to the grid's size.
        """
        assert waypoints.ndim == 2 and waypoints.shape[1] == 2
        if interpolation_kind == "linear":
            f = lambda x: np.interp(x=x, xp=waypoints[:, 0], fp=waypoints[:, 1])
        elif interpolation_kind == "cubic":
            f = cubic_interp1d(x=waypoints[:, 0], y=waypoints[:, 1])
        else:
            raise ValueError(f"Unknown interpolation kind: {interpolation_kind}")

        return cls(
            f=f,
            translation=translation,
            rotation=rotation,
            x_range=x_range,
            y_range=y_range,
            running_axis=running_axis,
            translation_length_unit=translation_length_unit,
            range_length_unit=range_length_unit,
        )

    def get_triangles_on_path(
        self,
        grid_size: Tuple[float, float],
        grid_resolution: Tuple[int, int],
        num_samples: int,
        grid_vertices: np.ndarray,
        grid_triangles: np.ndarray,
        stroke_radius: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect the grid triangles, i.e. a list of vertex indices triplets, that each contain at least one grid vertex close to the path.

        Args:
            grid_size (Tuple[float, float]): The extend of the grid along the x and y axis.
            grid_resolution (Tuple[int, int]): The number of grid points along each axis.
            num_samples (int): The number of path waypoints to sample.
            grid_vertices (np.ndarray): A Nx2 or Nx3 array of ``[x, y[, z]]`` coordinates of grid vertices. This function ssumes that the grid spans along the x and y axes, i.e. the first two axes of the specified vertex array.
            grid_triangles (np.ndarray): A Mx3 list of ``[v_1, v_2, v_3]`` vertex indices forming triangles.
            stroke_radius (float): The maximum distance a grid vertex is allowed to have to a path waypoint, in order or this vertex to still be considered as lying on the path.
        Returns:
            triangles_on_path (np.ndarray): the grid triangles laying on the path.
            centerline_indices (np.ndarray): The indices of grid points that describe the centerline of the path (minimum distance to the projected path).
        """
        assert np.all(np.array(grid_size) > 0)
        assert np.all(np.array(grid_resolution)) > 0
        assert num_samples > 0
        assert stroke_radius > 0.0

        # determine waypoint bounds on grid
        def calc_axis_range(range: Optional[np.ndarray], extend: float) -> Tuple[float, float]:
            if range is None:
                return (-0.5 * extend, 0.5 * extend)
            if self.range_length_unit == LengthUnit.RELATIVE:
                return (-0.5 * extend + range[0] * extend, -0.5 * extend + range[1] * extend)
            else:
                return (max(range[0], -0.5 * extend), min(range[1], 0.5 * extend))

        grid_extend = np.array(grid_size)
        wx_min, wx_max = calc_axis_range(self.x_range, grid_extend[0])
        wy_min, wy_max = calc_axis_range(self.y_range, grid_extend[1])
        # axes sample ranges for path
        origin_path = self.grid_to_path(np.array([0, 0]), grid_extend).reshape(2)
        sample_ranges = np.array([origin_path[0] + np.array([wx_min, wx_max]), origin_path[1] + np.array([wy_min, wy_max])])

        # create path samples
        u, fu = (1, 0) if self.running_axis == RunningAxis.Y else (0, 1)
        path = np.empty((num_samples, 2))
        path[:, u] = np.linspace(start=sample_ranges[u, 0], stop=sample_ranges[u, 1], num=num_samples)
        path[:, fu] = self.f(path[:, u])
        waypoints = self.path_to_grid(path, grid_extend)
        wx, wy = waypoints[:, 0], waypoints[:, 1]
        waypoints = waypoints[np.where(np.logical_and(np.logical_and(wx_min <= wx, wx <= wx_max), np.logical_and(wy_min <= wy, wy <= wy_max)))]
        assert waypoints.shape[0] > 1

        # use supplied grid
        grid = grid_vertices[:, :2]
        # find all the vertices on the grid that are close to a waypoint
        vertex_indices_on_path = np.flatnonzero(np.any(np.linalg.norm(grid[:, np.newaxis] - waypoints, axis=2) <= stroke_radius, axis=1))

        distances_grid_to_waypoints = np.linalg.norm(grid[:, np.newaxis] - waypoints, axis=2)
        centerline_indices = np.unique(np.argmin(distances_grid_to_waypoints, axis=0))

        # filter the triangles containing vertices close to a waypoint
        triangles_on_path = grid_triangles[np.any(np.isin(grid_triangles, vertex_indices_on_path), axis=1)].copy()

        # assert no duplicates
        assert np.unique(np.sort(grid_triangles, axis=1), axis=0).shape == grid_triangles.shape
        assert np.unique(np.sort(triangles_on_path, axis=1), axis=0).shape == triangles_on_path.shape

        return triangles_on_path, centerline_indices
