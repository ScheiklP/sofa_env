import numpy as np

from itertools import product
from typing import List, Callable, Dict, Tuple, Optional

from sofa_env.sofa_templates.rigid import PivotizedRigidObject
from sofa_env.utils.dquat_inverse_pivot_transform import pose_to_ptsd
from sofa_env.utils.dquat_pivot_transform import quat_ptsd_to_pose_with_offset
from sofa_env.utils.dual_quaternion import dquat_apply, dquat_mul, dquat_rotate_and_translate, dquat_translate
from sofa_env.utils.math_helper import euler_angles_to_quaternion, is_inside_mesh


def collision_free_ptsd(
    start_ptsd: np.ndarray,
    target_ptsd: np.ndarray,
    instrument: PivotizedRigidObject,
    instrument_offset: np.ndarray,
    instrument_points: List[np.ndarray],
    resolution: float,
    meshes: List[np.ndarray],
    spheres_centers: np.ndarray,
    spheres_radii: np.ndarray,
) -> bool:
    """Collision function for PTSD space.

    Args:
        start_ptsd (np.ndarray): Start configuration of the instrument
        target_ptsd (np.ndarray): Target configuration of the instrument
        instrument (PivotizedRigidObject): The instrument to check for collision
        instrument_offset (np.ndarray): Offset of the instrument (i.e. tip)
        instrument_points (List[np.ndarray]): Relevant points of the instrument, needed for collision checking
        resolution (float): Maximum distance of intermediate points between ``start_ptsd`` and ``target_ptsd``. A high value results in good
            performance but low accuracy.
        meshes (List[np.ndarray]): List of all meshes in the scene relevant for collision detection.
        spheres_centers (np.ndarray): Numpy array containing the centers of every sphere in the scene. Shape (X, 3)
        spheres_radii (np.ndarray): Radii of all spheres
        offset (np.ndarray): The instrument's offset

    Returns ``True`` if no collision was found between start and target ptsd configuration, ``False`` otherwise.
    """

    # Generate sample points for path
    rcm = instrument.remote_center_of_motion
    from_point = quat_ptsd_to_pose_with_offset(start_ptsd, instrument.remote_center_of_motion, instrument_offset)
    to_point = quat_ptsd_to_pose_with_offset(target_ptsd, instrument.remote_center_of_motion, instrument_offset)
    num_steps = int(np.floor(np.linalg.norm(to_point - from_point) / resolution))
    step_vec = target_ptsd - start_ptsd
    if not np.all(step_vec == 0):
        step_vec *= 1.0 / max(1.0, num_steps) / np.linalg.norm(step_vec)
    path_samples = [start_ptsd]
    for i in range(1, num_steps + 1):
        path_samples.append(start_ptsd + i * step_vec)
    path_samples.append(target_ptsd)

    for sample in path_samples:
        delta_ptsd = start_ptsd - sample
        delta_quat = euler_angles_to_quaternion(np.array([delta_ptsd[1], delta_ptsd[0], delta_ptsd[2]]))

        # Apply the difference on the sample points
        transformation_dquat = dquat_mul(dquat_translate(rcm[:3] + np.array([0.0, 0.0, delta_ptsd[3]])), dquat_rotate_and_translate(delta_quat, -rcm[:3], translate_first=True))
        transformed_points = np.array([dquat_apply(transformation_dquat, point) for point in instrument_points])

        # Check if the transformed points collide with an object
        point = quat_ptsd_to_pose_with_offset(sample, instrument.remote_center_of_motion, instrument_offset)[:3]
        is_in_a_mesh = any(map(lambda x: np.any(is_inside_mesh(x, transformed_points)), meshes))
        if is_in_a_mesh:
            return False
        is_in_sphere = any(map(lambda sphere: np.linalg.norm(point - sphere[0]) <= 3 * sphere[1], zip(spheres_centers, spheres_radii)))
        if is_in_sphere:
            return False

    return True


def collision_free_cartesian(
    from_point: np.ndarray,
    to_point: np.ndarray,
    instrument: PivotizedRigidObject,
    instrument_offset: np.ndarray,
    instrument_points: List[np.ndarray],
    resolution: float,
    meshes: List[np.ndarray],
    spheres_centers: np.ndarray,
    spheres_radii: np.ndarray,
) -> bool:
    """
    Collision function for cartesian space. Returns ``True`` if no collision was found between start and target position, ``False`` otherwise.

    Args:
        from_point (np.ndarray): Start configuration of the instrument
        to_point (np.ndarray): Target configuration of the instrument
        instrument (PivotizedRigidObject): The instrument to check for collision
        instrument_offset (np.ndarray): Offset of the instrument (i.e. tip)
        instrument_points (List[np.ndarray]): Relevant points of the instrument, needed for collision checking
        resolution (float): Maximum distance of intermediate points between ``from_point`` and ``to_point``. A high value results in good
            performance but low accuracy.
        meshes (List[np.ndarray]): List of all meshes in the scene relevant for collision detection.
        spheres_centers (np.ndarray): Numpy array containing the centers of every sphere in the scene. Shape (X, 3)
        spheres_radii (np.ndarray): Radii of all spheres
        offset (np.ndarray): The instrument's offset
    """
    # Generate sample points for path
    step_vec = to_point - from_point
    distance = np.linalg.norm(step_vec)
    num_steps = int(np.floor(distance / resolution))
    if not np.all(step_vec == 0):
        step_vec *= resolution / np.linalg.norm(step_vec)
    path_samples = [from_point]
    for i in range(1, num_steps + 1):
        path_samples.append(from_point + i * step_vec)
    path_samples.append(to_point)

    # Get initial configuration of the instrument
    rcm = instrument.remote_center_of_motion
    orientation = instrument.get_pose()[3:]
    initial_ptsd = instrument.ptsd_state

    if np.any(np.linalg.norm(np.array(path_samples) - spheres_centers[:, None], axis=-1) < spheres_radii[:, None]):
        return False

    # Get collision of sample points
    for sample in path_samples:
        delta_ptsd = pose_to_ptsd(np.hstack((sample, orientation)), rcm, link_offset=instrument_offset) - initial_ptsd
        delta_quat = euler_angles_to_quaternion(np.array([delta_ptsd[1], delta_ptsd[0], delta_ptsd[2]]))

        transformation_dquat = dquat_mul(dquat_translate(rcm[:3] + np.array([0.0, 0.0, delta_ptsd[3]])), dquat_rotate_and_translate(delta_quat, -rcm[:3], translate_first=True))
        transformed_points = np.array([dquat_apply(transformation_dquat, point) for point in instrument_points])
        is_in_a_mesh = any(map(lambda x: np.any(is_inside_mesh(x, transformed_points)), meshes))
        if is_in_a_mesh:
            return False
    return True


def sample_cartesian(bounds: List) -> np.ndarray:
    """Sample function for cartesian space

    Args:
        bounds (List): A list of Cartesian bounds for sampling. [[xmin, ymin, zmin], [xmax, ymax, zmax]]

    Returns:
        sample (np.ndarray): A sample in cartesian space within the given bounds
    """

    return np.random.uniform(*bounds)


def generate_sample_ptsd(bounds: List, rcm: np.ndarray = np.zeros(6), offset: np.ndarray = np.zeros(3)) -> Callable:
    """Returns a sample function for PTSD space.

    This function will generate PTSD samples that will reach at least every corner of
    the given bounds. However, some PTSD samples will result in cartesian coordinates outside that box.

    Args:
        bounds (np.ndarray): Cartesian vectors that define the cartesian space that the PTSD samples should cover
        rcm (np.ndarray): The remote center of motion of the instrument
        offset (np.ndarray): The instrument's offset
    """
    corners = [[bounds[tpl[i]][i] for i in range(3)] for tpl in product([0, 1], repeat=3)]
    ptsd_corners = [pose_to_ptsd(np.hstack((c, np.zeros(3), 1.0)), rcm, link_offset=offset) for c in corners]
    ptsd_min = [min([c[i] for c in ptsd_corners]) for i in range(4)]
    ptsd_max = [max([c[i] for c in ptsd_corners]) for i in range(4)]
    return lambda **_: np.random.uniform(ptsd_min, ptsd_max)


class RRTGraph:
    """RRT graph for motion planning in cartesian space.

    The RRT object can only plan for one instrument, so multiple instruments need just as many RRTs.
    For PTSD space motion planning, see ``PTSDRRTGraph``.
    """

    def __init__(
        self,
        instrument: PivotizedRigidObject,
        instrument_offset: np.ndarray = np.zeros(3),
        sample_function: Callable = sample_cartesian,
        collision_function: Callable = collision_free_cartesian,
    ):
        self.vertices: List[np.ndarray] = []
        self.edges: Dict[int, List[int]] = {}
        self.instrument = instrument
        self.instrument_points = np.concatenate(
            (
                instrument.collision_model_node[0].MechanicalObject.position.array(),
                instrument.collision_model_node[1].MechanicalObject.position.array(),
            ),
            axis=0,
        )
        self.instrument_offset = instrument_offset
        self.sample_function = sample_function
        self.collision_function = collision_function

    def get_vertices(self) -> List[np.ndarray]:
        return self.vertices

    def generate_sample_points(
        self,
        start: np.ndarray,
        target: np.ndarray,
        meshes: List[np.ndarray],
        spheres_centers: List[np.ndarray],
        spheres_radii: List[float],
        bounds: List,
        resolution: float = 5.0,
        iterations: int = 250,
        steer_length: float = 10.0,
    ):
        """Generates sample points for the RRT in cartesian space using its sample function.

        The ones that are inside of an obstacle will be deleted, so there won't be a guaranteed number
        of vertices for the graph after the function call.

        Args:
            start (np.ndarray): Start coordinates (normally the current position of the instrument)
            target (np.ndarray): Target coordinates for the instrument
            meshes (List[np.ndarray]): List of all meshes in the scene relevant for collision detection.
            spheres_centers (List[np.ndarray]): Numpy array containing the centers of every sphere in the scene. Shape (X, 3)
            spheres_radii (List[float]): Radii of all spheres
            bounds (List): A list of Cartesian bounds for sampling. [[xmin, ymin, zmin], [xmax, ymax, zmax]]
            resolution (float): Maximum distance of intermediate points between ``from_point`` and ``to_point``. A high value results in good
            performance but low accuracy.
            iterations (float): Number of sample points
            steer_length (float): The steering length (see ``_steer``)
        """
        target_connected = False
        self.vertices.append(start)
        for _ in range(iterations):
            sample = self.sample_function(bounds=bounds)
            nearest_index, nearest = self._get_nearest(sample)
            new_point = _steer(nearest, sample, steer_length)
            if self.collision_function(nearest, new_point, self.instrument, self.instrument_offset, self.instrument_points, resolution, meshes, spheres_centers, spheres_radii):
                self._add_edge(nearest_index, len(self.vertices))
                self.vertices.append(new_point)

        for vertex_index, vertex in sorted(enumerate(self.vertices), key=lambda x: np.linalg.norm(x[1] - target)):
            if self.collision_function(vertex, target, self.instrument, self.instrument_offset, self.instrument_points, resolution, meshes, spheres_centers, spheres_radii):
                self._add_edge(vertex_index, len(self.vertices))
                self.vertices.append(target)
                target_connected = True
                break
        if not target_connected:
            raise RuntimeError("Could not connect target to graph. You could try increasing the number of samples")

    def get_edges(self) -> List[Tuple[int, int]]:
        """Returns the edges of the RRT in form of a list of tuples of the vertices' indices.
        For example, if a RRT has only two vertices and one edge connecting the first vertex to the second one,
        the output of ``get_edges`` would be [(0, 1)], no matter their coordinates.
        """
        edges = []
        for start in self.edges:
            for to in self.edges[start]:
                edges.append((start, to))
        return edges

    def get_path(self, start: np.ndarray, target: np.ndarray) -> List[Tuple[int, int]]:
        """Returns the path from ``start`` to ``target`` as a list of tuples of the vertices' indices.
        Could raise a ``RuntimeError`` if start or target are not in the graph or if there is no path between them.

        Args:
            start (np.ndarray): Start coordinates (normally the current position of the instrument)
            target (np.ndarray): Target coordinates for the instrument
        """
        start_index = self._index_of(start)
        if start_index == -1:
            raise RuntimeError(f"Start node '{start} not found'")
        target_index = self._index_of(target)
        if target_index == -1:
            raise RuntimeError(f"Target node '{target}' not found")

        paths: List[List[int]] = [[start_index]]
        while paths:
            path = paths.pop()
            if path[-1] == target_index:
                return path
            for neighbor in sorted(self.edges[path[-1]], key=lambda x: -np.linalg.norm(x - target)):
                if neighbor not in path:
                    paths.append(path + [neighbor])
        raise RuntimeError("No path was found")

    def get_smoothed_path(self, start: np.ndarray, target: np.ndarray, meshes: List[np.ndarray], spheres_centers: np.ndarray, spheres_radii: np.ndarray) -> List[Tuple[int, int]]:
        """Returns the smoothed path from ``start`` to ``target`` as a list of tuples of the vertices' indices.
        See ``get_path`` for more information.

        Args:
            start (np.ndarray): Start coordinates (normally the current position of the instrument)
            target (np.ndarray): Target coordinates for the instrument
            meshes (List[np.ndarray]): List of all meshes in the scene relevant for collision detection.
            spheres_centers (List[np.ndarray]): Numpy array containing the centers of every sphere in the scene. Shape (X, 3)
            spheres_radii (List[float]): Radii of all spheres

        Returns:
            smoothed_path (List[Tuple[int, int]]): The smoothed path as a list of tuples of the vertices' indices.
        """
        path = self.get_path(start, target)
        # TODO why is this hardcoded? If not required, add it to the function signature and set a reasonable default value.
        # TODO add a boolean flag to RRTWrapper's __init__, to turn smoothing on/off. Add a NotImplementedError
        # TODO if smoothing is turned on, and add a TODO in the docstring that says that the smoothed path is not collision free.
        steps_left = 7
        while steps_left > 0:
            indices = np.random.randint(0, len(path), 2)
            i1, i2 = np.min(indices), np.max(indices)
            if len(path) > 2 and self.collision_function(self.vertices[i1], self.vertices[i2], self.instrument, self.instrument_offset, self.instrument_points, 10.0, meshes, spheres_centers, spheres_radii):
                path = path[: i1 + 1] + path[i2:]
                steps_left += 1
            else:
                steps_left -= 1
        return path

    def _index_of(self, point: np.ndarray) -> int:
        """Returns the index of the given point. If the point is not in the graph, -1 is returned."""
        for i, v in enumerate(self.get_vertices()):
            if np.allclose(v, point, atol=1e-4):
                return i
        return -1

    def _get_nearest(self, point: np.ndarray) -> tuple[int, np.ndarray]:
        """Returns the index and the coordinates of the nearest vertex to the given point."""
        dist = float("inf")
        nearest_index = 0
        for i, v in enumerate(self.vertices):
            d = np.linalg.norm(v - point)
            if d < dist:
                nearest_index = i
                dist = d

        return nearest_index, self.vertices[nearest_index]

    def _add_edge(self, s: int, t: int):
        """Adds an edge between the vertices with indices ``s`` and ``t``. Only meant to be used internally."""
        if s in self.edges:
            self.edges[s].append(t)
        else:
            self.edges[s] = [t]
        if t in self.edges:
            self.edges[t].append(s)
        else:
            self.edges[t] = [s]


def _steer(from_node: np.ndarray, to_node: np.ndarray, d: float) -> np.ndarray:
    """Implements the ``steer`` function for the RRT algorithm.

    Returns a point that has a distance of ``d`` from ``from_node`` and is on the line between ``from_node`` and ``to_node``.

    Args:
        from_node (np.ndarray): The starting point
        to_node (np.ndarray): The target point
        d (float): The distance between the returned point and ``from_node``
    """
    distance = np.linalg.norm(from_node - to_node)
    if distance == 0:
        return to_node
    return from_node + (d / distance) * (to_node - from_node)


class PTSDRRTGraph(RRTGraph):
    """Represents a RRT graph of points in the PTSD space.

    The PTSD space equivalent of the ``RRTGraph`` class.

    Args:
        instrument (PivotizedRigidObject): The instrument to be used for collision detection
        instrument_offset (np.ndarray): The offset of the instrument
        sample_function (Optional[Callable]): The function to be used for sampling points in the PTSD space. Defaults to ``sample_ptsd``
        collision_function (Optional[Callable]): The function to be used for collision detection. Defaults to ``collision_free_ptsd``
    """

    def __init__(
        self,
        instrument: PivotizedRigidObject,
        instrument_offset: np.ndarray = np.zeros(3),
        sample_function: Optional[Callable] = None,
        collision_function: Optional[Callable] = collision_free_ptsd,
    ):

        super().__init__(instrument, instrument_offset, sample_function, collision_function)

    def get_vertices(self) -> List[np.ndarray]:
        """Returns the vertices of the graph in the PTSD space. Note that the vertices are cartesian coordinates, not points in the PTSD space."""
        return [quat_ptsd_to_pose_with_offset(x, self.instrument.remote_center_of_motion, self.instrument_offset)[:3] for x in self.vertices]

    def generate_sample_points(
        self,
        start: np.ndarray,
        target: np.ndarray,
        meshes: List,
        spheres_centers: List[np.ndarray],
        spheres_radii: List[float],
        bounds: List,
        resolution: float = 5.0,
        iterations: int = 150,
        steer_length: float = 10.0,
    ):
        """Generates sample points for the RRT in PTSD space using its sample function.

        The ones that are inside of an obstacle will be deleted, so there won't be a guaranteed number
        of vertices for the graph after the function call.

        Args:
            start (np.ndarray): Start coordinates (normally the current position of the instrument)
            target (np.ndarray): Target coordinates for the instrument
            meshes (List[np.ndarray]): List of all meshes in the scene relevant for collision detection.
            spheres_centers (List[np.ndarray]): Numpy array containing the centers of every sphere in the scene. Shape (X, 3)
            spheres_radii (List[float]): Radii of all spheres
            bounds (List): A list of Cartesian bounds for sampling. [[xmin, ymin, zmin], [xmax, ymax, zmax]]
            resolution (float): Maximum distance of intermediate points between ``from_point`` and ``to_point``. A high value results in good
            performance but low accuracy.
            iterations (float): Number of sample points
            steer_length (float): The steering length (see ``_steer``)
        """

        if self.sample_function is None:
            self.sample_function = generate_sample_ptsd(rcm=self.instrument.remote_center_of_motion, offset=self.instrument_offset, bounds=bounds)

        s = pose_to_ptsd(np.hstack((start, np.zeros(3), 1.0)), self.instrument.remote_center_of_motion, link_offset=self.instrument_offset)
        t = pose_to_ptsd(np.hstack((target, np.zeros(3), 1.0)), self.instrument.remote_center_of_motion, link_offset=self.instrument_offset)
        return super().generate_sample_points(
            start=s,
            target=t,
            meshes=meshes,
            spheres_centers=spheres_centers,
            spheres_radii=spheres_radii,
            resolution=resolution,
            iterations=iterations,
            bounds=bounds,
            steer_length=steer_length,
        )
