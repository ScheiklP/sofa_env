import numpy as np
from numba import njit
from numpy.random import Generator


@njit
def farthest_point_sampling(points: np.ndarray, num_samples: int, rng: Generator) -> np.ndarray:
    starting_point_index = rng.integers(low=0, high=points.shape[0])
    sampled_indices = np.zeros(num_samples, dtype=np.uint16)
    sampled_points = np.zeros((num_samples, 3), dtype=np.float32)
    num_points = points.shape[0]

    sampled_points[0] = points[starting_point_index]

    dists = np.zeros((num_samples, num_points))
    min_dists = np.zeros(num_points)

    for i in range(num_samples - 1):
        for j, point in enumerate(points):
            dists[i, j] = np.linalg.norm(point - sampled_points[i])
            min_dists[j] = np.min(dists[: i + 1, j])

        farthest_point_index = np.argmax(min_dists)
        sampled_indices[i + 1] = farthest_point_index
        sampled_points[i + 1] = points[farthest_point_index]

    return sampled_indices
