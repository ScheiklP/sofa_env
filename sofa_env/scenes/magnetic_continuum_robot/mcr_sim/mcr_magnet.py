import numpy as np


class Magnet:
    """
    A class used to build a magnet object.

    :param length: The length of the magnet
    :type length: float
    :param outer_diam: The outer diameter of the magnet
    :type outer_diam: float
    :param inner_diam: The inner diameter of the magnet
    :type inner_diam: float
    :param remanence: The remanence of the magnet (T)
    :type remanence: float
    :param color: The color of instrument used for visualization [r, g, b, alpha]
    :type color: list[float]
    """

    def __init__(self, length, outer_diam, inner_diam, remanence, color=[0.2, 0.2, 0.2, 1.0]):

        self.length = length
        self.outer_diam = outer_diam
        self.inner_diam = inner_diam
        self.remanence = remanence

        self.color = color

        self.mu_0 = (4.0 * np.pi) * 1e-7

        self.volume = self.length * np.pi * ((self.outer_diam / 2.0) ** 2 - (self.inner_diam / 2.0) ** 2.0)

        # dipole moment
        self.dipole_moment = (1.0 / self.mu_0) * self.remanence * self.volume
