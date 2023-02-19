import Sofa
import numpy as np
from splib3.numerics import Quat, Vec3
from scipy.spatial.transform import Rotation as R


class MagController(Sofa.Core.Controller):
    """
    A class that takes the desired magnetic field inputs and calculates the
    torque applied on the magnets of the magnetic instrument. The torques are
    applied to the SOFA mechanical model at every time step.

    :param e_mns: The object defining the eMNS
    :param instrument: The object defining the instrument
    :param T_sim_mns: The transform defining the pose of the sofa_sim frame center in Navion frame [x, y, z, qx, qy, qz, qw]
    :type T_sim_mns: list[float]
    :param field_des: The desired magnetic field (m)
    :type field_des: float
    :param `*args`: The variable arguments are passed to the SofaCoreController
    :param `**kwargs`: The keyword arguments arguments are passed to the SofaCoreController
    """

    def __init__(self, e_mns, instrument, T_sim_mns, field_des=np.array([0.0, 0.0, 0.0]), *args, **kwargs):

        # These are needed (and the normal way to override from a python class)
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.e_mns = e_mns
        self.instrument = instrument
        self.T_sim_mns = T_sim_mns
        self.field_des = field_des

        self.magnet_moment = instrument.magnets[0].dipole_moment
        self.BG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.num_nodes = len(self.instrument.index_mag)

        # Position of the entry point
        self.initPos = np.array([self.T_sim_mns[0], self.T_sim_mns[1], self.T_sim_mns[2]])

    def onAnimateBeginEvent(self, event):
        """
        Apply the torque on the magntic nodes given a desired field and
        the pose of the nodes.
        """

        self.num_nodes = len(self.instrument.MO.position)

        for i in range(0, len(self.instrument.index_mag)):

            pos = self.instrument.MO.position[self.num_nodes - self.instrument.index_mag[i] - 1]
            quat = Quat(pos[3], pos[4], pos[5], pos[6])

            # Update magnetic model with new pose of catheters
            actualPos = np.array([pos[0], pos[1], pos[2]]) + np.array([self.T_sim_mns[0], self.T_sim_mns[1], self.T_sim_mns[2]])  # pose of the tip in Navion frame

            currents = self.e_mns.field_to_currents(field=self.field_des, position=actualPos)
            field = self.e_mns.currents_to_field(currents=currents, position=actualPos)

            self.BG = field

            B = Vec3(self.BG[0] * self.magnet_moment, self.BG[1] * self.magnet_moment, self.BG[2] * self.magnet_moment)
            magnetic_field = B

            # torque on magnet
            r = R.from_quat(quat)
            X = r.apply([1.0, 0.0, 0.0])
            T = Vec3()
            T = T.cross(X, magnetic_field)

            # Update forces and torques
            self.instrument.CFF.forces[self.instrument.index_mag[i]][:] = [0, 0, 0, T[0], T[1], T[2]]

        # visualze magnetic field arrow in SOFA gui
        self.instrument.CFF_visu.force = [magnetic_field[0], magnetic_field[1], magnetic_field[2], 0, 0, 0]
