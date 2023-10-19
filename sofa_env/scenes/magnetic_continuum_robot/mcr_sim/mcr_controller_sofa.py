import Sofa
import Sofa.Core
import numpy as np

from sofa_env.scenes.magnetic_continuum_robot.mcr_sim import mcr_mag_controller
from scipy.spatial.transform import Rotation as R

# Increment field angle in rad
DFIELD_ANGLE = 3.0 * np.pi / 180.0


class ControllerSofa(Sofa.Core.Controller):
    """
    A class that interfaces with the SOFA controller and with the magnetic
    field controller.
    On keyboard events, the desired magnetic field and insertion inputs are
    sent to the controllers.

    :param root_node: The sofa root node
    :param e_mns: The object defining the eMNS
    :param instrument: The object defining the instrument
    :param environment: The object defining the environment
    :param T_sim_mns: The transform defining the pose of the sofa_sim frame center in Navion frame [x, y, z, qx, qy, qz, qw]
    :type T_sim_mns: list[float]
    :param T_sim_mns: The inital magnetic field direction and magnitude (T)
    :type T_sim_mns: ndarray
    """

    def __init__(
        self,
        root_node,
        e_mns,
        instrument,
        T_sim_mns,
        mag_field_init=np.array([0.01, 0.01, 0.0]),
        *args,
        **kwargs,
    ):
        # These are needed (and the normal way to override from a python class)
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root_node = root_node
        self.e_mns = e_mns
        self.instrument = instrument
        self.T_sim_mns = T_sim_mns
        self.mag_field_init = mag_field_init

        self.dfield_angle = 0.0

        self.mag_controller = mcr_mag_controller.MagController(
            root_node=self.root_node,
            e_mns=self.e_mns,
            instrument=self.instrument,
            T_sim_mns=self.T_sim_mns,
        )
        self.root_node.addObject(self.mag_controller)

        self.mag_controller.field_des = self.mag_field_init
        self.invalid_action = False

    def onKeypressedEvent(self, event):
        """Send magnetic field and insertion inputs when keys are pressed."""
        key = event["key"]
        # J key : z-rotation +
        if ord(key) == 76:
            self.rotateZ(-1)

        # L key : z-rotation -
        if ord(key) == 74:
            self.rotateZ(1)

        # I key : x-rotation +
        if ord(key) == 73:
            self.rotateX(-1)

        # K key : x-rotation -
        if ord(key) == 75:
            self.rotateX(1)

    def rotateZ(self, val):
        r = R.from_rotvec(val * DFIELD_ANGLE * np.array([0, 0, 1]))
        self.mag_controller.field_des = r.apply(self.mag_controller.field_des)

    def rotateX(self, val):
        r = R.from_rotvec(val * DFIELD_ANGLE * np.array([1, 0, 0]))
        self.mag_controller.field_des = r.apply(self.mag_controller.field_des)

    def insertRetract(self, val):
        d_step = 0.0015
        irc_xtip = self._getXTipValue()
        if (float(irc_xtip) + (val * d_step)) > 0.51:
            self.invalid_action = True
        else:
            self.invalid_action = False
            self.instrument.IRC.xtip[0] += val * d_step

    def _getXTipValue(self):
        return self.instrument.IRC.xtip[0]

    def reset(self) -> None:
        """Reset magnetic field"""
        super().reset()
        self.mag_controller.field_des = self.mag_field_init
        self.dfield_angle = 0.0

        # Reset insertRetract state
        self.instrument.IRC.xtip[0] = 0.0
        self.invalid_action = False

    def get_mag_field_des(self):
        return self.mag_controller.field_des

    def get_pos_catheter(self, num_points):
        pos_catheter = ()
        factor = int(31 / num_points) + 1
        for i in range(num_points):
            pos_catheter = np.append(pos_catheter, self.instrument.MO.position.array()[i * factor][:3])
        return pos_catheter

    def get_pos_quat_catheter_tip(self):
        return self.instrument.MO.position.array()[32]
