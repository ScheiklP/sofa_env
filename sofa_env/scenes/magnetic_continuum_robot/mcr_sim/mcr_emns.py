import numpy as np
from mag_manip import mag_manip


class EMNS:
    """
    A class used to build an eMNS object.

    :param name: The name of the eMNS object
    :type name: str
    :param calibration_path: The path to the eMNS calibartion file
    :type name: str
    """

    def __init__(
        self,
        name="emns",
        calibration_path="../calib/Navion_2_Calibration_24-02-2020.yaml",
    ):

        self.name = name
        self.calibration_path = calibration_path
        self.forward_model = mag_manip.ForwardModelMPEM()
        self.forward_model.setCalibrationFile(calibration_path)

    def currents_to_field(
        self,
        currents=np.array([0.0, 0.0, 0.0]),
        position=np.array([0.0, 0.0, 0.0]),
    ):
        """
        Apply forward model to compute the magnetic field at a given position.
        """

        bg_jac = self.forward_model.getFieldActuationMatrix(position)
        field = bg_jac.dot(currents)

        return field

    def field_to_currents(self, field=np.array([0.0, 0.0, 0.0]), position=np.array([0.0, 0.0, 0.0])):
        """
        Apply backward model to compute the currents needed to generate a
        magnetic field at a given position.
        """

        bg_jac = self.forward_model.getFieldActuationMatrix(position)
        currents = np.linalg.inv(bg_jac).dot(field)

        return currents
