import numpy as np

from sofa_env.utils.dual_quaternion import axis_angle_to_dquat, dquat_apply, dquat_conjugate, dquat_prod, dquat_rotate_and_translate, dquat_translate, point_to_dquat

TOLERANCE = 5e-5


class TestDualQuaternionFunctions:
    def test_transformation(self):
        """Multiplies two quaternions, applies their product to a vector, and checks the results.

        Numbers from https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/other/dualQuaternion/index.htm
        """

        q1 = axis_angle_to_dquat(np.array([1.0, 0.0, 0.0]), np.pi)
        assert np.allclose(q1, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), atol=TOLERANCE)

        q2 = dquat_translate(np.array([4.0, 2.0, 6.0]))
        assert np.allclose(q2, np.array([0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 3.0, 0.0]), atol=TOLERANCE)

        q = dquat_prod(q1, q2)
        assert np.allclose(q, np.array([1.0, 0.0, 0.0, 0.0, 0.0, -3.0, 1.0, -2.0]), atol=TOLERANCE)  # i -2e -3ej +1ek

        v = point_to_dquat(np.array([3.0, 4.0, 5.0]))

        assert np.allclose(dquat_prod(q, v, dquat_conjugate(q)), np.array([0.0, 0.0, 0.0, 1.0, 7.0, -6.0, -11.0, 0.0]), atol=TOLERANCE)

    def test_rotate_and_translate(self):
        """Tests the function rotate_and_translate"""
        t = np.array([1.0, 2.0, 3.0])
        r = np.zeros(4)
        r[0] = np.sin(np.pi / 4.0)
        r[3] = np.cos(np.pi / 4.0)

        q = dquat_rotate_and_translate(r, t)
        v = np.array([2.0, 5.0, 1.0])
        assert np.allclose(dquat_apply(q, v), np.array([3.0, 1.0, 8.0]), atol=TOLERANCE)

        q_ = dquat_rotate_and_translate(r, t, translate_first=True)
        assert np.allclose(dquat_apply(q_, v), np.array([3.0, -4.0, 7.0]), atol=TOLERANCE)
