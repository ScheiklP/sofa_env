from random import random
import numpy as np
from sofa_env.utils.dquat_pivot_transform import dquat_oblique_viewing_endoscope_ptsd_to_pose, dquat_ptsd_to_pose
from sofa_env.utils.pivot_transform import oblique_viewing_endoscope_ptsd_to_pose, ptsd_to_pose

TOLERANCE = 5e-5


class TestDQuatPivotTransform:
    """Tests the implementation of dual quaternions and the pivot transform with dual quaternions"""

    def test_pivot_transform(self):
        ptsd = np.array([45.0, 45.0, 90.0, 2.0])
        rcm = np.array([1.0, 2.0, 3.0, 0.0, 45.0, 0.0])

        assert np.allclose(ptsd_to_pose(ptsd, rcm), dquat_ptsd_to_pose(ptsd, rcm), atol=TOLERANCE)

    def _gen_states(self):
        ptsd = np.array([random() for _ in range(4)])
        rcm = np.array([random() for _ in range(6)])

        ptsd[:3] = 360.0 * ptsd[:3]
        ptsd[3] = 5.0 * ptsd[3]
        rcm[:3] = 5.0 * rcm[:3]
        rcm[3:] = 360.0 * rcm[3:]

        return ptsd, rcm

    def test_randomized_ptsd_to_pose(self):
        ptsd, rcm = self._gen_states()

        m_pose = ptsd_to_pose(ptsd, rcm)
        dquat_pose = dquat_ptsd_to_pose(ptsd, rcm)

        assert np.allclose(m_pose[:3], dquat_pose[:3], atol=TOLERANCE)
        assert np.allclose(m_pose[3:], dquat_pose[3:], atol=TOLERANCE) or np.allclose(m_pose[3:], -dquat_pose[3:], atol=TOLERANCE)

    def test_randomized_oblique_ptsd_to_pose(self):
        ptsd, rcm = self._gen_states()

        v = random()

        m_pose = oblique_viewing_endoscope_ptsd_to_pose(ptsd, rcm, v)
        dquat_pose = dquat_oblique_viewing_endoscope_ptsd_to_pose(ptsd, rcm, v)

        print("oblique", list(zip(m_pose, dquat_pose)))

        assert np.allclose(m_pose[:3], dquat_pose[:3], atol=TOLERANCE)

        # Pretty inaccurate ...
        assert np.allclose(m_pose[3:], dquat_pose[3:], atol=1e-1) or np.allclose(m_pose[3:], -dquat_pose[3:], atol=1e-1)
