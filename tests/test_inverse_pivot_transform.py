import numpy as np

from sofa_env.utils.dquat_inverse_pivot_transform import spin_quaternion_close_to, pose_to_ptsd
from sofa_env.utils.math_helper import multiply_quaternions, rotated_z_axis
from sofa_env.utils.pivot_transform import ptsd_to_pose

TOLERANCE = 5e-5


class TestInversePivotTransform:
    def test_spin_quaternion_close_to(self):
        q = np.random.rand(4)
        q = q / np.linalg.norm(q)

        # 1) Test if the quaternion will not be spun unnecessarily
        assert np.allclose(spin_quaternion_close_to(q, q), q)

        random_spin = np.random.random() * 2.0 - 1.0
        q_spin = np.empty(4)
        q_spin[:3] = np.sqrt(1.0 - random_spin ** 2) * rotated_z_axis(q)
        q_spin[3] = random_spin

        correction = spin_quaternion_close_to(multiply_quaternions(q_spin, q), q)
        # 2) Test if it will correct a spun quaternion to its former spin
        assert np.allclose(correction, q, atol=TOLERANCE) or np.allclose(correction, -q, atol=TOLERANCE)

    def test_simple(self):
        rcm_pose = np.array([-150.0, -200.0, 0.0, 0.0, 90.0, 180.0])
        target_point = np.array([-50.0, 15.0, -25.0])

        ptsd = pose_to_ptsd(np.hstack((target_point, np.array([0.5, 0.5, 0.5, 0.5]))), rcm_pose)
        pose = ptsd_to_pose(ptsd, rcm_pose)

        ptsd2 = pose_to_ptsd(pose, rcm_pose)
        result = ptsd_to_pose(ptsd2, rcm_pose)

        assert np.allclose(pose, result, atol=TOLERANCE)
