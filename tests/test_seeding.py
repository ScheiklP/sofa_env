import numpy as np
import pytest

import contextlib
import os
import filelock


@pytest.fixture(scope="session")
def lock(tmp_path_factory):
    base_temp = tmp_path_factory.getbasetemp()
    lock_file = base_temp.parent / "serial.lock"
    yield filelock.FileLock(lock_file=str(lock_file))
    with contextlib.suppress(OSError):
        os.remove(path=lock_file)


@pytest.fixture()
def serial(lock):
    with lock.acquire(poll_intervall=0.1):
        yield


class TestSeeding:
    def logic(self, env) -> None:
        seeds = [123, 456, 789]
        for seed in seeds:
            env.seed(seed)
            org_reset_obs = env.reset()
            org_image = env.render(mode="rgb_array")
            for _ in range(5):
                for _ in range(10):
                    obs, _, _, _ = env.step(env.action_space.sample())
                    assert not np.allclose(obs, org_reset_obs, atol=1e-6, rtol=1e-6)
                env.seed(seed)
                reset_obs = env.reset()
                image = env.render(mode="rgb_array")
                assert np.allclose(reset_obs, org_reset_obs, atol=1e-6, rtol=1e-6)
                assert np.allclose(image, org_image, atol=1e-6, rtol=1e-6)

    def test_deflect_spheres(self, serial) -> None:
        from sofa_env.scenes.deflect_spheres.deflect_spheres_env import DeflectSpheresEnv, RenderMode, ObservationType, ActionType

        env = DeflectSpheresEnv(
            observation_type=ObservationType.STATE,
            render_mode=RenderMode.HEADLESS,
            action_type=ActionType.VELOCITY,
            image_shape=(124, 124),
            frame_skip=1,
            time_step=0.1,
            settle_steps=10,
            single_agent=False,
            individual_agents=True,
        )

        self.logic(env)

    def test_rope_threading(self, serial) -> None:
        from sofa_env.scenes.rope_threading.rope_threading_env import RopeThreadingEnv, RenderMode, ObservationType, ActionType

        eye_config = [
            (60, 10, 0, 90),
            (10, 10, 0, 90),
            (10, 60, 0, -45),
            (60, 60, 0, 90),
        ]

        env = RopeThreadingEnv(
            observation_type=ObservationType.STATE,
            render_mode=RenderMode.HEADLESS,
            action_type=ActionType.VELOCITY,
            image_shape=(124, 124),
            frame_skip=10,
            time_step=0.01,
            settle_steps=20,
            create_scene_kwargs={
                "eye_config": eye_config,
                "eye_reset_noise": {
                    "low": np.array([-20.0, -20.0, 0.0, -15]),
                    "high": np.array([20.0, 20.0, 0.0, 15]),
                },
                "randomize_gripper": True,
                "randomize_grasp_index": True,
                "start_grasped": True,
            },
            fraction_of_rope_to_pass=0.05,
            only_right_gripper=False,
            individual_agents=True,
            num_rope_tracking_points=10,
        )

        self.logic(env)
