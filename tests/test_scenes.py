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
    with lock.acquire(poll_interval=0.1):
        yield


def basic(env_cls, obs_enum, act_enum, render_enum):
    observation_types = [obs_type for obs_type in obs_enum]
    action_types = [action_type for action_type in act_enum]

    for obs_type in observation_types:
        for action_type in action_types:
            env = env_cls(observation_type=obs_type, render_mode=render_enum.HEADLESS, action_type=action_type)

            obs, info = env.reset()
            obs, rew, term, trunc, info = env.step(env.action_space.sample())
            env.close()


class TestReach:
    def test_basic(self, serial) -> None:
        from sofa_env.scenes.reach.reach_env import ReachEnv, RenderMode, ObservationType, ActionType

        basic(ReachEnv, ObservationType, ActionType, RenderMode)


class TestDeflectSpheres:
    def test_basic(self, serial) -> None:
        from sofa_env.scenes.deflect_spheres.deflect_spheres_env import DeflectSpheresEnv, RenderMode, ObservationType, ActionType

        basic(DeflectSpheresEnv, ObservationType, ActionType, RenderMode)


class TestSearchForPoint:
    def test_basic(self, serial) -> None:
        from sofa_env.scenes.search_for_point.search_for_point_env import SearchForPointEnv, RenderMode, ObservationType, ActionType

        basic(SearchForPointEnv, ObservationType, ActionType, RenderMode)


class TestTissueManipulation:
    def test_basic(self, serial) -> None:
        from sofa_env.scenes.tissue_manipulation.tissue_manipulation_env import TissueManipulationEnv, RenderMode, ObservationType, ActionType

        basic(TissueManipulationEnv, ObservationType, ActionType, RenderMode)


class TestPickAndPlace:
    def test_basic(self, serial) -> None:
        from sofa_env.scenes.pick_and_place.pick_and_place_env import PickAndPlaceEnv, RenderMode, ObservationType, ActionType

        basic(PickAndPlaceEnv, ObservationType, ActionType, RenderMode)


class TestGraspLiftTouch:
    def test_basic(self, serial) -> None:
        from sofa_env.scenes.grasp_lift_touch.grasp_lift_touch_env import GraspLiftTouchEnv, RenderMode, ObservationType, ActionType

        basic(GraspLiftTouchEnv, ObservationType, ActionType, RenderMode)


class TestRopeCutting:
    def test_basic(self, serial) -> None:
        from sofa_env.scenes.rope_cutting.rope_cutting_env import RopeCuttingEnv, RenderMode, ObservationType, ActionType

        basic(RopeCuttingEnv, ObservationType, ActionType, RenderMode)


class TestPrecisionCutting:
    def test_basic(self, serial) -> None:
        from sofa_env.scenes.precision_cutting.precision_cutting_env import PrecisionCuttingEnv, RenderMode, ObservationType, ActionType

        basic(PrecisionCuttingEnv, ObservationType, ActionType, RenderMode)


class TestTissueDissection:
    def test_basic(self, serial) -> None:
        from sofa_env.scenes.tissue_dissection.tissue_dissection_env import TissueDissectionEnv, RenderMode, ObservationType, ActionType

        basic(TissueDissectionEnv, ObservationType, ActionType, RenderMode)


class TestThreadInHole:
    def test_basic(self, serial) -> None:
        from sofa_env.scenes.thread_in_hole.thread_in_hole_env import ThreadInHoleEnv, RenderMode, ObservationType, ActionType

        basic(ThreadInHoleEnv, ObservationType, ActionType, RenderMode)


class TestRopeThreading:
    def test_basic(self, serial) -> None:
        from sofa_env.scenes.rope_threading.rope_threading_env import RopeThreadingEnv, RenderMode, ObservationType, ActionType

        basic(RopeThreadingEnv, ObservationType, ActionType, RenderMode)


class TestLigatingLoop:
    def test_basic(self, serial) -> None:
        from sofa_env.scenes.ligating_loop.ligating_loop_env import LigatingLoopEnv, RenderMode, ObservationType, ActionType

        basic(LigatingLoopEnv, ObservationType, ActionType, RenderMode)


class TestTissueRetraction:
    def test_basic(self, serial) -> None:
        from sofa_env.scenes.tissue_retraction.tissue_retraction_env import TissueRetractionEnv, RenderMode, ObservationType, ActionType

        basic(TissueRetractionEnv, ObservationType, ActionType, RenderMode)


class TestBimanualTissueManipulation:
    def test_basic(self, serial) -> None:
        from sofa_env.scenes.bimanual_tissue_manipulation.bimanual_tissue_manipulation_env import BimanualTissueManipulationEnv, RenderMode, ObservationType, ActionType

        basic(BimanualTissueManipulationEnv, ObservationType, ActionType, RenderMode)
