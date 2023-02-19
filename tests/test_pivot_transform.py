import json
import numpy as np
import pytest

from pathlib import Path
from sofa_env.utils.pivot_transform import ptsd_to_pose, generate_ptsd_to_pose

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng()


@pytest.fixture(scope="module")
def test_cases():
    with open(HERE / "pivot_transform_test_cases.json") as f:
        test_cases = json.load(f)
    for case in test_cases:
        case["rcm_pose"] = np.array(case["rcm_pose"])
        case["ptsd"] = np.array(case["ptsd"])
        case["7dpose"] = np.array(case["7dpose"])
    return test_cases


class TestPyPivotTransform:
    def test_correctness(self, test_cases):
        for case in test_cases:
            rcm_pose = case["rcm_pose"]
            ptsd = case["ptsd"]
            pose = case["7dpose"]
            assert np.allclose(pose, ptsd_to_pose(ptsd, rcm_pose), atol=5e-5)

    def test_generation(self, test_cases, rng):
        # because generating jitted functions is very slow, only run 20 random
        # test cases
        for index in rng.choice(len(test_cases), size=20, replace=False):
            case = test_cases[index]
            rcm_pose = case["rcm_pose"]
            ptsd = case["ptsd"]
            pose = case["7dpose"]

            transform = generate_ptsd_to_pose(rcm_pose)

            assert np.allclose(pose, transform(ptsd), atol=5e-5)
