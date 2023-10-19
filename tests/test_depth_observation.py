import Sofa
import Sofa.Core

import gymnasium.spaces as spaces
import numpy as np
import pytest

from pathlib import Path
from sofa_env.base import SofaEnv, RenderMode
from typing import Optional, Tuple, Union

from sofa_env.sofa_templates.scene_header import add_plugins


THIS_FILE = Path(__file__)
PLUGIN_LIST = [
    "Sofa.Component.IO.Mesh",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.LinearSolver.Iterative",
    "Sofa.GL.Component.Rendering3D",
    "Sofa.GL.Component.Shader",
]


def createScene(root: Sofa.Core.Node, image_shape=(None, None)):
    add_plugins(root, PLUGIN_LIST)
    root.addObject("VisualStyle", displayFlags="showAll")
    root.addObject("EulerImplicitSolver")
    root.addObject("CGLinearSolver", iterations=200, tolerance=1e-09, threshold=1e-09)

    floor = root.addChild("floor")
    floor_loader = floor.addObject("MeshObjLoader", filename="mesh/floorFlat.obj", handleSeams=True)
    floor.addObject("OglModel", name="VisualModel", src=floor_loader.getLinkPath(), color="red")

    cube = root.addChild("cube")
    cube_loader = cube.addObject("MeshObjLoader", filename="mesh/cube_low_res.obj", handleSeams=True, translation=[2, 0, 1])
    cube.addObject("OglModel", name="VisualModel", src=cube_loader.getLinkPath(), color="green")

    # place light and a camera
    root.addObject("LightManager")
    root.addObject("DirectionalLight", direction=[0, 1, 0])
    camera = root.addObject(
        "InteractiveCamera",
        name="camera",
        position=[0, 15, 0],
        lookAt=[0, 0, 0],
        fieldOfView=45,
        zNear=0.63,
        zFar=55.69,
        widthViewport=image_shape[1],
        heightViewport=image_shape[0],
    )

    return {"root_node": root, "camera": camera}


class DepthEnv(SofaEnv):
    def __init__(
        self,
        scene_path: Union[str, Path] = THIS_FILE,
        time_step: float = 0.01,
        frame_skip: int = 1,
        render_mode: RenderMode = RenderMode.HEADLESS,
        create_scene_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__(
            scene_path,
            time_step=time_step,
            frame_skip=frame_skip,
            render_mode=render_mode,
            create_scene_kwargs=create_scene_kwargs,
        )

        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(600, 600, 3), dtype=np.uint8)

    def _do_action(self, action: np.ndarray) -> None:
        pass

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        rgb_observation = super().step(action)

        self._depth_observation = self.get_depth_from_open_gl()
        self._rgb_observation = rgb_observation

        return rgb_observation, 1.0, False, {}

    def reset(self) -> Optional[np.ndarray]:
        return super().reset()


@pytest.fixture(scope="module")
def depth_env() -> DepthEnv:
    env = DepthEnv(
        create_scene_kwargs={"image_shape": (600, 600)},
    )
    env.reset()
    env.step(env.action_space.sample())

    return env


class TestDepthEnv:
    def test_floor_distance(self, depth_env: DepthEnv) -> None:
        assert np.allclose(depth_env._depth_observation[0, 0], 15)

    def test_cube_distance(self, depth_env: DepthEnv) -> None:
        assert np.allclose(depth_env._depth_observation[400, 400], 11.5)
