import Sofa
import Sofa.Core
import gymnasium.spaces as spaces
import numpy as np
import pytest

from pathlib import Path
from sofa_env.base import SofaEnv, RenderMode
from typing import Optional, Tuple, Union

from sofa_env.utils.camera import world_to_pixel_coordinates

THIS_FILE = Path(__file__)


def createScene(root: Sofa.Core.Node, image_shape=(None, None)):
    root.addObject("RequiredPlugin", pluginName="Sofa.Component.IO.Mesh", name="Sofa.Component.IO.Mesh")
    root.addObject("RequiredPlugin", pluginName="Sofa.GL.Component.Rendering3D", name="Sofa.GL.Component.Rendering3D")
    root.addObject("RequiredPlugin", pluginName="Sofa.GL.Component.Shader", name="Sofa.GL.Component.Shader")
    root.addObject("RequiredPlugin", pluginName="Sofa.Component.ODESolver.Backward", name="Sofa.Component.ODESolver.Backward")
    root.addObject("RequiredPlugin", pluginName="Sofa.Component.LinearSolver.Iterative", name="Sofa.Component.LinearSolver.Iterative")

    root.addObject("VisualStyle", displayFlags="showAll")
    root.addObject("EulerImplicitSolver")
    root.addObject("CGLinearSolver", iterations=200, tolerance=1e-09, threshold=1e-09)

    floor = root.addChild("floor")
    floor_loader = floor.addObject("MeshOBJLoader", filename="mesh/floorFlat.obj", handleSeams=True)
    floor.addObject("OglModel", name="VisualModel", src=floor_loader.getLinkPath(), color="red")

    cube = root.addChild("cube")
    cube_loader = cube.addObject("MeshOBJLoader", filename="mesh/cube_low_res.obj", handleSeams=True, translation=[-5.0, 0.0, -5.0], scale=0.1)
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


class TransformationEnv(SofaEnv):
    def __init__(
        self,
        scene_path: Union[str, Path] = THIS_FILE,
        time_step: float = 0.1,
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
        self.observation_space = spaces.Box(low=0, high=255, shape=self.create_scene_kwargs["image_shape"] + (3,), dtype=np.uint8)

    def _do_action(self, action: np.ndarray) -> None:
        pass

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        rgb_observation = super().step(action)
        return rgb_observation, 1.0, False, {}

    def reset(self) -> np.ndarray:
        return super().reset()

    def get_pixel_coordinates(self, position: np.ndarray) -> Tuple[int, int]:
        pixel_position_row, pixel_position_col = world_to_pixel_coordinates(position, self.scene_creation_result["camera"])

        return pixel_position_row, pixel_position_col


@pytest.fixture(scope="module")
def transform_env() -> TransformationEnv:
    env = TransformationEnv(
        create_scene_kwargs={"image_shape": (800, 800)},
    )
    env.reset()

    return env


class TestTransformationEnv:
    def test_find_cube(self, transform_env: TransformationEnv) -> None:
        row, col = transform_env.get_pixel_coordinates(np.array([-5.0, 0.0, -5.0]))
        # numbers were created and checked manually
        assert row == 79
        assert col == 78

    def test_find_center(self, transform_env: TransformationEnv) -> None:
        row, col = transform_env.get_pixel_coordinates(np.array([0.0, 0.0, 0.0]))
        assert row in (399, 400, 401)
        assert col in (399, 400, 401)
