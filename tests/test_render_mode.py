import Sofa
import Sofa.Core

import gymnasium.spaces as spaces
import numpy as np
import pytest

import contextlib
import os
import filelock

from pathlib import Path
from sofa_env.base import SofaEnv, RenderMode
from typing import Optional, Tuple, Union

from sofa_env.sofa_templates.scene_header import add_plugins


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


THIS_FILE = Path(__file__)
PLUGIN_LIST = [
    "Sofa.Component.IO.Mesh",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.LinearSolver.Iterative",
    "Sofa.GL.Component.Rendering3D",
    "Sofa.GL.Component.Shader",
]


def createScene(root: Sofa.Core.Node, image_shape=(600, 600)):
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


class RenderEnv(SofaEnv):
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

        return rgb_observation, 1.0, False, {}

    def reset(self) -> Optional[np.ndarray]:
        return super().reset()


class TestRenderMode:
    def test_human_render(self, serial) -> None:
        env = RenderEnv(render_mode=RenderMode.HUMAN)
        reset_obs = env.reset()
        obs, _, _, _ = env.step(env.action_space.sample())

        assert reset_obs is not None, f"{reset_obs=}"
        assert obs is not None, f"{obs=}"

        assert reset_obs.shape == (600, 600, 3), f"{reset_obs.shape=}"
        assert obs.shape == (600, 600, 3), f"{obs.shape=}"

    def test_headless_render(self, serial) -> None:
        env = RenderEnv(render_mode=RenderMode.HEADLESS)
        reset_obs = env.reset()
        obs, _, _, _ = env.step(env.action_space.sample())

        assert reset_obs is not None, f"{reset_obs=}"
        assert obs is not None, f"{obs=}"

        assert reset_obs.shape == (600, 600, 3), f"{reset_obs.shape=}"
        assert obs.shape == (600, 600, 3), f"{obs.shape=}"

    def test_remote_render(self, serial) -> None:
        env = RenderEnv(render_mode=RenderMode.REMOTE)
        reset_obs = env.reset()
        obs, _, _, _ = env.step(env.action_space.sample())

        assert reset_obs is not None, f"{reset_obs=}"
        assert obs is not None, f"{obs=}"

        assert reset_obs.shape == (600, 600, 3), f"{reset_obs.shape=}"
        assert obs.shape == (600, 600, 3), f"{obs.shape=}"

    def test_no_render(self, serial) -> None:
        env = RenderEnv(render_mode=RenderMode.NONE)
        reset_obs = env.reset()
        obs, _, _, _ = env.step(env.action_space.sample())

        assert reset_obs is None, f"{reset_obs=}"
        assert obs is None, f"{obs=}"

    def test_manual_render(self, serial) -> None:
        env = RenderEnv(render_mode=RenderMode.MANUAL)
        reset_obs = env.reset()
        obs, _, _, _ = env.step(env.action_space.sample())

        assert reset_obs is None, f"{reset_obs=}"
        assert obs is None, f"{obs=}"

        obs = env.update_rgb_buffer()
        assert obs is not None, f"{obs=}"
        assert obs.shape == (600, 600, 3), f"{obs.shape=}"

        obs = env.update_rgb_buffer_remote()
        assert obs is not None, f"{obs=}"
        assert obs.shape == (600, 600, 3), f"{obs.shape=}"

        obs = env.get_depth_from_pyglet()
        assert obs is not None, f"{obs=}"
        assert obs.shape == (600, 600, 1), f"{obs.shape=}"
        assert obs.dtype == np.uint8, f"{obs.dtype=}"

        obs = env.get_depth_from_open_gl()
        assert obs is not None, f"{obs=}"
        assert obs.shape == (600, 600, 1), f"{obs.shape=}"
        assert obs.dtype == np.float32, f"{obs.dtype=}"
