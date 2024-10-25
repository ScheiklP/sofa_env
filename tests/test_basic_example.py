import Sofa
import Sofa.Core

import gymnasium.spaces as spaces
import numpy as np

from pathlib import Path
from sofa_env.base import SofaEnv, RenderMode
from typing import Optional, Tuple, Union

from sofa_env.sofa_templates.scene_header import add_plugins

THIS_FILE = Path(__file__)
HERE = THIS_FILE.parent

PLUGIN_LIST = [
    "Sofa.Component.IO.Mesh",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.LinearSolver.Iterative",
    "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.Mass",
    "Sofa.Component.Constraint.Projective",
    "Sofa.GL.Component.Rendering3D",
    "Sofa.GL.Component.Shader",
]


def createScene(root: Sofa.Core.Node, image_shape=(None, None)):
    """Example scene from the SofaPython3 repository."""

    add_plugins(root, PLUGIN_LIST)
    root.addObject("DefaultAnimationLoop")
    root.gravity = [0, -1.0, 0]
    root.addObject("VisualStyle", displayFlags="showAll")
    volume_mesh_loader = root.addObject("MeshGmshLoader", filename="mesh/liver.msh")
    surface_mesh_loader = root.addObject("MeshOBJLoader", filename="mesh/liver-smooth.obj")

    root.addObject("EulerImplicitSolver")
    root.addObject("CGLinearSolver", iterations=200, tolerance=1e-09, threshold=1e-09)

    liver = root.addChild("liver")

    liver.addObject("TetrahedronSetTopologyContainer", src=volume_mesh_loader.getLinkPath())
    liver.addObject("TetrahedronSetGeometryAlgorithms", template="Vec3d")
    liver.addObject("MechanicalObject", template="Vec3d", showObject=True, showObjectScale=3)

    liver.addObject("TetrahedronFEMForceField", youngModulus=1000, poissonRatio=0.4, method="large")
    liver.addObject("UniformMass", totalMass=10.0)
    liver.addObject("FixedProjectiveConstraint", indices=[2, 3, 50])

    visual = liver.addChild("visual")
    visual.addObject("OglModel", src=surface_mesh_loader.getLinkPath(), color="red")
    visual.addObject("BarycentricMapping")

    # place light and a camera
    root.addObject("LightManager")
    root.addObject("DirectionalLight", direction=[0, 1, 0])
    camera = root.addObject(
        "InteractiveCamera",
        name="camera",
        position=[0, 15, 0],
        lookAt=[0, 0, 0],
        distance=37,
        fieldOfView=45,
        zNear=0.63,
        zFar=55.69,
        widthViewport=image_shape[1],
        heightViewport=image_shape[0],
    )

    return {"root_node": root, "camera": camera}


class ExampleEnv(SofaEnv):
    def __init__(
        self,
        scene_path: Union[str, Path] = THIS_FILE,
        time_step: float = 0.01,
        frame_skip: int = 1,
        render_mode: RenderMode = RenderMode.HUMAN,
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


class TestExampleEnv:
    def test_rendered_env(self) -> None:
        env = ExampleEnv(
            create_scene_kwargs={"image_shape": (600, 600)},
        )

        old_obs = env.reset().copy()
        assert isinstance(old_obs, np.ndarray)

        assumed_first_obs = np.load(HERE / "basic_example_first_obs.npy")
        assert np.allclose(assumed_first_obs, old_obs)

        for _ in range(100):
            obs, _, _, _ = env.step(env.action_space.sample())

            assert isinstance(obs, np.ndarray)
            assert obs.shape == (600, 600, 3)
            assert not np.allclose(obs, old_obs)

            old_obs = np.copy(obs)
