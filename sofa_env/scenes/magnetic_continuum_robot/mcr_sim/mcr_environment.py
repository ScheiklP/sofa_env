import Sofa
from scipy.spatial.transform import Rotation as R
import numpy as np


class Environment(Sofa.Core.Controller):
    """
    A class used to define environment objects and to  build the SOFA collision
    and visualiztion model of the environment.

    :param root_node: The sofa root node
    :type root_node:
    :param environment_stl: The path to the environment STL mesh file
    :type environment_stl: str
    :param name: The name of the environment
    :type name: str
    :param T_env_sim: The transform defining the pose of the environment with respect to simulation frame [x, y, z, qx, qy, qz, qw]
    :type T_env_sim:
    :param flip_normals: A flag that defines the direction of the surface of the mesh object
    :type flip_normals: bool
    :param color: The color of environment used for visualization [r, g, b, alpha]
    :type color: list[float]
    :param `*args`: The variable arguments are passed to the SofaCoreController
    :param `**kwargs`: The keyword arguments arguments are passed to the SofaCoreController
    """

    def __init__(
        self,
        root_node,
        environment_stl,
        name="environment",
        T_env_sim=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        flip_normals=False,
        color=[1.0, 0.0, 0.0, 0.3],
        *args,
        **kwargs,
    ):

        # These are needed (and the normal way to override from a python class)
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root_node = root_node
        self.environment_stl = environment_stl
        self.name_env = name

        self.color = color

        self.T_env_sim = T_env_sim
        r = R.from_quat(self.T_env_sim[3:7])
        rot_env_sim = (r.as_euler("xyz", degrees=True)).tolist()

        # collision model environment
        self.CollisionModel = root_node.addChild("CollisionModel")
        self.CollisionModel.addObject("RequiredPlugin", name="Sofa.Component.IO.Mesh")
        self.CollisionModel.addObject(
            "MeshSTLLoader",
            filename=self.environment_stl,
            flipNormals=flip_normals,
            triangulate=True,
            name="meshLoader",
            rotation=rot_env_sim,
            translation=self.T_env_sim[0:2],
            scale="0.001",
        )
        self.CollisionModel.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Constant")
        self.CollisionModel.addObject("Mesh", position="@meshLoader.position", triangles="@meshLoader.triangles", drawTriangles="0")
        self.CollisionModel.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")
        self.MO = self.CollisionModel.addObject("MechanicalObject", position=[0, 0, 0], scale=1, name="DOFs1")
        self.CollisionModel.addObject("RequiredPlugin", name="Sofa.Component.Collision.Geometry")
        self.CollisionModel.addObject("TriangleCollisionModel", moving=False, simulated=False)
        self.CollisionModel.addObject("LineCollisionModel", moving=False, simulated=False)
        self.CollisionModel.addObject("PointCollisionModel", moving=False, simulated=False)

        # # visual model environment
        VisuModel = self.CollisionModel.addChild("VisuModel")
        VisuModel.addObject("RequiredPlugin", name="Sofa.GL.Component.Rendering3D")
        VisuModel.addObject("OglModel", name="VisualOgl_model", src="@../meshLoader", color=self.color)

    def get_vessel_tree_positions(self):
        positions_vessel_tree = self.MO.position.array()
        return positions_vessel_tree
