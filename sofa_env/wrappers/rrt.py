import time
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from typing import Any, Callable, List, Union, Dict, Tuple
from abc import ABC, abstractmethod
from pathlib import Path

from sofa_env.base import RenderMode
from sofa_env.scenes.deflect_spheres.deflect_spheres_env import ActionType, DeflectSpheresEnv, Mode, ObservationType
from sofa_env.scenes.deflect_spheres.sofa_objects.post import State
from sofa_env.sofa_templates.rigid import PivotizedRigidObject
from sofa_env.utils.dquat_inverse_pivot_transform import pose_to_ptsd
from sofa_env.utils.math_helper import point_rotation_by_quaternion
from sofa_env.utils.motion_planning import create_linear_motion_plan, create_ptsd_motion_plan
from sofa_env.utils.rrt import PTSDRRTGraph, RRTGraph


class RRTWrapperMode(ABC):
    """Mode for the RRTWrapper.

    Defines the motion planning mode and the corresponding functions needed for RRT in different spaces.

    Args:
        motion_plan_func (Callable): Function that creates a complete motion plan from a list of target points.
    """

    def __init__(self, motion_plan_func: Callable) -> None:
        super().__init__()
        self.motion_plan_func = motion_plan_func

    @abstractmethod
    def get_instrument_instruction(self, instrument: PivotizedRigidObject, target: np.ndarray, instrument_offset: np.ndarray = np.zeros(3)) -> np.ndarray:
        """Returns the instruction for the instrument to reach the target position.

        The instrument instruction defines how the instrument's PTSD state should change to reach the target position.

        Args:
            instrument (PivotizedRigidObject): Instrument to be moved.
            target (np.ndarray): Target position in the instrument's space.
            instrument_offset (np.ndarray): Cartesian [XYZ] offset between the instrument tip and the point of reference for planning.

        Returns:
            (np.ndarray): List of actions (PTSD deltas) for the instrument to reach the target position.
        """
        pass

    @abstractmethod
    def create_rrt_graph(self, instrument: PivotizedRigidObject, offset: np.ndarray) -> RRTGraph:
        """Defines how the RRT graph (and which one) is createdself.

        Args:
            offset (np.ndarray): Cartesian [XYZ] offset between the instrument tip and the point of reference for planning.

        Returs:
            (RRTGraph): The RRT graph needed for the mode.
        """
        pass

    def target_points_to_motion_plan(self, target_points: List[np.ndarray], time_step: float) -> List[np.ndarray]:
        """Returns a complete motion plan from a list of target points (not necessarily in cartesian space, i.e. PTSD space).

        Args:
            target_points (List[np.ndarray]): List of target points in a certain space depending on the mode.

        Returns:
            (List[np.ndarray]): List of target points and intermediate points in the same space as the target points.
        """

        if len(target_points) < 2:
            raise RuntimeError("At least two points are needed to create a motion plan")
        motion_plan = []
        from_point = target_points[0]
        for target_point in target_points[1:]:
            motion_plan.extend(self.motion_plan_func(target_point, from_point, time_step, 4))
            from_point = target_point

        return motion_plan


class CartesianMode(RRTWrapperMode):
    """Mode for the RRTWrapper that uses cartesian coordinates."""

    def __init__(self):
        super().__init__(create_linear_motion_plan)

    def create_rrt_graph(self, instrument: PivotizedRigidObject, offset: np.ndarray = np.zeros(3)) -> RRTGraph:
        return RRTGraph(instrument, instrument_offset=offset)

    def get_instrument_instruction(self, instrument: PivotizedRigidObject, target: np.ndarray, instrument_offset: np.ndarray = np.zeros(3)) -> np.ndarray:
        """Returns the instruction for the instrument to reach the target position. Therefore, the difference between the current state and the target
        state is calculated to reach the target position.

        The instrument instruction defines how the instrument's PTSD state should change to reach the target position.

        Args:
            instrument (PivotizedRigidObject): Instrument to be moved.
            target (np.ndarray): Target position in the instrument's space.
            instrument_offset (np.ndarray): Cartesian [XYZ] offset between the instrument tip and the point of reference for planning.

        Returns:
            (np.ndarray): List of actions (PTSD deltas) for the instrument to reach the target position.
        """
        current_ptsd = instrument.ptsd_state
        orientation = instrument.get_pose()[3:]
        target_ptsd = pose_to_ptsd(np.hstack((target, orientation)), instrument.remote_center_of_motion, link_offset=instrument_offset)
        return target_ptsd - current_ptsd


class PTSDMode(RRTWrapperMode):
    """Mode for the RRTWrapper that uses PTSD coordinates."""

    def __init__(self):
        super().__init__(create_ptsd_motion_plan)

    def create_rrt_graph(self, instrument: PivotizedRigidObject, offset: np.ndarray = np.zeros(3)) -> RRTGraph:
        return PTSDRRTGraph(instrument, instrument_offset=offset)

    def get_instrument_instruction(self, instrument: PivotizedRigidObject, target: np.ndarray, instrument_offset: np.ndarray = np.zeros(3)) -> np.ndarray:
        """Returns the instruction for the instrument to reach the target position. In this case, this is just the difference between
        the current state and the target state.

        The instrument instruction defines how the instrument's PTSD state should change to reach the target position.

        Args:
            instrument (PivotizedRigidObject): Instrument to be moved.
            target (np.ndarray): Target position in the instrument's space.
            instrument_offset (np.ndarray): Cartesian [XYZ] offset between the instrument tip and the point of reference for planning.

        Returns:
            (np.ndarray): List of actions (PTSD deltas) for the instrument to reach the target position.
        """
        return target - instrument.ptsd_state


def render_color_to_material_string(color: Tuple[float, float, float], type: str = "tree") -> str:
    """Generates a SOFA material string from an RGB triplet.

    Args:
        color (Tuple[float, float, float]): RGB value in [0, 1].
        type (str): Either "tree" or "path".
    """
    if type == "tree":
        return f"Tree Diffuse 1 {color[0]} {color[1]} {color[2]} 0.6 Ambient 0 0 0 0 1 Specular 0 0 0 0 1 Emissive 1 {color[0]} {color[1]} {color[2]} 0.75 Shininess 1 1000"
    elif type == "path":
        return f"Path Diffuse 1 {color[0]} {color[1]} {color[2]} 0.8 Ambient 0 0 0 0 1 Specular 0 0 0 0 1 Emissive 1 {color[0]} {color[1]} {color[2]} 0.9 Shininess 1 1000"
    else:
        raise ValueError(f"[ERROR] Unknown material type {type}.")


class RRTMotionPlanningWrapper(gym.Wrapper):
    """Motion planning wrapper for SOFA simulation environments that finds a collision free path from the current to a target position.

    Notes:
        The createScene function of the SofaEnv should return a dict with all objects ...

        ``position_containers`` are objects where ``object.position.array()`` is valid (e.g. ``MechanicalObject``).
        ``triangle_containers`` are objects where ``object.triangles.array()`` is valid (e.g. ``OglModel``).

        Example:
            >>> sofa_objects["motion_planning"] = {}
            >>> sofa_objects["motion_planning"]["collision_objects"]["triangles"]["position_containers"] = []
            >>> sofa_objects["motion_planning"]["collision_objects"]["triangles"]["triangle_containers"] = []
            >>> sofa_objects["motion_planning"]["collision_objects"]["spheres"]["position_containers"] = []
            >>> sofa_objects["motion_planning"]["collision_objects"]["spheres"]["radii"] = []
            >>> sofa_objects["motion_planning"]["instruments"] = {}
            >>> for obj in objects:
                    sofa_objects["motion_planning"]["collision_objects"]["triangles"]["position_containers"].append(objects[obj].visual_model_node.OglModel)
                    sofa_objects["motion_planning"]["collision_objects"]["triangles"]["triangle_containers"].append(objects[obj].visual_model_node.OglModel)
            >>> sofa_objects["motion_planning"]["instruments"]["cauter"] = instrument_1
            >>> sofa_objects["motion_planning"]["instruments"]["gripper"] = instrument_2
            >>> return sofa_objects

    Args:
        env (gymnasium.Env): The environment to wrap.
        config (dict): Configuration dict for the motion planning wrapper with the following keys:
            instruments: List of instrument names.
            render_color: Dict of RGB colors for the instruments. Contains a numpy array of shape (2, 3) for each instrument name.
                The first row is the color for the tree and the second row is the color for the path.
        visualize_rrt (bool): Whether to render the RRT.
    """

    def __init__(self, env: gym.Env, config: dict, visualize_rrt: bool = False):
        super().__init__(env)
        # Parse instruments
        colors = config.get("render_color", {})
        self.render_colors = {name: colors.get(name, np.random.rand(3)) for name in config["instruments"]}
        if isinstance(env.action_space, spaces.Dict):
            self.sample_action = env.action_space.sample()
            for key in self.sample_action:
                self.sample_action[key][:] = 0.0
            self.instrument_handler: Callable[[dict[str, PivotizedRigidObject]], Any] = lambda x: x
        elif isinstance(env.action_space, spaces.Box) and len(config["instruments"]) == 1:
            if len(env.action_space.sample()) < 4:
                raise RuntimeError("Action space must be at least 4-dimensional")
            self.instrument_handler: Callable[[dict[str, PivotizedRigidObject]], Any] = lambda x: list(x.values())[0]
        else:
            raise NotImplementedError("Please implement Dict action space if your env contains more than one instrument to plan for")

        # Parse offsets
        self.instruments_offsets = config.get("offsets", {})

        self.visualize_rrt = visualize_rrt

        self.nodes = {}

    def reset(self, **kwargs) -> Union[np.ndarray, dict]:
        reset_observation = self.env.reset(**kwargs)
        self.instruments = self.env.scene_creation_result["motion_planning"]["instruments"]

        spheres = self.env.scene_creation_result["motion_planning"]["collision_objects"]["spheres"]
        self.sphere_centers = np.array([sc.position.array()[0, :3] for sc in spheres["position_containers"]])
        self.sphere_radii = np.array(spheres["radii"])
        # meshes
        triangles = self.env.scene_creation_result["motion_planning"]["collision_objects"]["triangles"]["triangle_containers"]
        positions = self.env.scene_creation_result["motion_planning"]["collision_objects"]["triangles"]["position_containers"]
        self.meshes = [positions[i][triangles[i], :] for i, _ in enumerate(positions)]

        self.rrt_graphs = {}

        self.paths = {}
        self.motion_plans = {}

        return reset_observation

    def step(self, instruments_targets: Dict[str, np.ndarray]) -> Tuple:
        """Step function  of the wrapper.

        Args:
            instruments_targets (Dict[str, np.ndarray]): A dictionary of instrument names and target positions.

        Returns: the result of the wrapped environment's step function.
        """
        env_instructions = self.sample_action.copy()
        if isinstance(env_instructions, dict):
            for instrument_name in instruments_targets:
                instrument = self.instruments[instrument_name]
                env_instructions[instrument_name] = self.mode.get_instrument_instruction(
                    instrument,
                    instruments_targets[instrument_name],
                    instrument_offset=self.instruments_offsets.get(instrument_name, np.zeros(3)),
                )
        else:
            instrument = self.instruments.values()[0]
            env_instructions = self.mode.get_instrument_instruction(
                instrument,
                instruments_targets.values()[0],
                self.instruments_offsets.get(instrument_name, np.zeros(3)),
            )

        return self.env.step(self.instrument_handler(env_instructions))

    def plan_motion(self, config: dict) -> dict:
        """Creates a motion plan for the given instrument, mode, and target position.

        Note:
            ``execute_motion_plan`` should be called before calling this function again. Otherwise, the motion plan could lead into
            another instrument.

        Warning:
            Currently only implemented for instruments with ``SphereCollisionModels`` (see ``_plan_motion``).

        Args:
            config (dict): A dictionary containing the following keys:
                Required:
                - instrument_name (str): The name of the instrument to plan for.
                - target (np.ndarray): The target position of the instrument.
                - bounds (List): A list of Cartesian bounds for sampling. [[xmin, ymin, zmin], [xmax, ymax, zmax]]

                Optional:
                - mode (str): The motion planning mode. Can be "cartesian" or "ptsd". Defaults to "cartesian".
                - meshes (list[np.ndarray]): A list of meshes to use for collision checking. Defaults to the meshes of the scene if not specified.
                - sphere_centers (np.ndarray): The centers of the spheres to use for collision checking. Defaults to the spheres of the scene if not specified.
                - sphere_radii (np.ndarray): The radii of the spheres to use for collision checking. Defaults to the spheres of the scene if not specified.
                - resolution (float): The resolution of the RRT. Defaults to 5.0.
                - iterations (int): The number of iterations to run the RRT for. Defaults to 250.
                - steer_length (float): The steer length of the RRT. Defaults to 10.0.

        Returns:
            dict: A dictionary containing the following keys:
                - target_points (np.ndarray): The target points of the instrument in the created motion plan.
                - instrument_name (str): The name of the instrument to plan for. Is required for execution of the motion plan (see ``execute_motion_plan``)
        """

        instrument_name = config["instrument_name"]
        target = config["target"]
        mode = config.get("mode", "cartesian")
        self.meshes = config.get("meshes", self.meshes)
        self.sphere_centers = config.get("sphere_centers", self.sphere_centers)
        self.sphere_radii = config.get("sphere_radii", self.sphere_radii)
        resolution = config.get("resolution", 5.0)
        iterations = config.get("iterations", 250)
        steer_length = config.get("steer_length", 10.0)
        return {"target_points": self._plan_motion(target, instrument_name, mode=mode, bounds=config["bounds"], resolution=resolution, iterations=iterations, steer_length=steer_length), "instrument_name": instrument_name}

    def execute_motion_plan(self, motion_plan: dict) -> None:
        """Executes a given motion plan and draws the RRT graph.

        Args:
            motion_plan (dict): A motion plan as returned by ``plan_motion``. It should therefore contain the following keys:
                - target_points (np.ndarray): The target points of the instrument in the created motion plan.
                - instrument_name (str): The name of the instrument to plan for.
        """

        instrument_name = motion_plan["instrument_name"]

        if self.visualize_rrt:
            self.draw_rrt(instrument_name)
        for point in motion_plan["target_points"]:
            self.step({instrument_name: point})

    def _plan_motion(self, target: np.ndarray, instrument_name: str, bounds: List, mode: str = "cartesian", resolution: float = 5.0, iterations: int = 250, steer_length: float = 10.0) -> List:
        """Calculates a motion plan for a specific instrument to a given target. Only meant for internal use. For external use, use ``plan_motion``.

        Warning:
            Currently only implemented for instruments with ``SphereCollisionModels``.
            ``instrument.collision_model_node[1].SphereCollisionModel``

        Args:
            target (np.ndarray): The target position of the instrument.
            instrument_name (str): The name of the instrument to plan for.
            bounds (List): A list of Cartesian bounds for sampling. [[xmin, ymin, zmin], [xmax, ymax, zmax]]
            mode (str): The mode to use for motion planning. Can be either "cartesian" or "ptsd". Defaults to "cartesian".
            resolution (float): The resolution of the RRT. Defaults to 5.0.
            iterations (int): The number of iterations to run the RRT for. Defaults to 250.
            steer_length (float): The steer length of the RRT. Defaults to 10.0.
        """
        if mode == "cartesian":
            self.mode = CartesianMode()
        elif mode == "ptsd":
            self.mode = PTSDMode()
        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented.")

        self.rrt_graphs[instrument_name] = self.mode.create_rrt_graph(self.instruments[instrument_name], offset=self.instruments_offsets.get(instrument_name, np.zeros(3)))

        other_instruments = [value for key, value in self.instruments.items() if key != instrument_name]
        instrument_sphere_positions = np.concatenate([instrument.collision_model_node[0].MechanicalObject.position.array() for instrument in other_instruments] + [instrument.collision_model_node[1].MechanicalObject.position.array() for instrument in other_instruments])
        instrument_sphere_radii = np.concatenate([instrument.collision_model_node[0].SphereCollisionModel.listRadius.array() for instrument in other_instruments] + [instrument.collision_model_node[1].SphereCollisionModel.listRadius.array() for instrument in other_instruments])
        sphere_centers = np.concatenate((self.sphere_centers, instrument_sphere_positions))
        sphere_radii = np.concatenate((self.sphere_radii, instrument_sphere_radii))
        print(f"Started motion planning for instrument '{instrument_name}'")
        t = time.perf_counter()
        # calculate motion plan and return it
        rrt = self.rrt_graphs[instrument_name]
        ioff = point_rotation_by_quaternion(self.instruments_offsets.get(instrument_name, np.zeros(3)), self.instruments[instrument_name].get_pose()[3:])
        start = self.instruments[instrument_name].get_pose()[:3] + ioff
        rrt.generate_sample_points(start, target, self.meshes, sphere_centers, sphere_radii, bounds=bounds, resolution=resolution, iterations=iterations, steer_length=steer_length)
        path = rrt.get_smoothed_path(start, target, self.meshes, sphere_centers, sphere_radii)
        self.paths[instrument_name] = path
        target_points = [rrt.vertices[x] for x in path]
        motion_plan = self.mode.target_points_to_motion_plan(target_points, self.env.time_step)
        print(f"Completed motion plan for instrument '{instrument_name}' after {(time.perf_counter() - t):.2f} seconds")

        return motion_plan

    def draw_rrt(self, instrument_name: str) -> None:
        """Draws the RRT graph for a specific instrument.

        Only meant for internal use. Called in ``plan_motion`` if ``visualize_rrt=True``.

        Args:
            instrument_name (str): The name of the instrument to draw the RRT graph for.
        """
        if self.nodes.get(instrument_name, None) is not None:
            self.nodes[instrument_name].OglModel.isEnabled = False
        self.path_visual = self.env._sofa_root_node.addChild("beam_visual")
        path = self.paths[instrument_name]
        path_edges = [(path[i], path[i + 1]) for i in range(len(self.paths[instrument_name]) - 1)]
        non_path_edges = [item for item in self.rrt_graphs[instrument_name].get_edges() if item not in path_edges]
        tree_color = render_color_to_material_string(self.render_colors[instrument_name], type="tree")
        path_color = render_color_to_material_string(self.render_colors[instrument_name], type="path")

        self.path_visual.addObject(
            "OglModel",
            edges=non_path_edges,
            position=self.rrt_graphs[instrument_name].get_vertices(),
            lineWidth=5.0,
            material=tree_color,
        )
        self.path_visual.addObject(
            "OglModel",
            edges=path_edges,
            position=self.rrt_graphs[instrument_name].get_vertices(),
            lineWidth=20.0,
            material=path_color,
        )
        self.nodes[instrument_name] = self.path_visual
        self.path_visual.init()
        self.env.sofa_simulation.initVisual(self.path_visual)


def createScene(*args, **kwargs):
    """Example extension for DeflectSpheresEnv's createScene function"""
    from sofa_env.scenes.deflect_spheres.scene_description import createScene as original_createScene

    scene_creation_result = original_createScene(*args, **kwargs)

    scene_creation_result["motion_planning"] = {
        "instruments": {
            "right_cauter": scene_creation_result["right_cauter"],
            "left_cauter": scene_creation_result["left_cauter"],
        },
        "collision_objects": {
            "triangles": {
                "position_containers": [],
                "triangle_containers": []
                # "position_containers": [post.visual_node.OglModel.position.array() for post in posts],
                # "triangle_containers": [post.visual_node.OglModel.triangles.array() for post in posts],
            },
            "spheres": {
                "position_containers": [post.collision_node.MechanicalObject for post in scene_creation_result["posts"]],
                "radii": [post.sphere_radius for post in scene_creation_result["posts"]],
            },
        },
    }

    return scene_creation_result


if __name__ == "__main__":
    """Example use of the RRTMotionPlanningWrapper."""

    this_file = Path(__file__).resolve()

    image_shape = (800, 800)

    env = DeflectSpheresEnv(
        scene_path=this_file,
        observation_type=ObservationType.STATE,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.CONTINUOUS,
        image_shape=image_shape,
        frame_skip=1,
        time_step=1 / 30,
        settle_steps=10,
        single_agent=False,
        individual_agents=True,
        mode=Mode.WITHOUT_REPLACEMENT,
    )

    env = RRTMotionPlanningWrapper(
        env,
        {
            "instruments": ["left_cauter", "right_cauter"],
            "offsets": {
                "left_cauter": np.array([0.0, 0.0, 14.4]),
                "right_cauter": np.array([0.0, 0.0, 14.4]),
            },
            "render_color": {"left_cauter": (1.0, 0.0, 0.0)},
        },
        visualize_rrt=True,
    )

    reset_obs = env.reset()

    spheres_centers = env.sphere_centers
    spheres_radii = env.sphere_radii

    # In order to be able to let the instrument move to a target sphere, it must not be seen as an obstacle. Otherwise,
    # the RRT would not be able to find a collision free path. The same applies to the previous target of the instrument
    # because at the start, the instrument WILL collide with it.
    left_prev_index, right_prev_index = -1, -1
    for _ in range(5):
        index = env.env.active_post_index
        sc, sr = [], []
        if env.env.posts[index].state == State.ACTIVE_LEFT:
            for i in range(len(spheres_centers)):
                if i != index and i != left_prev_index:
                    sc.append(spheres_centers[i])
                    sr.append(spheres_radii[i])
            sc, sr = np.array(sc), np.array(sr)
            motion_plan = env.plan_motion(
                {
                    "instrument_name": "left_cauter",
                    "target": spheres_centers[index],
                    "mode": "cartesian",
                    "meshes": env.meshes,
                    "sphere_centers": sc,
                    "sphere_radii": sr,
                    "bounds": [[-50, -50, 0], [50, 50, 60]],
                }
            )
            env.execute_motion_plan(motion_plan)
            left_prev_index = index
        else:
            for i in range(len(spheres_centers)):
                if i != index and i != right_prev_index:
                    sc.append(spheres_centers[i])
                    sr.append(spheres_radii[i])
            sc, sr = np.array(sc), np.array(sr)
            motion_plan = env.plan_motion(
                {
                    "instrument_name": "right_cauter",
                    "target": spheres_centers[index],
                    "mode": "ptsd",
                    "meshes": env.meshes,
                    "sphere_centers": sc,
                    "sphere_radii": sr,
                    "bounds": [[-50, -50, 0], [50, 50, 60]],
                }
            )
            env.execute_motion_plan(motion_plan)
            right_prev_index = index
