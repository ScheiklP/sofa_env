from splib3.numerics import Quat, Vec3
import math
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple
from pathlib import Path
import numpy as np

import Sofa.Core
from sofa_env.scenes.magnetic_continuum_robot.mcr_sim import mcr_environment, mcr_instrument, mcr_emns, mcr_simulator, mcr_controller_sofa, mcr_magnet
from sofa_env.sofa_templates.camera import Camera, CAMERA_PLUGIN_LIST


HERE = Path(__file__).resolve().parent
PLUGIN_LIST = [
    "SofaPython3",
    "SoftRobots",
    "BeamAdapter",
] + CAMERA_PLUGIN_LIST


def createScene(
    root_node: Sofa.Core.Node,
    image_shape: Tuple[Optional[int], Optional[int]] = (None, None),
    debug_rendering: bool = False,
    positioning_camera: bool = False,
):
    """
    Creates the flat 2d scene of the MCREnv.

    Args:
        root_node (Sofa.Core.Node): SET BY ENV. The simulation tree's root node.
        image_shape (Tuple[Optional[int], Optional[int]]): SET BY ENV. Height and Width of the rendered images.
        debug_rendering (bool): Whether to render more information for debugging purposes.
        positioning_camera (bool): Whether to make the camera controllable with keyboard input for manual positioning.

    Returns:
        scene_creation_result = {
            "mcr_controller_sofa": controller_sofa,
            "mcr_environment": environment,
            "camera": camera,
        }
    """

    # Calibration file for eMNS
    cal_path = str(HERE / "calib/Navion_2_Calibration_24-02-2020.yaml")

    # Parameters instrument
    young_modulus_body = 170e6  # (Pa)
    young_modulus_tip = 21e6  # (Pa)
    length_body = 0.5  # (m)
    length_tip = 0.034  # (m)
    outer_diam = 0.00133  # (m)
    inner_diam = 0.0008  # (m)

    length_init = 0.35

    # Parameters environment
    environment_stl = str(HERE / "meshes/flat_models/flat_model_circles.stl")

    # Parameter magnet
    magnet_length = 4e-3  # (m)
    magnet_id = 0.86e-3  # (m)
    magnet_od = 1.33e-3  # (m)
    magnet_remanence = 1.45  # (T)

    # Parameter for beams
    nume_nodes_viz = 600
    num_elem_body = 30
    num_elem_tip = 3

    # Transforms
    # Sofa sim frame in Navion

    # Model in sofa sim frame
    rot_env_sim = [0, 0, 0]  # rpy angles
    transl_env_sim = [0, 0, 0]

    # transforms (translation , quat)
    T_sim_mns = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1]

    # Define entry pose in model
    # starting pose in environment frame
    T_start_env = [-0.04, 0.01, 0.002, 0.0, 0.0, 0.0, 1.0]

    X = Vec3(T_start_env[0], T_start_env[1], T_start_env[2])
    r = R.from_euler("xyz", rot_env_sim, degrees=True)
    X = r.apply(X)

    q = Quat.createFromEuler([rot_env_sim[0] * math.pi / 180, rot_env_sim[1] * math.pi / 180, rot_env_sim[2] * math.pi / 180])
    qrot = Quat(T_start_env[3], T_start_env[4], T_start_env[5], T_start_env[6])
    q.rotateFromQuat(qrot)

    # starting pose sofa sim frame
    T_start_sim = [X[0] + transl_env_sim[0], X[1] + transl_env_sim[1], X[2] + transl_env_sim[2], q[0], q[1], q[2], q[3]]

    # transform environment
    quat_env_sim = Quat.createFromEuler([rot_env_sim[0] * np.pi / 180, rot_env_sim[1] * np.pi / 180, rot_env_sim[2] * np.pi / 180])
    T_env_sim = [transl_env_sim[0], transl_env_sim[1], transl_env_sim[2], quat_env_sim[0], quat_env_sim[1], quat_env_sim[2], quat_env_sim[3]]

    ###################
    # Camera and lights
    ###################
    root_node.addObject("RequiredPlugin", name="Sofa.GL.Component.Shader")
    root_node.addObject("RequiredPlugin", name="Sofa.Component.Visual")
    root_node.addObject("RequiredPlugin", name="Sofa.Component.ODESolver.Backward")
    root_node.addObject("RequiredPlugin", name="Sofa.Component.LinearSolver.Iterative")
    root_node.addObject("RequiredPlugin", name="Sofa.Component.Constraint.Projective")

    root_node.addObject("LightManager", listening=True, ambient=(0.8, 0.8, 0.8, 0.8))
    root_node.addObject("DirectionalLight", direction=(1, 1, 0), color=(0.8, 0.8, 0.8, 1.0))
    root_node.addObject("DirectionalLight", direction=(-1, 1, 0), color=(0.8, 0.8, 0.8, 1.0))

    placement_kwargs = {
        "position": [-0.0, 0.0, 1.1],
        "lookAt": [0.0, -0.001, 0.0],
        "orientation": [0.0, 0.0, 0.0, 0.0],
    }

    light_source_kwargs = {
        "cutoff": 45.0 / 1.2,
        "color": [0.8] * 4,
        "attenuation": 0.0,
        "exponent": 1.0,
        "shadowsEnabled": False,
    }
    camera = Camera(
        root_node=root_node,
        placement_kwargs=placement_kwargs,
        with_light_source=True,
        show_object=debug_rendering,
        show_object_scale=1.0,
        light_source_kwargs=light_source_kwargs,
        vertical_field_of_view=12,
        width_viewport=image_shape[0],
        height_viewport=image_shape[1],
        z_near=1.0,
        z_far=1000.0,
    )

    if positioning_camera:
        root_node.addObject(camera)

    # simulator
    mcr_simulator.Simulator(root_node=root_node)

    # eMNS
    navion = mcr_emns.EMNS(name="Navion", calibration_path=cal_path)

    # environment
    environment = mcr_environment.Environment(root_node=root_node, environment_stl=environment_stl, T_env_sim=T_env_sim, color=[1.0, 0.0, 0.0, 0.3])

    # magnet
    magnet = mcr_magnet.Magnet(length=magnet_length, outer_diam=magnet_od, inner_diam=magnet_id, remanence=magnet_remanence, color=[0.2, 0.2, 0.2, 1.0])

    # magnets on both ends of flexible segment
    magnets = [0.0 for i in range(num_elem_tip)]
    magnets[0] = magnet
    magnets[1] = magnet

    # instrument
    instrument = mcr_instrument.Instrument(
        name="mag_gw",
        root_node=root_node,
        length_body=length_body,
        length_tip=length_tip,
        outer_diam=outer_diam,
        inner_diam=inner_diam,
        young_modulus_body=young_modulus_body,
        young_modulus_tip=young_modulus_tip,
        magnets=magnets,
        num_elem_body=num_elem_body,
        num_elem_tip=num_elem_tip,
        nume_nodes_viz=nume_nodes_viz,
        T_start_sim=T_start_sim,
        fixed_directions=[0, 0, 1, 0, 0, 0],
        color=[0.2, 0.8, 1.0, 1.0],
    )

    # ros-based controller
    controller_sofa = mcr_controller_sofa.ControllerSofa(
        root_node=root_node,
        e_mns=navion,
        instrument=instrument,
        environment=environment,
        length_init=length_init,
        T_sim_mns=T_sim_mns,
    )
    root_node.addObject(controller_sofa)

    ###############
    # Returned data
    ###############
    scene_creation_result = {
        "mcr_controller_sofa": controller_sofa,
        "mcr_environment": environment,
        "camera": camera,
    }
    return scene_creation_result
