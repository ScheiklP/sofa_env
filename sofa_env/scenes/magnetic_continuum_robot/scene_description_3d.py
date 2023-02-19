from splib3.numerics import Quat, Vec3
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple
from pathlib import Path

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
    Creates the aortic arch 3d scene of the MCREnv.

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
    environment_stl = str(HERE / "meshes/anatomies/J2-Naviworks.stl")

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

    # Transforms (translation , quat)
    T_sim_mns = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1]

    # Environment in sofa sim frame
    rot_env_sim = [-0.7071068, 0, 0, 0.7071068]
    transl_env_sim = [0.0, -0.45, 0.0]

    T_env_sim = [transl_env_sim[0], transl_env_sim[1], transl_env_sim[2], -0.7071068, 0, 0, 0.7071068]

    # Starting pose in environment frame
    T_start_env = [-0.075, -0.001, -0.020, 0.0, -0.3826834, 0.0, 0.9238795]

    trans_start_env = Vec3(T_start_env[0], T_start_env[1], T_start_env[2])
    r = R.from_quat(rot_env_sim)
    trans_start_env = r.apply(trans_start_env)

    quat_start = Quat(rot_env_sim)
    qrot = Quat(T_start_env[3], T_start_env[4], T_start_env[5], T_start_env[6])
    quat_start.rotateFromQuat(qrot)

    # Starting pose sofa_sim frame
    T_start_sim = [trans_start_env[0] + transl_env_sim[0], trans_start_env[1] + transl_env_sim[1], trans_start_env[2] + transl_env_sim[2], quat_start[0], quat_start[1], quat_start[2], quat_start[3]]

    ###################
    # Camera and lights
    ###################
    root_node.addObject("RequiredPlugin", name="Sofa.GL.Component.Shader")
    root_node.addObject("RequiredPlugin", name="Sofa.Component.Visual")

    root_node.addObject(
        "LightManager",
        listening=True,
        ambient=(
            0.8,
            0.8,
            0.8,
            0.8,
        ),
    )

    root_node.addObject("DirectionalLight", direction=(1, 1, 0), color=(0.8, 0.8, 0.8, 1.0))
    root_node.addObject("DirectionalLight", direction=(-1, 1, 0), color=(0.8, 0.8, 0.8, 1.0))

    placement_kwargs = {"position": [-0.35, -1.0, -1.0], "lookAt": [0.0, -0.3, 0.0], "orientation": [0.0, 0.0, 0.0, 0.0]}

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
        vertical_field_of_view=17,
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
    environment = mcr_environment.Environment(root_node=root_node, environment_stl=environment_stl, name="aortic_arch", T_env_sim=T_env_sim, flip_normals=True, color=[1.0, 0.0, 0.0, 0.3])

    # magnet
    magnet = mcr_magnet.Magnet(length=magnet_length, outer_diam=magnet_od, inner_diam=magnet_id, remanence=magnet_remanence)

    # magnets on both ends of flexible segment
    magnets = [0.0 for i in range(num_elem_tip)]
    magnets[0] = magnet
    magnets[1] = magnet

    # instrument
    instrument = mcr_instrument.Instrument(
        name="mcr",
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
        color=[0.2, 0.8, 1.0, 1.0],
    )

    # sofa-based controller
    controller_sofa = mcr_controller_sofa.ControllerSofa(
        root_node=root_node,
        e_mns=navion,
        instrument=instrument,
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
