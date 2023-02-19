from functools import partial
from pathlib import Path
import Sofa
import Sofa.Core

from sofa_env.sofa_templates.camera import Camera, CAMERA_PLUGIN_LIST
from sofa_env.sofa_templates.deformable import DeformableObject, DEFORMABLE_PLUGIN_LIST
from sofa_env.sofa_templates.motion_restriction import add_fixed_constraint_in_bounding_box, MOTION_RESTRICTION_PLUGIN_LIST
from sofa_env.sofa_templates.rigid import ControllableRigidObject, RIGID_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import add_scene_header, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST
from sofa_env.sofa_templates.collision import add_collision_model, COLLISION_PLUGIN_LIST

PLUGIN_LIST = DEFORMABLE_PLUGIN_LIST + RIGID_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST + CAMERA_PLUGIN_LIST + MOTION_RESTRICTION_PLUGIN_LIST + VISUAL_PLUGIN_LIST + COLLISION_PLUGIN_LIST

HERE = Path(__file__).resolve().parent


def createScene(root_node: Sofa.Core.Node):

    add_scene_header(
        root_node=root_node,
        plugin_list=PLUGIN_LIST,
        dt=0.01,
        collision_detection_method_kwargs={
            "alarmDistance": 10.0,
            "contactDistance": 3.0,
        },
    )

    root_node.addObject("LightManager")

    root_node.addObject("DirectionalLight", direction=(1, 1, 0), color=(0.8, 0.8, 0.8, 1.0))
    root_node.addObject("DirectionalLight", direction=(-1, 1, 0), color=(0.8, 0.8, 0.8, 1.0))

    camera = Camera(
        root_node=root_node,
        placement_kwargs={
            "position": [50.0, 0.0, 500],
            "lookAt": [0, 0, 0],
        },
        z_near=0.1,
        z_far=800.0,
        width_viewport=600,
        height_viewport=600,
    )

    add_wall_visual = partial(add_visual_model, color=(0, 0, 1))
    deformable_wall = DeformableObject(
        parent_node=root_node,
        name="wall",
        volume_mesh_path=HERE / Path("meshes/wall.msh"),
        total_mass=10,
        visual_mesh_path=HERE / Path("meshes/wall.stl"),
        collision_mesh_path=HERE / Path("meshes/wall.stl"),
        rotation=(0, 90, 0),
        add_visual_model_func=add_wall_visual,
    )

    add_fixed_constraint_in_bounding_box(
        attached_to=deformable_wall.node,
        min=(-20, -100, -100),
        max=(20, -60, 100),
    )

    add_sphere_visual = partial(add_visual_model, color=(1, 0, 0))
    add_sphere_collision = partial(add_collision_model, contact_stiffness=100)
    controllable_sphere = ControllableRigidObject(
        parent_node=root_node,
        name="sphere",
        pose=(50, 50, 0, 0, 0, 0, 1),
        total_mass=5,
        visual_mesh_path=HERE / Path("meshes/sphere.obj"),
        collision_mesh_path=HERE / Path("meshes/sphere.obj"),
        show_object=True,
        show_object_scale=30,
        scale=20,
        add_visual_model_func=add_sphere_visual,
        add_collision_model_func=add_sphere_collision,
    )

    return {
        "root_node": root_node,
        "camera": camera,
        "controllable_sphere": controllable_sphere,
        "deformable_wall": deformable_wall,
    }
