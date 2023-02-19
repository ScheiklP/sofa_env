import numpy as np

from typing import Optional, Union, Tuple, List

import Sofa.Core

from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.rope import Rope, RopeCollisionType, ROPE_PLUGIN_LIST

TRANSFER_ROPE_PLUGIN_LIST = (
    [
        "Sofa.Component.Engine.Select",  # <- [SphereROI]
    ]
    + ROPE_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
)


class TransferRope(Rope):
    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        radius: float,
        waypoints: List[np.ndarray],
        poses: List[np.ndarray],
        total_mass: Optional[float] = 1.0,
        young_modulus: Union[float, int] = 5e7,
        poisson_ratio: float = 0.49,
        beam_radius: Optional[float] = None,
        fix_start: bool = True,
        fix_end: bool = False,
        collision_group: Optional[int] = None,
        collision_type: RopeCollisionType = RopeCollisionType.SPHERES,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        rope_color: Tuple[int, int, int] = (255, 0, 0),
        ode_solver_rayleigh_mass: float = 0.1,
        ode_solver_rayleigh_stiffness: float = 0.0,
        visual_resolution: int = 10,
        mechanical_damping: float = 0.2,
        show_object: bool = False,
        show_object_scale: float = 1.0,
    ) -> None:

        super().__init__(
            parent_node=parent_node,
            name=name,
            radius=radius,
            total_mass=total_mass,
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
            beam_radius=beam_radius,
            fix_start=fix_start,
            fix_end=fix_end,
            collision_group=collision_group,
            collision_type=collision_type,
            animation_loop_type=animation_loop_type,
            rope_color=rope_color,
            ode_solver_rayleigh_mass=ode_solver_rayleigh_mass,
            ode_solver_rayleigh_stiffness=ode_solver_rayleigh_stiffness,
            visual_resolution=visual_resolution,
            mechanical_damping=mechanical_damping,
            show_object=show_object,
            show_object_scale=show_object_scale,
            poses=poses,
        )

        self.sphere_roi_center = waypoints
        self.sphere_radius = 4.0

    def get_indices_in_sphere(self, index: int) -> np.ndarray:
        return np.flatnonzero(np.linalg.norm(self.get_positions() - self.sphere_roi_center[index], axis=1) < self.sphere_radius)
