import numpy as np
from typing import Callable, Optional

import Sofa
import Sofa.Core

from sofa_env.scenes.tissue_dissection.sofa_objects.cauter import PivotizedCauter
from sofa_env.scenes.tissue_dissection.sofa_objects.cauter import CAUTER_PLUGIN_LIST as ORG_CAUTER_PLUGIN_LIST
from sofa_env.sofa_templates.rigid import RIGID_PLUGIN_LIST, PivotizedRigidObject
from sofa_env.scenes.grasp_lift_touch.sofa_objects.point_of_interest import PointOfInterest

CAUTER_PLUGIN_LIST = RIGID_PLUGIN_LIST + ORG_CAUTER_PLUGIN_LIST


def cauter_collision_model_func(attached_to: Sofa.Core.Node, self: PivotizedRigidObject) -> Sofa.Core.Node:
    """Create a collision model for the cauter."""

    collision_model_positions = [
        [0.0, -1.5, 14.4],
        [0.0, -1.2, 14.4],
        [0.0, -0.9, 14.4],
        [0.0, -0.6, 14.4],
        [0.0, -0.3, 14.4],
        [0.0, 0.0, 14.4],
        [0.0, 0.4, 14.4],
        [0.0, 0.7, 14.4],
        [0.0, 1.0, 14.4],
        [0.0, 1.3, 14.4],
        [0.0, 1.6, 14.4],
        [0.0, 1.6, 14.4],
        [0.0, 1.5, 13.4],
        [0.0, 1.4, 12.4],
        [0.0, 1.2, 11.4],
        [0.0, 1.0, 10.4],
        [0.0, 0.6, 9.4],
        [0.0, 0.3, 8.4],
        [0.0, 0.0, 7.4],
    ]
    collision_model_positions.extend([[0.0, 0.0, z] for z in range(-50, 6, 2)])
    collision_model_radii = [0.8 for _ in range(len(collision_model_positions))]

    collision_node = attached_to.addChild("collision")
    self.collision_mechanical_object = collision_node.addObject("MechanicalObject", template="Vec3d", position=collision_model_positions)
    self.collision_center_index = 5
    self.collision_tip_index = 0
    self.sphere_collision_model = collision_node.addObject("SphereCollisionModel", radius=collision_model_radii)
    collision_node.addObject("RigidMapping")

    return collision_node


class Cauter(PivotizedCauter):
    def __init__(
        self,
        poi_to_touch: Optional[PointOfInterest] = None,
        activation_deadzone: float = 0.1,
        add_collision_model_func: Callable = cauter_collision_model_func,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            add_collision_model_func=add_collision_model_func,
            *args,
            **kwargs,
        )
        # When to trigger cauterization / target activation
        self.activation_deadzone = activation_deadzone
        self.active = False
        self.poi = poi_to_touch

    def onAnimateBeginEvent(self, _) -> None:
        if self.poi is not None:
            if self.poi.is_in_point_of_interest(self.get_collision_center_position()) and self.active:
                self.poi.activated = True

    def do_action(self, action: np.ndarray) -> None:
        """Combine state (python) and pose (sofa) update in one function to move the tool.

        Args:
            action: Action in the form of an array with delta ptsd and absolute activation values.
        """

        self.set_state(state=self.ptsd_state + action[:4])
        if abs(action[-1]) > self.activation_deadzone:
            self.active = True if action[-1] > 0 else False

    def set_extended_state(self, state: np.ndarray) -> None:
        """Set the ptsd state and the activation state.

        Args:
            state: State in the form of an array with ptsd and activation values.
        """
        self.set_state(state=state[:4])
        if abs(state[-1]) > self.activation_deadzone:
            self.active = True if state[-1] > 0 else False

    def get_ptsda_state(self):
        return np.append(self.ptsd_state, float(self.active))

    def get_collision_center_position(self) -> np.ndarray:
        """Get the position of the cutting center (the middle of the active tip) in world coordinates."""
        return self.collision_mechanical_object.position.array()[self.collision_center_index]

    def get_collision_tip_position(self) -> np.ndarray:
        """Get the position of the cutting tip (the round end of the active tip) in world coordinates."""
        return self.collision_mechanical_object.position.array()[self.collision_tip_index]
