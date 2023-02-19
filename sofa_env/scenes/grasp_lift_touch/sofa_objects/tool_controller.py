import numpy as np

from enum import Enum, unique

import Sofa
import Sofa.Core

from sofa_env.scenes.grasp_lift_touch.sofa_objects.cauter import Cauter
from sofa_env.scenes.grasp_lift_touch.sofa_objects.gripper import Gripper


@unique
class Tool(Enum):
    GRIPPER = 0
    CAUTER = 1


def switchTool(tool: Tool):
    return Tool.CAUTER if tool == Tool.GRIPPER else Tool.GRIPPER


class ToolController(Sofa.Core.Controller):
    def __init__(self, gripper: Gripper, cauter: Cauter) -> None:
        super().__init__()
        self.name = "ToolController"
        self.gripper = gripper
        self.cauter = cauter

        self.active_tool = Tool.GRIPPER

    def onKeypressedEvent(self, event):
        key = event["key"]
        if ord(key) == 1:  # Tab
            self.active_tool = switchTool(tool=self.active_tool)
        if ord(key) == 32:  # Space
            self.print_tool_state()
        if ord(key) == 19:  # up
            self.do_action(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        elif ord(key) == 21:  # down
            self.do_action(np.array([-1.0, 0.0, 0.0, 0.0, 0.0]))
        elif ord(key) == 18:  # left
            self.do_action(np.array([0.0, 1.0, 0.0, 0.0, 0.0]))
        elif ord(key) == 20:  # right
            self.do_action(np.array([0.0, -1.0, 0.0, 0.0, 0.0]))
        elif key == "T":
            self.do_action(np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
        elif key == "G":
            self.do_action(np.array([0.0, 0.0, -1.0, 0.0, 0.0]))
        elif key == "V":
            self.do_action(np.array([0.0, 0.0, 0.0, 1.0, 0.0]))
        elif key == "D":
            self.do_action(np.array([0.0, 0.0, 0.0, -1.0, 0.0]))
        elif key == "K":
            self.do_action(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
        elif key == "L":
            self.do_action(np.array([0.0, 0.0, 0.0, 0.0, -1.0]))
        else:
            pass

    def print_tool_state(self):
        ptsd_state_gripper: np.ndarray = self.gripper.ptsd_state
        ptsd_state_cauter: np.ndarray = self.cauter.ptsd_state

        print("==================================================")
        print("Gripper Tool State")
        print(f"pan: {ptsd_state_gripper[0]}, tilt: {ptsd_state_gripper[1]}, spin: {ptsd_state_gripper[2]}, depth: {ptsd_state_gripper[3]}")
        print("Cauter Tool State")
        print(f"pan: {ptsd_state_cauter[0]}, tilt: {ptsd_state_cauter[1]}, spin: {ptsd_state_cauter[2]}, depth: {ptsd_state_cauter[3]}")
        print("==================================================")

    def do_action(self, action: np.ndarray):
        if self.active_tool == Tool.GRIPPER:
            self.gripper.do_action(action=action)
        if self.active_tool == Tool.CAUTER:
            self.cauter.do_action(action=action)
