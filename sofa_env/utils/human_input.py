import threading
from typing import List
from inputs import UnpluggedError, devices


def get_gamepad(id: int = 0):
    """Get gamepad device by id."""
    try:
        gamepad = devices.gamepads[id]
    except IndexError:
        raise UnpluggedError(f"No gamepad at id {id} found.")
    return gamepad.read()


class XboxController:
    """Class to interface an Xbox 360 controller.

    Args:
        id (int): The id of the controller. Defaults to 0.
    """
    MAX_TRIGGER_VALUE = 2 ** 10
    MAX_JOYSTICK_VALUE = 2 ** 15
    DEADZONE = 0.09

    def __init__(self, id: int = 0) -> None:

        self.id = id
        self.left_joystick_y = 0.0
        self.left_joystick_x = 0.0
        self.right_joystick_y = 0.0
        self.right_joystick_x = 0.0
        self.left_trigger = 0.0
        self.right_trigger = 0.0
        self.left_bumper = 0.0
        self.right_bumper = 0.0
        self.a = 0.0
        self.x = 0.0
        self.y = 0.0
        self.b = 0.0
        self.left_thumb = 0.0
        self.right_thumb = 0.0
        self.back = 0.0
        self.start = 0.0
        self.left_d_pad = 0.0
        self.right_d_pad = 0.0
        self.up_d_pad = 0.0
        self.down_d_pad = 0.0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, daemon=True)
        self._monitor_thread.start()

    def read(self) -> List[float]:
        """Reads the current state of the controller."""
        lx = self.clip_deadzone(self.left_joystick_x)
        ly = self.clip_deadzone(self.left_joystick_y)
        rx = self.clip_deadzone(self.right_joystick_x)
        ry = self.clip_deadzone(self.right_joystick_y)
        lt = self.clip_deadzone(self.left_trigger)
        rt = self.clip_deadzone(self.right_trigger)
        return [lx, ly, rx, ry, lt, rt]

    def is_alive(self) -> bool:
        return self._monitor_thread.is_alive()

    def _monitor_controller(self):
        """This function is run in a separate thread and constantly monitors the controller."""
        while True:
            events = get_gamepad(self.id)
            for event in events:
                if event.code == "ABS_Y":
                    self.left_joystick_y = event.state / XboxController.MAX_JOYSTICK_VALUE
                elif event.code == "ABS_X":
                    self.left_joystick_x = event.state / XboxController.MAX_JOYSTICK_VALUE
                elif event.code == "ABS_RY":
                    self.right_joystick_y = event.state / XboxController.MAX_JOYSTICK_VALUE
                elif event.code == "ABS_RX":
                    self.right_joystick_x = event.state / XboxController.MAX_JOYSTICK_VALUE
                elif event.code == "ABS_Z":
                    self.left_trigger = event.state / XboxController.MAX_TRIGGER_VALUE
                elif event.code == "ABS_RZ":
                    self.right_trigger = event.state / XboxController.MAX_TRIGGER_VALUE
                elif event.code == "BTN_TL":
                    self.left_bumper = event.state
                elif event.code == "BTN_TR":
                    self.right_bumper = event.state
                elif event.code == "BTN_SOUTH":
                    self.a = event.state
                elif event.code == "BTN_NORTH":
                    self.x = event.state
                elif event.code == "BTN_WEST":
                    self.y = event.state
                elif event.code == "BTN_EAST":
                    self.b = event.state
                elif event.code == "BTN_THUMBL":
                    self.left_thumb = event.state
                elif event.code == "BTN_THUMBR":
                    self.right_thumb = event.state
                elif event.code == "BTN_SELECT":
                    self.back = event.state
                elif event.code == "BTN_START":
                    self.start = event.state
                elif event.code == "BTN_TRIGGER_HAPPY1":
                    self.left_d_pad = event.state
                elif event.code == "BTN_TRIGGER_HAPPY2":
                    self.right_d_pad = event.state
                elif event.code == "BTN_TRIGGER_HAPPY3":
                    self.up_d_pad = event.state
                elif event.code == "BTN_TRIGGER_HAPPY4":
                    self.down_d_pad = event.state

    def clip_deadzone(self, value: float) -> float:
        """Clips the value to the deadzone."""
        if abs(value) < XboxController.DEADZONE:
            return 0.0
        return value
