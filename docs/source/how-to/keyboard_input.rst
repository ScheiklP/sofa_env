Controlling Instruments Through Keyboard Inputs
-----------------------------------------------

During the development process of a scene (particularly while calling it with ``runSofa``), it can be helpful to control the interactive objects of your scene through keyboard inputs.
SOFA already has an event handling, that can be triggered by pressing and holding down ``control``, and then pressing the keys you want to send to your scene.
To process these key events in your scene, you have to add a ``Sofa.Core.Controller`` object to your scene, that implements a ``onKeypressedEvent(self, event)`` or ``onKeyreleasedEvent(self, event)``.
For an example, see :meth:`sofa_env.scenes.grasp_lift_touch.sofa_objects.tool_controller.ToolController.onKeypressedEvent`.

.. code-block:: python

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
            elif ord(key) == 32:  # Space
                self.print_tool_state()
            elif ord(key) == 19:  # up
                self.do_action(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))


Do not forget, that the ``Sofa.Core.Controller`` object has the be added to the simulation graph.

.. code-block:: python

    controller = ToolController(...)
    scene_node.addObject(controller)


.. warning::

   The keyboard events are not registered, when running the scene through the python intrepreter with ``python3 my_env.py``, only through ``runSofa scene_description.py``.
