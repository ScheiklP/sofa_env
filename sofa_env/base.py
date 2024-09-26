import abc
import importlib
import gymnasium as gym
import os

import numpy as np

from enum import Enum, unique
from pathlib import Path
from typing import Dict, Optional, Union, Any, Tuple

from sofa_env.utils.io import SuppressOutput


@unique
class RenderMode(Enum):
    """RenderMode options for SofaEnv.

    This enum specifies if you want to simulate

    - state based without rendering anything (NONE),
    - generate image observations headlessly without creating a window (HEADLESS),
    - generate image observations headlessly, but only when ``env.update_rgb_buffer()`` or ``env.update_rgb_buffer_remote()`` are called manually (MANUAL).
    - create a window to observe the simulation for a remote workstation or WSL (REMOTE).
    - or create a window to observe the simulation (HUMAN).
    """

    HUMAN = 0
    HEADLESS = 1
    REMOTE = 2
    NONE = 3
    MANUAL = 4


@unique
class RenderFramework(Enum):
    """RenderFramework options for SofaEnv.

    This enum specifies which library will be used between pyglet and pygame
    """

    PYGLET = 0
    PYGAME = 1


class SofaEnv(gym.Env, metaclass=abc.ABCMeta):
    """Abstract class for SOFA simulation environments.
    Specific simulations can either subclass this class or use it as a template and implement everything themselves.

    In either case, there are a few requirements on SofaEnv classes:

    - no SOFA code should be called in ``__init__()``. Instead, all initialization should be done on the first call to ``reset()``. This class does this by checking the status of a boolean on each call to ``reset()``.
    - the class must possess members called observation_space and action_space, which are subclasses of ``gymnasium.spaces.Space``.
    - getting rendered images from SOFA assumes, that you created a camera in your scene description. Your createScene function should thus return a dictionary, that has a key value pair of ``'camera': Union[Sofa.Core.Object, sofa_env.sofa_templates.camera.Camera]``.
    - the scene_description script has to contain a ``createScene`` function that takes at least a ``Sofa.Core.Node`` as its root node. It can optionally accept keyword arguments that you pass through ``create_scene_kwargs``.
    - you are responsible for implementing the functions ``step``, ``reset``, and ``_do_action``.

    Notes:
        - if using vectorized environments, MUST use a subprocess wrapper because only one SOFA simulation can exist per process. It will not throw an error, but the simulations will be invalid.
        - for rendering or generating image observations, you have the options to set ``render_mode`` to one of the ``RenderMode`` enum cases. (1) ``HUMAN`` will create a pyglet window and render the images into that window. This classe's step and reset function will also return this image as a numpy array. (2) ``HEADLESS`` will do the same thing, but without a window. Pyglet will use EGL to create a render context, that does not need an actual window. (3) ``REMOTE`` Create and show a pyglet Window (similar to HUMAN render mode) for a remote workstation or when working under WSL. The exported display has to be adjusted and a display server like Xming or Mobaxterm is needed for the visualization. (4) ``NONE`` is the case where you are not interested in visual observations. Use this, if you are only interested in the states of the simulation.

    Args:
        scene_path (Union[Path, str]): absolute path to the scene file (.py) to load.
        time_step (float): size of simulation time step in seconds (default: 0.01).
        frame_skip (int): number of simulation time steps taken (call ``_do_action`` and advance simulation) each time step is called (default: 1).
        render_mode (RenderMode): create a window (``RenderMode.HUMAN``) or run headless (``RenderMode.HEADLESS``).
        create_scene_kwargs (Optional[dict]): a dictionary to pass additional keyword arguments to the ``createScene`` function.
        render_framework (RenderFramework): choose between pyglet and pygame for rendering
        suppress_sofa_init_messages (bool): suppresses SOFA messages in the console. SOFA will print a lot of information to the console by default on initialization of the scene.
    """

    def __init__(
        self,
        scene_path: Union[str, Path],
        time_step: float = 0.01,
        frame_skip: int = 1,
        render_mode: RenderMode = RenderMode.NONE,
        create_scene_kwargs: Optional[dict] = None,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        suppress_sofa_init_messages: bool = False,
    ) -> None:

        # HUMAN -> create and show a pyglet window
        # HEADLESS -> no pyglet window created
        # REMOTE -> create and show a pyglet window for remote workstation or WSL
        # NONE -> no visuals
        # MANUAL -> same as headless, but only render when update_rgb_buffer is called manually
        self.internal_render_mode = render_mode
        self.render_framework = render_framework
        self._initialized = False
        self._modules_imported = False
        self._scene_path = Path(scene_path)
        self._window = None
        self.suppress_sofa_init_messages = suppress_sofa_init_messages

        if not self.internal_render_mode == RenderMode.NONE:
            self.render_mode = "rgb_array"
            self.metadata = {
                "render_fps": 1 / (time_step * frame_skip),
            }

            # Set tha function that returns the rgb buffer
            self._maybe_update_rgb_buffer = self._update_rgb_buffer

            # Figure out what function returns the rgb data
            if self.internal_render_mode == RenderMode.REMOTE:
                self._get_rgb = self.get_rgb_from_open_gl

            elif self.internal_render_mode == RenderMode.MANUAL:
                self._maybe_update_rgb_buffer = lambda *args, **kwargs: None
            else:
                if self.render_framework == RenderFramework.PYGLET:
                    self._get_rgb = self.get_rgb_from_pyglet
                elif self.render_framework == RenderFramework.PYGAME:
                    self._get_rgb = self.get_rgb_from_pygame

            # Figure out which window flip function to call
            # and where to get depth data from
            if self.render_framework == RenderFramework.PYGLET:
                self._flip_window = self._flip_pyglet_window
                self.get_depth = self.get_depth_from_pyglet
            elif self.render_framework == RenderFramework.PYGAME:
                self._flip_window = self._flip_pygame_window
                self.get_depth = self.get_depth_from_pygame

        else:
            self._maybe_update_rgb_buffer = lambda *args, **kwargs: None

        self.create_scene_kwargs = create_scene_kwargs if create_scene_kwargs is not None else {}
        self.time_step = time_step
        self.frame_skip = frame_skip
        self.seed_sequence: Optional[np.random.SeedSequence] = None
        # Flag that indicates whether there is a new seed that has not been consumed yet.
        # This seed is consumed in the reset function.
        self.unconsumed_seed = False

        self.rng: Optional[np.random.Generator] = None

    def step(self, action: Any) -> Union[np.ndarray, None]:
        """Runs ``#frame_skip`` timesteps of the environment's dynamics. The action will be applied each time.
        When the end of an episode is reached, you are responsible for calling ``reset()`` to reset this environment's state.

        Args:
            action (instance of self.action_space): an action that should be applied ``frame_skip`` times as defined in ``_do_action``.

        Returns:
            rgb_buffer (Union[np.ndarray, None]): The current visual observation from the env, if the render mode is not ``RenderMode.NONE``

        Note:
            In order for this function to work properly, please implement ``env._do_action(action)`` to describe how
            an action should be applied to the SOFA simulation. The function will be called ``frame_skip`` times.

            Your implementation of ``env.step(action)`` should

            1. call ``rgb_array = super().step(action)``, (``rgb_array`` will be ``None`` if ``render_mode`` is ``RenderMode.NONE``)
            2. calculate and return next observation, reward, done, and info.

            - observation (instance of ``self.observation_space``): the observation from the env
            - reward (float) : amount of reward for the current state, state action pair, or state transition
            - done (bool): whether the episode has ended
            - info (dict): auxiliary diagnostic information for logging and debugging

            For full control of how steps are applied to the simulation, you can also reimplement the complete step function and do not call the ``super().step(action)``.
            Just make sure you also call ``self._maybe_update_rgb_buffer()`` to get valid images.
        """

        # Progress Simulation n steps
        for _ in range(self.frame_skip):
            self._do_action(action)
            self.sofa_simulation.animate(self._sofa_root_node, self._sofa_root_node.getDt())

        return self._maybe_update_rgb_buffer()

    def reset(self, seed: Union[int, np.random.SeedSequence, None] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Union[np.ndarray, None], Dict]:
        """Resets the SOFA simulation.
        If this is the first ``env.reset()``, the simulation is initialized.

        Args:
            seed (Union[int, np.random.SeedSequence, None]): the seed to use for the environment.
            options (Optional[Dict[str, Any]]): a dictionary of options.

        Returns:
            rgb_buffer (Union[np.ndarray, None]): the current visual observation from the env, if the render mode is not ``RenderMode.NONE``

        Note:
            Your implementation of this function should
            1. call ``super().reset()``,
            2. manually reset any (SOFA) components that need resetting (e.g. by setting a new pose), and
            3. return the initial observation.

            Most environments have SOFA objects that each have their own random number generator.
            To correctly seed the environment, you need to set the seed of each SOFA object.
            However, we can only do that, if the environment is already initialized, so ``reset()`` was called at least once.
            See ``reset()`` in ``DeflectSpheresEnv`` for an example.
        """

        if self.seed_sequence is None or seed is not None:
            if isinstance(seed, np.random.SeedSequence):
                self.seed_sequence = seed
            else:
                self.seed_sequence = np.random.SeedSequence(seed)
            self.rng = np.random.default_rng(self.seed_sequence)
            self.unconsumed_seed = True

        if not self._initialized:
            with SuppressOutput(suppress_stdout=self.suppress_sofa_init_messages, suppress_stderr=self.suppress_sofa_init_messages):
                self._init_sim()

        self.sofa_simulation.reset(self._sofa_root_node)

        return self._maybe_update_rgb_buffer(), {}

    def render(self, mode: Optional[str] = None) -> np.ndarray:
        """Returns the rgb observation from the simulation."""
        if self.internal_render_mode == RenderMode.NONE:
            raise RuntimeError("Calling env.render() is invalid when render_mode was set to RenderMode.NONE.")
        return self._rgb_buffer

    def close(self) -> None:
        """Performs necessary cleanup when environment is no longer needed."""
        if hasattr(self, "sofa_simulation"):
            self.sofa_simulation.unload(self._sofa_root_node)
        if hasattr(self, "_window") and self._window is not None:
            if self.render_framework == RenderFramework.PYGLET:
                self._window.close()
            else:
                self.pygame.quit()

    def _init_sim(self) -> None:
        """Initializes simulation by creating the scene graph."""

        if not self._modules_imported:
            # The SOFA and pyglet modules have a state that depends on when and where they were imported.
            # Since we want that state to be valid in the process, where the SOFA simulation is actually initialized, we import them here.
            self.sofa = importlib.import_module("Sofa")
            self.sofa_core = importlib.import_module("Sofa.Core")
            # self.sofa_runtime = importlib.import_module("SofaRuntime")
            self.sofa_simulation = importlib.import_module("Sofa.Simulation")
            self.camera_templates = importlib.import_module("sofa_env.sofa_templates.camera")

            if not self.internal_render_mode == RenderMode.NONE:
                if self.render_framework == RenderFramework.PYGLET:
                    self.pyglet = importlib.import_module("pyglet")
                    self.pyglet.options["vsync"] = False
                    self.pyglet.options["xsync"] = False

                elif self.render_framework == RenderFramework.PYGAME:
                    self.pygame = importlib.import_module("pygame")

                if self.internal_render_mode == RenderMode.HEADLESS or self.internal_render_mode == RenderMode.MANUAL:
                    if self.render_framework == RenderFramework.PYGLET:
                        # Setting this object will determine which classes are used by pyglet as Display, Screen, and Window.
                        # If headless is True, it will use EGL and HeadlessDisplay, HeadlessScreen, ...
                        # If not, it will determine the correct values based on the OS (Linux, Windows, Mac)
                        self.pyglet.options["headless"] = True
                    elif self.render_framework == RenderFramework.PYGAME:
                        raise NotImplementedError("Headless rendering is not supported for PyGame, as PyGame currently does not support EGL. See https://github.com/ScheiklP/sofa_env/issues/7.")

                self.sofa_gl = importlib.import_module("Sofa.SofaGL")
                self.opengl_gl = importlib.import_module("OpenGL.GL")
                self.opengl_glu = importlib.import_module("OpenGL.GLU")

            # Check if the file with the createScene function exists
            if not self._scene_path.is_absolute():
                self._scene_path = self._scene_path.absolute()
            if not self._scene_path.is_file():
                raise FileNotFoundError(f"Could not find file {self._scene_path}.")

            # Load the file as a module and make sure there is actually a createScene function
            try:
                self._scene_description_module = importlib.machinery.SourceFileLoader("scene_description", str(self._scene_path)).load_module()
            except FileNotFoundError:
                raise FileNotFoundError(f"Failed to load {self._scene_path} during simulation initialization.")

            if not hasattr(self._scene_description_module, "createScene"):
                raise KeyError(f"Module {self._scene_description_module} has no function createScene(root_node: Sofa.Core.Node, **kwargs).")

            self._modules_imported = True

        # Create root node on which the entire scene graph is built, and set the delta T of the simulation
        self._sofa_root_node = self.sofa_core.Node(f"root_{hex(id(self))}")
        self._sofa_root_node.dt.value = self.time_step

        # Generate the scene using createScene function from imported module and pass the create_scene_kwargs to the function
        try:
            self.scene_creation_result = getattr(self._scene_description_module, "createScene")(self._sofa_root_node, **self.create_scene_kwargs)
        except AttributeError as error:
            raise AttributeError(f"Could not create scene using scene file: {self._scene_path}. \n <<{error}>>")

        # Initialize SOFA simulation object
        self.sofa_simulation.init(self._sofa_root_node)

        # If we want to render any images, we have to make sure there is a camera
        if not self.internal_render_mode == RenderMode.NONE:
            if not isinstance(self.scene_creation_result, dict) and "camera" in self.scene_creation_result and isinstance(self.scene_creation_result["camera"], (self.sofa_core.Object, self.camera_templates.Camera)):
                raise KeyError("When creating a scene that should be rendered, please make sure createScene() returns a dictionary with a key value pair of 'camera': Union[Sofa.Core.Object, sofa_env.sofa_templates.camera.Camera].")

            if isinstance(self.scene_creation_result["camera"], self.camera_templates.Camera):
                self._camera_object = self.scene_creation_result["camera"].sofa_object
            else:
                self._camera_object = self.scene_creation_result["camera"]

            # Create a (headless) pyglet or pygame window for rendering based on render_framework
            if self.render_framework == RenderFramework.PYGLET:
                self._init_pyglet_window()
                self._rgb_buffer = np.zeros_like(self.get_rgb_from_pyglet(), dtype=np.uint8)
            elif self.render_framework == RenderFramework.PYGAME:
                self._init_pygame_window()
                self._rgb_buffer = np.zeros_like(self.get_rgb_from_pygame(), dtype=np.uint8)

        # Set flag so that initialization only happens once
        self._initialized = True

    def _flip_pyglet_window(self):
        """Flips the pyglet window."""
        self._window.flip()

    def _flip_pygame_window(self):
        """Flips the pygame window."""
        self.pygame.display.flip()

    def _update_rgb_buffer(self) -> np.ndarray:
        """Updates the visuals in sofa, writes the rgb array to the envs rgb_buffer, flips the window, and returns the rgb array."""
        self._update_sofa_visuals()
        rgb_array = self._get_rgb()
        self._rgb_buffer[:] = rgb_array
        self._flip_window()
        return rgb_array

    def _init_pyglet_window(self):
        """Creates a pyglet window.
        The window is either an actual window on a physical or virtual (VNC or xvfb) display,
        or just a buffer on an OpenGL context created by EGL (no actual display).
        """

        # If headless pyglet will use egl to create a context, display, and screen
        # If not, we can either pass a display name (read from the environment variable DISPLAY) to use, or let pyglet choose one.

        if not self.pyglet.options["headless"]:
            # Read the DISPLAY environment variable (e.g. ":0")
            display_name = os.environ.get("DISPLAY", None)
        else:
            # The headless version of get_display takes no arguments, because it creates its own HeadlessDisplay with EGL
            display_name = None

        if display_name is None:
            # Either headless display or chosen by pyglet
            display = self.pyglet.canvas.get_display()
        else:
            # Actual display by name
            display = self.pyglet.canvas.Display(display_name)

        screen = display.get_screens()  # available screens
        config = screen[0].get_best_config()  # selecting the first screen
        context = config.create_context(None)  # create GL context

        # Create the pyglet window
        self._window = self.pyglet.window.Window(
            height=self._camera_object.heightViewport.value,
            width=self._camera_object.widthViewport.value,
            display=display,
            config=config,
            context=context,
        )

        self._init_opengl()

    def _init_pygame_window(self):
        """Creates a pygame window."""

        self.pygame.init()
        self._window = self.pygame.display.set_mode((self._camera_object.heightViewport.value, self._camera_object.widthViewport.value), self.pygame.DOUBLEBUF | self.pygame.OPENGL)

        self._init_opengl()

    def _init_opengl(self):
        """Initializes the opengl context."""
        self.opengl_gl.glClear(self.opengl_gl.GL_COLOR_BUFFER_BIT | self.opengl_gl.GL_DEPTH_BUFFER_BIT)
        self.opengl_gl.glEnable(self.opengl_gl.GL_LIGHTING)
        self.opengl_gl.glEnable(self.opengl_gl.GL_DEPTH_TEST)
        self.opengl_gl.glDepthFunc(self.opengl_gl.GL_LESS)

        self.sofa_gl.glewInit()
        self.sofa_simulation.initVisual(self._sofa_root_node)
        self.sofa_simulation.initTextures(self._sofa_root_node)

        self.opengl_gl.glMatrixMode(self.opengl_gl.GL_PROJECTION)
        self.opengl_gl.glLoadIdentity()

        # Vertical field of view, aspect ratio, zNear, zFar
        self.opengl_glu.gluPerspective(
            self._camera_object.fieldOfView.value,
            (self._camera_object.widthViewport.value / self._camera_object.heightViewport.value),
            self._camera_object.zNear.value,
            self._camera_object.zFar.value,
        )
        self.opengl_gl.glMatrixMode(self.opengl_gl.GL_MODELVIEW)
        self.opengl_gl.glLoadIdentity()

    def _update_sofa_visuals(self) -> None:
        """Calls sofa and opengl functions to update the rgb and depth information."""

        self.sofa_simulation.updateVisual(self._sofa_root_node)

        self.opengl_gl.glViewport(0, 0, self._camera_object.widthViewport.value, self._camera_object.heightViewport.value)

        self.opengl_gl.glClear(self.opengl_gl.GL_COLOR_BUFFER_BIT | self.opengl_gl.GL_DEPTH_BUFFER_BIT)
        self.opengl_gl.glMatrixMode(self.opengl_gl.GL_PROJECTION)
        self.opengl_gl.glLoadIdentity()

        # Get the camera parameters from the simulation.
        # The camera parameters that you set in the scene description
        # will determine what is rendered to the display.
        # field of view, aspect ratio, zNear, zFar.
        self.opengl_glu.gluPerspective(
            self._camera_object.fieldOfView.value,
            (self._camera_object.widthViewport.value / self._camera_object.heightViewport.value),
            self._camera_object.zNear.value,
            self._camera_object.zFar.value,
        )

        self.opengl_gl.glMatrixMode(self.opengl_gl.GL_MODELVIEW)
        self.opengl_gl.glLoadIdentity()

        self.opengl_gl.glMultMatrixd(self._camera_object.getOpenGLModelViewMatrix())
        self.sofa_gl.draw(self._sofa_root_node)

    def get_rgb_from_pygame(self) -> np.ndarray:
        """Reads the rgb buffer from pygame and returns a copy."""
        rgb_array = self.pygame.surfarray.array3d(self._window)
        return np.copy(np.flipud(rgb_array))

    def get_depth_from_pygame(self) -> np.ndarray:
        """Reads the rgb buffer from openGl for pygame and returns a copy."""

        self.opengl_gl.glViewport(0, 0, self._window.get_width(), self._window.get_height())
        depth_buffer = np.empty((self._window.get_height(), self._window.get_width()), dtype=np.float32)
        self.opengl_gl.glReadPixels(0, 0, self._window.get_width(), self._window.get_height(), self.opengl_gl.GL_DEPTH_COMPONENT, self.opengl_gl.GL_FLOAT, depth_buffer)

        depth_buffer = (depth_buffer * 255).astype(np.uint8)
        depth_array = np.expand_dims(depth_buffer, axis=2)

        return np.copy(np.flipud(depth_array))

    def get_rgb_from_pyglet(self) -> np.ndarray:
        """Reads the rgb buffer from pyglet and returns a copy.

        Note:
            Pyglet widows have a front and back buffer that are exchanged when ``window.flip()`` is called.
            The content of the back buffer is undefined, which is why we return a copy.
            Without the copy, we could get out a bit more performance from the environment.
        """
        rgba_buffer = self.pyglet.image.get_buffer_manager().get_color_buffer()
        rgba_image_data = rgba_buffer.get_image_data()
        rgba_array = np.frombuffer(rgba_image_data.get_data(), dtype=np.uint8)
        rgba_array = rgba_array.reshape(rgba_buffer.height, rgba_buffer.width, 4)

        return np.copy(np.flipud(rgba_array)[:, :, :3])

    def get_depth_from_pyglet(self) -> np.ndarray:
        """Reads the depth buffer from pyglet and returns a copy.

        Note:
            This function differs from ``get_depth_from_open_gl`` in that this function returns the depth buffer as a uint8 value in [0, 255].
            ``get_depth_from_open_gl`` returns the depth information as float values in absolute distances to the camera.
        """
        depth_buffer = self.pyglet.image.get_buffer_manager().get_depth_buffer()
        depth_image_data = depth_buffer.get_image_data()
        depth_array = np.frombuffer(depth_image_data.get_data(), dtype=np.uint8)
        depth_array = depth_array.reshape((depth_buffer.height, depth_buffer.width, 1))

        return np.copy(np.flipud(depth_array))

    def get_rgb_from_open_gl(self) -> np.ndarray:
        """Reads the rgb buffer from OpenGL and returns a copy."""
        gl = self.opengl_gl
        height = self._camera_object.heightViewport.value
        width = self._camera_object.widthViewport.value

        buffer = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image_array = np.fromstring(buffer, np.uint8)

        if image_array != []:
            image = image_array.reshape(height, width, 3)
            image = np.flipud(image)[:, :, :3]
        else:
            image = np.zeros((height, width, 3))

        return np.copy(image)

    def get_depth_from_open_gl(self) -> np.ndarray:
        """Reads the depth buffer from OpenGL and returns a depth array with absolute distance to the camera values."""
        z_near = self._camera_object.zNear.value
        z_far = self._camera_object.zFar.value
        height = self._camera_object.heightViewport.value
        width = self._camera_object.widthViewport.value
        # Pyglet's get_image_data() always reads the buffer as unsigned byte.
        # We want to read the depth buffer as float, so we call glReadPixels directly
        depth_buffer = self.opengl_gl.glReadPixels(0, 0, width, height, self.opengl_gl.GL_DEPTH_COMPONENT, self.opengl_gl.GL_FLOAT)
        depth_array = np.frombuffer(depth_buffer, dtype=np.float32)
        depth_array = depth_array.reshape(height, width, 1)
        depth_array = np.flipud(depth_array)
        # Linearize the depth array values [0, 1] to their actual physical values -> how far are the objects away from the camera
        # multiplied "regular" equation with -1 to get larger values for objects that are farther away from the camera
        linearized_depth_array = z_far * z_near / (z_far + depth_array * (z_near - z_far))

        return linearized_depth_array

    @abc.abstractmethod
    def _do_action(self, action) -> None:
        return
