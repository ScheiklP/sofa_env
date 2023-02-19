.. _getting_started:

Getting Started
===============

To get started, please follow the :ref:`installation instructions <installation>` to set up SOFA and SofaPython3.

After that, either look through the implemented :ref:`scenes`, or have a look at the workflow below to directly get started with implementing your own
reinforcement learning environment.


Environment and Scene Development Workflow
------------------------------------------

1. Create a new directory in ``sofa_env/sofa_env/scenes`` for your scene
2. Add a ``scene_description.py`` file that contains a ``createScene`` function. Look into ``sofa_env/sofa_env/scenes/controllable_object_example/scene_description.py`` for an example.
3. Add your SOFA components to the scene. ``sofa_env.sofa_templates`` contains a few standard components.
4. Iteratively test your ``createScene`` function by passing it to the SOFA binary. ``$SOFA_ROOT/bin/runSofa <path_to_your_scene_description.py>``
5. Let ``createScene`` return the ``root_node``, your camera (if you want to render images), and additional interactive components from your scene as a ``dict``.
6. Create your ``SofaEnv`` class that describes interactions with the scene. Have a look at :ref:`env_functions` and ``sofa_env/sofa_env/scenes/controllable_object_example/controllable_env.py`` for an example.
7. Implement your environment's ``_do_action``, ``step``, and ``reset`` functions.
8. Iteratively test your environment directly through python. ``python3 <path_to_your_env.py>``
