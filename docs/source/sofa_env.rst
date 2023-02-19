.. _base_class:

SofaEnv Base Class
==================

SofaEnv is the abstract class for SOFA simulation environments. When you create a new scene, you should use SofaEnv as the parent class of your task specific environment.
SofaEnv handles calling the ``createScene`` function of your scene description, setting up rendering if needed, acting as the interface to the actual SOFA simulation, and exposing Gym functions.

.. note::
   Reading the RGB buffer from OpenGL through pyglet does not work on WSL -> set the rendermode to ``WSL`` if you want to retrieve image observations.

.. automodule:: sofa_env.base
    :members:
    :undoc-members:
    :show-inheritance:
