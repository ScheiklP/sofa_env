Introduction
============

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   :hidden:

   getting_started.rst
   sofa_env.rst
   sofa_templates.rst
   scenes.rst
   utils.rst
   wrappers.rst
   how-to.rst
   setting_up_sofa.rst
   explanations.rst

SofaEnv
#######

The goal of this project is providing a `Gym`_ interface to `SOFA`_ in order to use SOFA as a physics-based environment for reinforcement learning.

The Simulation Open Framework Architecture (SOFA) is "an efficient framework dedicated to research, prototyping and development of physics-based simulations".
In particular its ability to perform fast and realistic soft body simulations make this framework appealing for reinforcement learning.

`Gym`_ environments provide a simple interface between reinforcement learning algorithms and learning environments.
The basic components of a gym environment are their state and action spaces as well as a set of interface functions.

SofaEnv provides modular and simple to use functions to

- define how actions, passed to the Gym environment affect what happens in the SOFA simulation, and
- observe the state of the simulation.

Contents
########

If you are new to the project, please have a look at the :ref:`getting_started` section.

The project consists of:

- The :ref:`base_class` that implements the Gym interface for SOFA simulations
- :ref:`sofa_templates` that combine SOFA components to model high-level objects such as deformable objects and pivotized instruments.
- A set of predefined :ref:`scenes` and their learning environments.
- A set of :ref:`utils` to generate point cloud observations, perform motion planning, and more.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Gym: https://gym.openai.com/
.. _SOFA: http://sofa-framework.org
