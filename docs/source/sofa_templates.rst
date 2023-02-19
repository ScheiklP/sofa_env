.. _sofa_templates:

Sofa Templates
==============

These templates bundle common SOFA components to achieve specific behaviors that are commonly used in scene descriptions.
SOFA requires the user to know a lot about which components work.
SofaEnv hides a lot of the complexity and uses easy to use interfaces to figure out:

- which specific combination of components are required to achieve a certain behavior,
- in which order to add the components to the scene graph,
- how to structure the graph.

E.g. a deformable object, a cuttable object, and partially rigidified object have almost the same components, but they are arranged completely different in the graph.

Scene Header
------------

The function ``sofa_env.sofa_templates.scene_header.add_scene_header`` takes care of adding basic components that basically every scene requires. It adds basic components such as the ``AnimationLoopType`` that determines the event loop of the scene and whether to add components for collision detection. The utility is bundled in this function because the required components are highly interdependent. 

The function ``sofa_env.sofa_templates.scene_header.add_plugins`` can be used to easily aggregate the plugins defined in each submodule and add a unique set to the scene graph.

.. automodule:: sofa_env.sofa_templates.scene_header
    :members:
    :undoc-members:
    :show-inheritance:

.. _basic_objects:

Basic Objects
-------------

Rigid Objects
^^^^^^^^^^^^^

Rigid objects are physical bodies whose position or state can be described by a single pose (position and orientation) in space.

.. automodule:: sofa_env.sofa_templates.rigid
    :members:
    :undoc-members:
    :show-inheritance:

Deformable Objects
^^^^^^^^^^^^^^^^^^

In contrast to rigid objects, deformable objects consist of many different points and relative positions between the points change.

.. automodule:: sofa_env.sofa_templates.deformable
    :members:
    :undoc-members:
    :show-inheritance:

Materials
"""""""""

.. automodule:: sofa_env.sofa_templates.materials
    :members:
    :undoc-members:
    :show-inheritance:

Camera
^^^^^^

Cameras are used to render image observations of the scene.

.. automodule:: sofa_env.sofa_templates.camera
    :members:
    :undoc-members:
    :show-inheritance:


Scene Utilities
-----------------

Collision Models
^^^^^^^^^^^^^^^^

.. automodule:: sofa_env.sofa_templates.collision
    :members:
    :undoc-members:
    :show-inheritance:

Visual Models
^^^^^^^^^^^^^

.. automodule:: sofa_env.sofa_templates.visual
    :members:
    :undoc-members:
    :show-inheritance:

Mesh Loader
^^^^^^^^^^^

.. automodule:: sofa_env.sofa_templates.loader
    :members:
    :undoc-members:
    :show-inheritance:

Mapping Types
^^^^^^^^^^^^^

.. automodule:: sofa_env.sofa_templates.mappings
    :members:
    :undoc-members:
    :show-inheritance:

Motion Restriction
^^^^^^^^^^^^^^^^^^

.. automodule:: sofa_env.sofa_templates.motion_restriction
    :members:
    :undoc-members:
    :show-inheritance:

Numerical Solvers
^^^^^^^^^^^^^^^^^

.. automodule:: sofa_env.sofa_templates.solver
    :members:
    :undoc-members:
    :show-inheritance:

Topologies
^^^^^^^^^^

.. automodule:: sofa_env.sofa_templates.topology
    :members:
    :undoc-members:
    :show-inheritance:
