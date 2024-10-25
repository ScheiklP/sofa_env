Frequently Asked Questions (FAQ)
================================

* :ref:`Changing color <faq_color>`
* :ref:`FreeMotionAnimationLoop <faq_freemotion>`
* :ref:`Objects not moving <faq_freemotion_2>`
* :ref:`Component does nothing <faq_component_names>`


.. _faq_color:
.. list-table::
   :widths: 200
   :header-rows: 1

   * - How do I change the color of an object during runtime through python?
   * - You can use :py:meth:`sofa_env.sofa_templates.visual.set_color`

.. _faq_freemotion:
.. list-table::
   :widths: 200
   :header-rows: 1

   * - What is up with :class:`~sofa_env.sofa_templates.scene_header.AnimationLoopType`?
   * - | This parameter determines which of SOFA's animation loops should be used.
       | In short, ``DEFAULT`` is a very simple, but physically inaccurate animation loop,
       | because it handles collisions etc. by introducing forces on collision.
       | ``FREEMOTION`` on the other hand is a constraint based animation loop,
       | which is more physically accurate, but also has to solve more complex equations
       | to find solutions for all constraints such as interactions and collisions.
       | For a more detailed explanation, see the official `SOFA documentation <https://www.sofa-framework.org/community/doc/simulation-principles/animation-loop/>`_.
       | Using the ``FREEMOTION`` animation loop requires adding ``ConstraintCorrection`` components to simulation nodes.
       | ``SofaEnv`` handles this mostly automated by passing the ``AnimationLoopType`` to the objects found in :ref:`basic_objects`.

.. _faq_freemotion_2:
.. list-table::
   :widths: 200
   :header-rows: 1

   * - | My simulated objects are not moving, but SOFA shows no errors
       | and the python code seems to be working. Why is this?
   * - | This might be caused by using the ``FREEMOTION`` animation loop,
       | but not adding ``ConstraintCorrection`` components to the simulation nodes.


.. _faq_component_names:
.. list-table::
   :widths: 200
   :header-rows: 1

   * - I added components to the scene, but they seem to do nothing.
   * - | Maybe you added two components of the same type, to the same node.
       | For example two ``FixedProjectiveConstraint``s. If a node has more than one
       | component of the same type, you have to set different names for them.
       | E.g. ``node.addObject("FixedProjectiveConstraint", name="first_constraint")``

.. _faq_opengl:
.. list-table::
   :widths: 200
   :header-rows: 1

   * - Trying to run a scene gives a ``[ERROR]   [SofaRuntime] GLError: GLError``.
   * - | Make sure that you installed the proprietary GPU drivers
       | (e.g. check if `nvidia-smi` is available as a command).
       | You might also have to reinstall the ``pyopengl`` and ``pyopengl-accelerate`` packages.
