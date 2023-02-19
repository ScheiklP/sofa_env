Rope Grasping Implementation
############################

Explanation
-----------

A rope in ``sofa_env`` is modeled as a series of ``Rigid3d`` frames that are connected through a ``BeamFEMForceField``.
This means, that every point on the rope has a position, plus an orientation.
Grasping a rope is done by intruducing springs between points on the rope, and points on the gripper.
If these springs just account for the position of the rope, the rope will not follow the gripper realistically.
For example for rotating the gripper:

.. image:: ../images/rope/rope_grasping_vec_mapped.gif
  :width: 600
  :align: center


If the springs account for position and orientation, and we just use the original orientation of the gripper, the orientation will just snap to the one of the gripper.
Grasping a bent rope will lead to very wrong behavior, because the rope will always snap to a 90 degree angle in relation to the gripper.

.. image:: ../images/rope/rope_grasping_rigid_mapped.gif
  :width: 600
  :align: center


Our implementation accounts for that by updating the position and orientation of the reference points for grasping on the gripper separately.
The position is just updated through the absolute position of the gripper.
The orientation, however, is calculated based on the absolute orientation of the gripper, and a rotational offset.
In the beginning, this rotational offset is zero, but on grasping, the offset is set to the currently measured orientational difference between rope and gripper.
This way, if we change the orientation of the gripper in consecutive steps, the added offset will hold the rope in the relative orientation it first had during grasping.

.. image:: ../images/rope/rope_grasping_py_mapped.gif
  :width: 600
  :align: center


Relevant Code
-------------

In the SOFA scene graph, we add a two ``MechanicalObject`` to the gripper. A ``Vec3d`` that is ``RigidMapping`` to the ``motion_target_mechanical_object`` and thus follows the gripper.
The second one is a ``Rigid3d`` and is not mapped to any other node.

.. code-block:: python

    self.grasping_node = self.gripper_node.addChild("grasping")

    # A reference object that has as many points as points as there are collisions models, and is mapped to the motion target
    motion_reference_node = self.grasping_node.addChild("motion_reference")
    self.motion_mapping_mechanical_object = motion_reference_node.addObject(
        "MechanicalObject",
        template="Vec3d",
        position=[position for position in collision_spheres_config["positions"]],
        showObject=show_object,
        showObjectScale=show_object_scale / 2,
    )
    motion_reference_node.addObject("RigidMapping", input=self.physical_shaft_mechanical_object.getLinkPath())

    # The MechanicalObject that is used for grasping by introducing springs. No mapping, because position and orientation are set in onAnimateBeginEvent
    self.grasping_mechanical_object = self.grasping_node.addObject(
        "MechanicalObject",
        template="Rigid3d",
        position=[append_orientation(position) for position in collision_spheres_config["positions"]],
        showObject=show_object,
        showObjectScale=show_object_scale / 2,
    )

At the beginning of each simulation step, the ``Rigid3d`` model is updated with positions from the ``Vec3d`` model.
The orientation is first taken from the ``motion_target_mechanical_object`` and then further rotated with a saved ``orientation_delta``.
This way there is a constant difference in orientation between the ``Rigid3d`` model, and the gripper's ``motion_target_mechanical_object``.


.. code-block:: python

    with self.grasping_mechanical_object.position.writeable() as grasping_frames:
        grasping_frames[:, :3] = self.motion_mapping_mechanical_object.position.array()
        grasping_frames[:, 3:] = multiply_quaternions(
            self.motion_target_mechanical_object.position.array()[0, 3:],
            self.orientation_delta,
        )


The ``orientation_delta`` is calculated when the rope is grasped, based on the difference in orientation between the gripper's ``motion_target_mechanical_object`` and the local orientation of the rope, saved in ``point_orientation``.

.. code-block:: python

    self.orientation_delta[:] = multiply_quaternions(
        conjugate_quaternion(self.motion_target_mechanical_object.position.array()[0, 3:].copy()),
        point_orientation,
    )

We further want that the rope's axis is alinged with the gripper jaws.
We thus calculate the orientation difference that contributes to rotating the rope out of alignment with the jaws (the y axis),
and then factor out this rotation.

.. code-block:: python

    # Find the quaternion, that rotates the transformed (by orientation_delta) y axis into the original y axis.
    transformed_y_axis = rotated_y_axis(self.orientation_delta)
    original_y_axis = np.array([0.0, 1.0, 0.0])
    if not np.allclose(transformed_y_axis, original_y_axis):
        rotation_into_local_y_axis = quaternion_from_vectors(transformed_y_axis, original_y_axis)
        # Add this rotation to the orientation_delta to "undo" the part of the original orientation_delta that rotates
        # the y axis out of alignment -> orientation_delta will now only rotate around y of the gripper jaws
        self.orientation_delta[:] = multiply_quaternions(
            rotation_into_local_y_axis,
            self.orientation_delta,
        )

The attachment between gripper and rope is then implemented as a ``RestShapeSpringsForceField`` between the ``Rigid3d`` model and the rope.
The ``stiffness`` parameter controls the spring stiffness for translational differences, while ``angularStiffness`` controls spring stiffness for orientation differences.

.. code-block:: python

    self.grasping_force_field = self.rope.node.addObject(
        "RestShapeSpringsForceField",
        name=f"grasping_force_field_{name}",
        external_rest_shape=self.grasping_mechanical_object.getLinkPath(),
        stiffness=self.spring_stiffness_grasping,
        angularStiffness=self.angular_spring_stiffness_grasping,
        points=grasp_index_pair[1],
        external_points=grasp_index_pair[0],
    )
