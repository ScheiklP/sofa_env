.. _env_functions:

Interacting with the Simulation
===============================

This how-to guide will show you which functions you have to define in order to specify how the step function interacts with the SOFA simulation.

We will build this environment around this very basic scene description that just contains a controllable rigid object as it's agent and an uncontrollable rigid object as the target.

.. code-block:: python

    from sofa_env.sofa_templates.rigid import ControllableRigidObject, RigidObject, RIGID_PLUGIN_LIST
    from sofa_env.sofa_templates.scene_header import add_scene_header, SCENE_HEADER_PLUGIN_LIST

    PLUGIN_LIST = RIGID_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST

    def createScene(root_node: Sofa.Core.Node) -> dict:

        add_scene_header(root_node=root_node, plugin_list=PLUGIN_LIST)

        controllable_object = ControllableRigidObject(
            parent_node=root_node,
            name="object",
            pose=(50, 50, 0, 0, 0, 0, 1),
        )

        target = RigidObject(
            parent_node=root_node,
            name="target",
            pose=(50, 50, 0, 0, 0, 0, 1),
        )

        return {
            "root_node": root_node,
            "controllable_object": controllable_object,
            "target": target,
        }


Step
####

``observation, reward, done, info = env.step(action)`` is the main function to interact with the environment.
It applies the action to the sofa simulation, retrieves the next observation, calculates the reward, determines if the episode is done (environment is in a terminal state), and returns additional information as a dictionary.

.. code-block:: python

    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, dict]:
        rgb_observation = super().step(action)
        observation = self._get_observation(rgb_observation)
        reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()

        return observation, reward, done, info


To specify the environment's behavior in a granular way, we define the following functions.

* ``_do_action``
* ``_get_observation``
* ``_get_reward``
* ``_get_done``
* ``_get_info``

``_do_action``
**************

Applying an action to the SOFA simulation means

1. changing physical values of the simulation, such as specifying an object's new position, changing a springs's stiffness, or attaching objects to each other and then
2. triggering SOFA's animation loop for one or more steps.

Translating an agents action (defined by the environment's action space) to changes in the SOFA simulation is done in ``_do_action``.
This function has to be specified by you, since the environment class makes no assumptions about how you want to interact with the actual SOFA simulation and how the simulation graph is structured.

We will write code that takes the action, represented by a numpy array, and applies a change to the sphere's position in simulation.

.. code-block:: python

    def _do_action(self, action: np.ndarray) -> None:
        scaled_action = action * self.time_step * self.maximum_velocity

        old_pose = self.scene_creation_result["controllable_object"].get_pose()

        # Poses in SOFA are defined as [Cartesian position, quaternion for orientation]
        new_pose = old_pose + np.append(scaled_action, np.array([0, 0, 0, 1]))

        self.scene_creation_result["controllable_object"].set_pose(new_pose)


``_get_observation``
***********************

If you are not using pixel observations, you also have to specify how observations are determined from the current simulation state.
If you want to observe the Cartesian position of an object in the scene, for example, you have to define how these values are retrieved, optionally transformed and passed back from the env because, again, the env does not know how you define your simulation graph, how the elements are named, and what sort of values you want to read.
The observation could be anything from the Cartesian position of a rigid object, to the point wise stress of a deformable fem mesh.

.. code-block:: python

    def _get_observation(self, rgb_observation: Union[np.ndarray, None]) -> np.ndarray:

        if self._observation_type == ObservationType.RGB:
                observation = maybe_rgb_observation
        else:
            observation = self.observation_space.sample()
            observation[:] = self.scene_creation_result["controllable_object"].get_pose()[:3]

        return observation

``_get_reward``
***************

In ``_get_reward``, you specify the reward function of your environment.
You are free to define this function in any way you like.
Meaning you can calculate the reward based on the various conventions of a reward function.
For example by calculating a reward for the current state r(s), the state and the action that lead to the state f(s, a), or the previous state (make sure to save the relevant values), the action and the following state  r(s, a, s').
This function should return a single float.

For our example, we assume that the goal of this environment is the move the sphere to the target position.

The reward function for that goal could look something like this

.. code-block:: python

    def _get_reward(self) -> float:

        current_position = self.scene_creation_result["controllable_object"].get_pose()[:3]
        target_position = self.scene_creation_result["target"].get_pose()[:3]
 
        reward_features["distance_to_target"] = -np.linalg.norm(current_position - target_position)
        reward_features["time_step_cost"] = -1.0
        reward_features["successful_task"] = 10.0 * reward_features["distance_to_target"] <= self._distance_to_target_threshold

        self.reward_features = reward_features.copy()

        reward = 0.0
        for value in reward_features.values():
            reward += value

        return reward


``_get_done``
*************

In ``_get_done`` you check whether your environment is in a terminal state and return a boolean value.

.. code-block:: python

    def _get_done(self) -> bool:

        return reward_features["successful_task"] > 0

``_get_info``
*************

``_get_info`` returns a dictionary with additional information about the environments state, the episode, debugging information or anything else you want to pass to the learning algorithm.

.. code-block:: python

    def _get_info(self) -> dict:
        return self.reward_features


Reset
#####

In ``env.reset()`` you define how you want to reset the SOFA simulation as well as the environment.
SOFA's own reset function resets the state of the simulation components in the simulation graph to the state that was defined on scene creation.
Any additional behavior, like chosing a new position for objects, cleaning up any values of the previous episode, and setting a new goal are defined by you.

The reset function should be the first thing you call after instantiating the environment, since the first call initializes the SOFA simulation.

.. code-block:: python

    def reset(self) -> np.ndarray:
        super().reset()

        # sample new positions for object and target
        object_position = self.rng.uniform([-100.0] * 3, [100.0] * 3)
        target_position = self.rng.uniform([-100.0] * 3, [100.0] * 3)


        # set the new positions
        self.scene_creation_result["controllable_object"].set_pose(np.append(object_position, np.array([0, 0, 0, 1])))
        self.scene_creation_result["target"].set_pose(np.append(target, np.array([0, 0, 0, 1])))

        return self._get_observation(rgb_observation=self._maybe_update_rgb_buffer())

