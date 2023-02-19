.. _wrappers:

Environment Wrappers
====================

Recording Trajectories
----------------------
.. automodule:: sofa_env.wrappers.trajectory_recorder
    :members:
    :undoc-members:
    :show-inheritance:

Example Usage
^^^^^^^^^^^^^
.. code-block:: python

    def store_rgb_obs(self: TrajectoryRecorder, shape: Tuple[int, int] = image_shape_to_save):
        observation = self.env.render()
        observation = cv2.resize(
            observation,
            shape,
            interpolation=cv2.INTER_AREA,
        )
        self.trajectory["rgb"].append(observation)

    metadata = {
        "frame_skip": env.frame_skip,
        "time_step": env.time_step,
        "observation_type": env.observation_type.name,
        "reward_amount_dict": env.reward_amount_dict,
        "user_info": args.info,
    }

    env = TrajectoryRecorder(
        env,
        log_dir="trajectories",
        metadata=metadata,
        store_info=True,
        save_compressed_keys=["observation", "terminal_observation", "rgb", "info"],
        after_step_callbacks=[store_rgb_obs],
        after_reset_callbacks=[store_rgb_obs],
    )


Slowing Down Environments to Real Time
--------------------------------------
.. automodule:: sofa_env.wrappers.realtime
    :members:
    :undoc-members:
    :show-inheritance:

Point Cloud Observations
------------------------
.. automodule:: sofa_env.wrappers.point_cloud
    :members:
    :undoc-members:
    :show-inheritance:

Semantically Segmented Image Observations
-----------------------------------------
.. automodule:: sofa_env.wrappers.semantic_segmentation
    :members:
    :undoc-members:
    :show-inheritance:

Making an Environment Episodic
------------------------------
.. automodule:: sofa_env.wrappers.episodic_env
    :members:
    :undoc-members:
    :show-inheritance:

Example Usage
^^^^^^^^^^^^^
.. code-block:: python

    import numpy as np
    from sofa_env.scenes.reach.reach_env import ReachEnv, RenderMode, ObservationType, ActionType

    episode_length = 300
    env = ReachEnv(
        observation_type=ObservationType.STATE,
        render_mode=RenderMode.HUMAN,
        action_type=ActionType.CONTINUOUS,
        image_shape=(1024, 1024),
        frame_skip=1,
        time_step=1 / 30,
        observe_target_position=True,
        maximum_robot_velocity=35.0,
    )

    env = EpisodicEnvWrapper(
        env=env,
        reward_accumulation_func=np.sum,
        episode_length=episode_length,
        stop_on_done=True,
        return_buffers=True,
    )

    reset_obs = env.reset()

    action_plan = np.ones((episode_length, 3)) * (env.target_position - env.end_effector.get_pose()[:3]) / (35.0 * env.dt * 0.001 * episode_length)

    observations, rewards, dones, infos, cumulative_reward = env.step(action_plan)

Rapidly Exploring Random Tree
-----------------------------
.. automodule:: sofa_env.wrappers.rrt
    :members:
    :undoc-members:
    :show-inheritance:
