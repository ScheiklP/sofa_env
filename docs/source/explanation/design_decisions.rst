Reset
#####

We seperated the instantiation of the python object that represents the environment from the actual initialization of the SOFA simulation.
This means that after you instatiate the environment with ``env = SofaEnv()``, the simulation is not yet ready.
The simulation is initialized when you first call ``env.reset()``.


We made this destiction between instantiating the python object and initializing the SOFA simulation to be able to use parallel environments.
Only one SOFA simulation can exist in a single process (due to SOFA’s implementation).
By separating python object instantiation from SOFA simulation initialization we can create multiple environments in the parent process, move each environment to a separate process and then initialize the SOFA simulations in their own isolated processes.


``frame_skip`` and ``time_step``
################################

The stability of a simulation is closely related to the amount of time you want to simulate in a single simulation step, specified by ``time_step``.
The smaller this time step, the more stable and accurate the simulation will be, but it also means simulating a fixed amount of time comes at a cost of more computation time.
If we want to simulate the behavior of an object along one second and our time step is 0.1 seconds, we need 10 simulation steps and with 0.001 we need 1000 steps.
The first will be faster and the second more accurate and stable.
The optimal value is highly dependent on your specific simulation.

The frequency at which observations are returned from the environment (and thus the frequency at which the agent gets observations and has to act) can be decoupled from that with the ``frame_skip`` parameter.

The action, passed to the environment in the step function is applied ``frame_skip`` times, before updating the observation.
The time distance between observations is ``frame_skip`` * ``time_step``.
This way we can decouple observing new states from the simulation ``time_step``, to find an optimal trade off between simulation stability and speed without affecting the agent's observation.
Keep in mind, however, that the actions should represent physically meaningful things like a velocity in meters per second.
You will have to account for the simulations ``time_step``.

Additionally, we often try to make a task less ambiguous, by passing more than one observation to the agent at a time.
If the time step is very small, however, the difference between the observations is minimal and spacing them out more may be more beneficial to properly learning the task.
This can change the learning task from trying to choose route directions every millimeter to chosing them every meter.
