import Sofa


class Simulator(Sofa.Core.Controller):
    """
    A class used to define the physics and the solver of the SOFA simulation.

    :param root_node: The sofa root node
    :type root_node:
    :param dt: The time step (s)
    :type dt: float
    :param gravity: The gravity verctor (m/s^2)
    :type gravity: float
    :param friction_coef: The coeficient of friction
    :type friction_coef: float
    """

    def __init__(self, root_node, dt=0.01, gravity=[0, 0, 0], friction_coef=0.04, *args, **kwargs):

        # These are needed (and the normal way to override from a python class)
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root_node = root_node
        self.dt = dt
        self.gravity = gravity
        self.friction_coef = friction_coef

        self.root_node.addObject("RequiredPlugin", name="ImportSoftRob", pluginName="SoftRobots")
        self.root_node.addObject("RequiredPlugin", name="ImportBeamAdapt", pluginName="BeamAdapter")
        self.root_node.addObject("RequiredPlugin", name="ImportSofaPython3", pluginName="SofaPython3")

        self.root_node.dt = self.dt
        self.root_node.animate = True
        self.root_node.gravity = self.gravity

        self.root_node.addObject("RequiredPlugin", name="Sofa.Component.Visual")
        self.root_node.addObject(
            "VisualStyle",
            displayFlags="showVisualModels hideBehaviorModels \
                hideCollisionModels hideMappings hideForceFields \
                    hideInteractionForceFields",
        )
        self.root_node.addObject("RequiredPlugin", name="Sofa.Component.AnimationLoop")
        self.root_node.addObject("FreeMotionAnimationLoop")
        self.root_node.addObject("RequiredPlugin", name="Sofa.Component.Constraint.Lagrangian.Solver")
        self.lcp_solver = self.root_node.addObject("LCPConstraintSolver", mu=str(friction_coef), tolerance="1e-6", maxIt="10000", build_lcp="false")
        self.root_node.addObject("RequiredPlugin", name="Sofa.Component.Collision.Response.Contact")
        self.root_node.addObject("RequiredPlugin", name="Sofa.Component.Collision.Detection.Algorithm")
        self.root_node.addObject("RequiredPlugin", name="Sofa.Component.Collision.Detection.Intersection")
        self.root_node.addObject("CollisionPipeline", draw="0", depth="6", verbose="1")
        # NOTE: Commented out as it is deprecated
        # Replaced by following two objects
        # self.root_node.addObject(
        #     'BruteForceDetection',
        #     name='N2')
        self.root_node.addObject("BruteForceBroadPhase", name="N2_1")
        self.root_node.addObject("BVHNarrowPhase", name="N2_2")
        self.root_node.addObject("LocalMinDistance", contactDistance="0.002", alarmDistance="0.003", name="localmindistance", angleCone="0.02")
        self.root_node.addObject("CollisionResponse", name="Response", response="FrictionContactConstraint")
        self.root_node.addObject("RequiredPlugin", name="SofaMiscCollision")
        self.root_node.addObject("DefaultCollisionGroupManager", name="Group")
        self.root_node.addObject("DefaultVisualManagerLoop", name="VisualLoop")

        # set backbround color
        self.root_node.addObject("RequiredPlugin", name="Sofa.Component.Setting")
        self.root_node.addObject("BackgroundSetting", color="1 1 1")
