import Sofa
import Sofa.Core


class Listener(Sofa.Core.Controller):
    def __init__(self, contact_listener) -> None:
        super().__init__()
        self.listener = contact_listener

    def onAnimateBeginEvent(self, _) -> None:

        print(f"Found {self.listener.getNumberOfContacts()} contacts")

        contacts = self.listener.getContactElements()
        for contact in contacts:
            # (object, index, object, index)
            index_on_sphere = contact[1] if contact[0] == 0 else contact[3]
            index_on_edge = contact[3] if contact[0] == 0 else contact[1]

            print(f"Contact between sphere {index_on_sphere} and edge {index_on_edge}")

        print(f"Full contact information: {self.listener.getContactData()}")


def createScene(
    root_node: Sofa.Core.Node,
) -> Sofa.Core.Node:

    plugin_list = [
        "Sofa.Component.Collision.Detection.Algorithm",  # [BVHNarrowPhase, BruteForceBroadPhase, DefaultPipeline]
        "Sofa.Component.Collision.Detection.Intersection",  # [NewProximityIntersection]
        "Sofa.Component.Collision.Geometry",  # [SphereCollisionModel]
        "Sofa.Component.Collision.Response.Contact",  # [DefaultContactManager]
        "Sofa.Component.Constraint.Projective",  # [FixedProjectiveConstraint]
        "Sofa.Component.LinearSolver.Iterative",  # [CGLinearSolver]
        "Sofa.Component.Mass",  # [UniformMass]
        "Sofa.Component.ODESolver.Backward",  # [EulerImplicitSolver]
        "Sofa.Component.SolidMechanics.Spring",  # [StiffSpringForceField]
        "Sofa.Component.Topology.Container.Dynamic",  # [EdgeSetTopologyContainer, EdgeSetTopologyModifier]
        "Sofa.Component.Visual",  # [VisualStyle]
        "Sofa.Component.StateContainer",  # [MechanicalObject
    ]

    plugin_node = root_node.addChild("Plugins")

    for plugin in plugin_list:
        plugin_node.addObject("RequiredPlugin", pluginName=plugin, name=plugin)

    root_node.addObject("DefaultAnimationLoop")
    root_node.addObject("DefaultVisualManagerLoop")
    root_node.addObject(
        "VisualStyle",
        displayFlags=["showVisual", "showForceFields", "showCollisionModels", "showBehaviorModels", "showInteractionForceFields"],
    )

    root_node.addObject("DefaultPipeline")
    root_node.addObject("BruteForceBroadPhase")
    root_node.addObject("BVHNarrowPhase")
    root_node.addObject("DefaultContactManager", response="PenalityContactForceField")

    root_node.addObject(
        "NewProximityIntersection",
        alarmDistance=3.0,
        contactDistance=0.5,
    )

    root_node.gravity = [0.0, -918, 0.0]

    scene_node = root_node.addChild("scene")

    ##################
    # Collision Sphere
    ##################
    sphere_node = scene_node.addChild("cutting_sphere")
    sphere_node.addObject("CGLinearSolver")
    sphere_node.addObject("EulerImplicitSolver")
    sphere_node.addObject("PointSetTopologyContainer")
    sphere_node.addObject("MechanicalObject", template="Rigid3d", position=[5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0], showObject=False, showObjectScale=1.0)
    sphere_node.addObject("UniformMass", totalMass=0.001)
    sphere_node.addObject("SphereCollisionModel", radius=0.5, group=1)

    ################
    # EdgeSet Object
    ################
    edge_node = scene_node.addChild("edges")
    edge_node.addObject("CGLinearSolver")
    edge_node.addObject("EulerImplicitSolver")
    positions = [[x, 0.0, 0.0] for x in range(0, 20)]
    edges = [[x, x + 1] for x in range(len(positions) - 1)]
    edge_node.addObject("EdgeSetTopologyContainer", position=positions, edges=edges)
    edge_node.addObject("EdgeSetTopologyModifier")
    edge_node.addObject("MechanicalObject", showObject=True, showObjectScale=2.0)
    springs = [[first, second, 100, 0.5, 0] for first, second in edges]
    edge_node.addObject("StiffSpringForceField", spring=springs)
    edge_node.addObject("FixedProjectiveConstraint", indices=[0, len(positions) - 1])
    edge_node.addObject("LineCollisionModel")

    ##################
    # Contact Listener
    ##################
    contact_listener = scene_node.addObject(
        "ContactListener",
        collisionModel1=sphere_node.SphereCollisionModel.getLinkPath(),
        collisionModel2=edge_node.LineCollisionModel.getLinkPath(),
    )
    scene_node.addObject(Listener(contact_listener))
