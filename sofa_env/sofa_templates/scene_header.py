from typing import List, Optional, Tuple, Union
import Sofa.Core
from enum import Enum
import numpy as np

SCENE_HEADER_PLUGIN_LIST = [
    "Sofa.Component.SceneUtility",  # <- [RequiredPlugin]
    "Sofa.Component.AnimationLoop",  # <- [FreeMotionAnimationLoop]
    "Sofa.Component.Collision.Detection.Algorithm",  # <- [BVHNarrowPhase, BruteForceBroadPhase, DefaultPipeline]
    "Sofa.Component.Collision.Response.Contact",  # <- Needed to use components [DefaultContactManager]
    "Sofa.Component.Collision.Detection.Intersection",  # <- [NewProximityIntersection]
    "Sofa.Component.Constraint.Lagrangian.Solver",  # <- [LCPConstraintSolver]
    "Sofa.Component.Visual",  # <- [VisualStyle]
    "Sofa.Component.Setting",  # <- [BackgroundSetting]
]


VISUAL_STYLES = {
    "full_debug": ["showAll"],
    "debug": ["showVisual", "showForceFields", "showCollisionModels", "showBehaviorModels"],
    "normal": ["showVisual"],
}


class AnimationLoopType(Enum):
    """Describes the animation loop of the simulation. For documentation see
    - [DefaultAnimationLoop](https://www.sofa-framework.org/community/doc/components/animationloops/defaultanimationloop/)
    - [FreeMotionAnimationLoop](https://www.sofa-framework.org/community/doc/components/animationloops/freemotionanimationloop/) and [more](https://www.sofa-framework.org/community/doc/simulation-principles/constraint/lagrange-constraint/#freemotionanimationloop)

    TLDR:
        FreeMotionAnimationLoop includes more steps and components and is results in more realistic simulations. Required for complex constraint-based interactions.
        DefaultAnimationLoop is easier to set up and runs more stable.
    """

    DEFAULT = "DefaultAnimationLoop"
    FREEMOTION = "FreeMotionAnimationLoop"


class IntersectionMethod(Enum):
    """Describes how collisions are detected. For documentation see
    - [NewProximityIntersection](https://sofacomponents.readthedocs.io/en/latest/_modules/sofacomponents/CollisionAlgorithm/NewProximityIntersection.html)
    - [LocalMinDistance](https://www.sofa-framework.org/community/doc/components/collisions/intersectiondetections/localmindistance/)
    - [MinProximityIntersection](https://www.sofa-framework.org/community/doc/components/collisions/intersectiondetections/minproximityintersection/)
    - DiscreteIntersection

    TLDR:
        MinProximityIntersection is optimized for meshes.
        LocalMinDistance method is similar to MinProximityIntersection but in addition filters the list of DetectionOutput to keep only the contacts with the local minimal distance.
        NewProximityIntersection seems to be an improved method, but there is no documentation.
        DiscreteIntersection does not take arguments for alarmDistance and contactDistance.
    """

    MINPROXIMITY = "MinProximityIntersection"
    LOCALMIN = "LocalMinDistance"
    NEWPROXIMITY = "NewProximityIntersection"
    DISCRETE = "DiscreteIntersection"


INTERSECTION_METHOD_DEFAULT_KWARGS = {
    IntersectionMethod.MINPROXIMITY: {
        "alarmDistance": 1.0,
        "contactDistance": 0.5,
    },
    IntersectionMethod.LOCALMIN: {
        "alarmDistance": 1.0,
        "contactDistance": 0.5,
        "angleCone": 0.0,
    },
    IntersectionMethod.NEWPROXIMITY: {
        "alarmDistance": 1.0,
        "contactDistance": 0.5,
    },
    IntersectionMethod.DISCRETE: {},
}


class ContactManagerResponse(Enum):
    DEFAULT = "PenalityContactForceField"
    FRICTION = "FrictionContactConstraint"


class ConstraintSolverType(Enum):
    """Describes the solver used to solve for constraints in the FreeAnimationLoop.

    From the [cocumentation](https://www.sofa-framework.org/community/doc/simulation-principles/constraint/lagrange-constraint/#constraintsolver-in-sofa):

    Two different ConstraintSolver implementations exist in SOFA:
        - LCPConstraintSolver: this solvers targets on collision constraints, contacts with frictions which corresponds to unilateral constraints
        - GenericConstraintSolver: this solver handles all kind of constraints, i.e. works with any constraint resolution algorithm

    Moreover, you may find the class ConstraintSolver. This class does not implement a real solver but actually just browses the graph in order to find and use one of the two implementations mentioned above.
    """

    AUTOMATIC = "ConstraintSolver"
    GENERIC = "GenericConstraintSolver"
    LCP = "LCPConstraintSolver"


CONSTRAINT_SOLVER_DEFAULT_KWARGS = {
    ConstraintSolverType.AUTOMATIC: {},
    ConstraintSolverType.GENERIC: {
        "maxIterations": 1000,
        "tolerance": 0.001,
        "computeConstraintForces": False,
        "scaleTolerance": False,
        "multithreading": False,
    },
    ConstraintSolverType.LCP: {
        "tolerance": 0.001,
        "maxIt": 1000,
        "initial_guess": False,
        "build_lcp": False,
        "mu": 0.2,
    },
}


def add_scene_header(
    root_node: Sofa.Core.Node,
    plugin_list: List[str],
    gravity: Union[Tuple[float, float, float], np.ndarray] = (0.0, 0.0, 0.0),
    dt: Optional[float] = None,
    animation_loop: AnimationLoopType = AnimationLoopType.DEFAULT,
    constraint_solver: ConstraintSolverType = ConstraintSolverType.GENERIC,
    constraint_solver_kwargs: Optional[dict] = None,
    visual_style_flags: List[str] = VISUAL_STYLES["normal"],
    background_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    scene_has_collisions: bool = True,
    collision_pipeline_depth: int = 15,
    contact_friction: float = 0.0,
    collision_detection_method: IntersectionMethod = IntersectionMethod.NEWPROXIMITY,
    collision_detection_method_kwargs: Optional[dict] = None,
    scene_has_cutting: bool = False,
    cutting_distance: float = 1.2,
) -> None:
    """Add various standard components to the scene graph.

    Args:
        root_node (Sofa.Core.Node): The scene's root node.
        plugin_list (List[str]): Plugin names to load. This list should containt the contents of all the module level plugin lists that you use in your scene, such as SCENE_HEADER_PLUGIN_LIST.
        gravity (Union[Tuple[float, float, float], np.ndarray]): A vector to specify the scenes gravity.
        dt (Optional[float]): Time step of the simulation.
        animation_loop (AnimationLoopType): Which animation loop to use to simulate the scene.
        constraint_solver (ConstraintSolverType): Which solver to use if animation_loop is ``AnimationLoopType.FREEMOTION``.
        constraint_solver_kwargs (dict): Which solver to use to solve constraints if ``animation_loop`` is ``AnimationLoopType.FREEMOTION``.
        visual_style_flags (List[str]): List of flags that specify what is rendered.
        background_color (Tuple[float, float, float, float]): RGBA value of the scene's background.
        scene_has_collisions (bool): Whether components to detect and react to collisions should be added.
        collision_pipeline_depth (int): ?
        contact_friction (float): Friction coefficient, if ``animation_loop`` is ``AnimationLoopType.FREEMOTION``.
        collision_detection_method (IntersectionMethod): The algorithm used to detect collisions if ``scene_has_collisions`` is ``True``.
        collision_detection_method_kwargs (dict): Additional kwargs for the collision detection algorithm if ``scene_has_collisions`` is ``True``. If ``None``, read from ``INTERSECTION_METHOD_DEFAULT_KWARGS``.
        scene_has_cutting (bool): Whether the scene should simulate cutting.
        cutting_distance (float): Distance of objects at which cutting (removing volume mesh elements) should be triggered. This should probably be smaller then the ``collision_detection_method_kwargs["alarmDistance"]`` or ``collision_detection_method_kwargs["contactDistance"]`` to function properly.
    """

    root_node.gravity = gravity

    if dt is not None:
        root_node.dt = dt

    # Extend plugin list to support cutting.
    if scene_has_cutting:
        SCENE_HEADER_PLUGIN_LIST.append("SofaCarving")
        add_plugins(root_node=root_node, plugin_list=plugin_list + ["SofaCarving"])
    else:
        add_plugins(root_node=root_node, plugin_list=plugin_list)

    root_node.addObject(animation_loop.value)
    root_node.addObject("DefaultVisualManagerLoop")
    root_node.addObject("VisualStyle", displayFlags=visual_style_flags)
    root_node.addObject("BackgroundSetting", color=background_color)

    ############
    # Collisions
    ############
    if scene_has_collisions:
        root_node.addObject("CollisionPipeline", depth=collision_pipeline_depth, draw=False, verbose=False)
        root_node.addObject("BruteForceBroadPhase", name="BroadPhase")
        root_node.addObject("BVHNarrowPhase", name="NarrowPhase")
        if animation_loop == AnimationLoopType.FREEMOTION:
            collision_response = ContactManagerResponse.FRICTION
            collision_response_kwargs = {"responseParams": contact_friction}
        else:
            collision_response = ContactManagerResponse.DEFAULT
            collision_response_kwargs = {}

        root_node.addObject("CollisionResponse", response=collision_response.value, **collision_response_kwargs)

        if collision_detection_method_kwargs is None:
            collision_detection_method_kwargs = INTERSECTION_METHOD_DEFAULT_KWARGS[collision_detection_method]

        root_node.addObject(collision_detection_method.value, **collision_detection_method_kwargs)

    ###################
    # Constraint Solver
    ###################
    if animation_loop == AnimationLoopType.FREEMOTION:
        if constraint_solver_kwargs is None:
            constraint_solver_kwargs = CONSTRAINT_SOLVER_DEFAULT_KWARGS[constraint_solver]
        root_node.addObject(constraint_solver.value, **constraint_solver_kwargs)

    #########
    # Cutting
    #########
    if scene_has_cutting:
        if not scene_has_collisions:
            raise ValueError("Tried to add a CarvingManager without collisions enabled. Please set scene_has_collisions to True, or scene_has_cutting to False.")
        root_node.addObject("CarvingManager", active=True, carvingDistance=cutting_distance)


def add_plugins(root_node: Sofa.Core.Node, plugin_list: List[str]) -> None:
    """Adds a set of plugins to the scene graph.

    The list of plugins is filtered to remove duplicates.
    A Plugins node is added to the ``root_node``.

    Args:
        root_node (Sofa.Core.Node): The scene's root node.
        plugin_list (List[str]): Plugin names to load.

    Examples:
        >>> from sofa_env.sofa_templates.rigid import ControllableRigidObject, RIGID_PLUGIN_LIST
        >>> from sofa_env.sofa_templates.deformable import DeformableObject, DEFORMABLE_PLUGIN_LIST
        >>> scene_plugin_list = RIGID_PLUGIN_LIST + DEFORMABLE_PLUGIN_LIST
        >>> add_plugins(root_node, scene_plugin_list)
    """

    unique_plugins = set(plugin_list)
    plugin_node = root_node.addChild("Plugins")

    for plugin in unique_plugins:
        plugin_node.addObject("RequiredPlugin", pluginName=plugin, name=plugin)
