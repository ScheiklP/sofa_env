import Sofa
import numpy as np


class Instrument(Sofa.Core.Controller):
    """
    A class used to define instrument objects and to build the SOFA mechanical,
    collision and visual models of the magnetic instrument.

    :param root_node: The sofa root node
    :param magnets: ...
    :type magnets: list[magnet]
    :param name: The name of the instrument object
    :type name: str
    :param length_body: The length of the proximal segement of the instrument (m)
    :type length_body: float
    :param length_tip: The length of the distal segement of the instrument tip (m)
    :type length_tip: float
    :param outer_diam: The outer diameter of the proximal segement of the instrument (m)
    :type outer_diam: float
    :param inner_diam: The inner diameter of the proximal segement of the instrument (m)
    :type inner_diam: float
    :param young_modulus_body: The Young's modulus of the proximal segement of the instrument (Pa)
    :type young_modulus_body: float
    :param young_modulus_tip: The Young's modulus of the distal segment of the instrument (Pa)
    :type young_modulus_tip: float
    :param num_elem_body: The amount of elements on the proximal segment of mechanical model of the instrument
    :type num_elem_body: int
    :param num_elem_tip: The amount of elements on the distal segment of mechanical model of the instrument
    :type num_elem_tip: int
    :param num_elem_tip: The amount of elements on the visual model of the instrument
    :type num_elem_tip: int
    :param T_start_sim: The transform defining the start pose of the instrument with respect to simulation frame [x, y, z, qx, qy, qz, qw]
    :type T_start_sim: list[float]
    :param fixed_directions: A parameter that fixes the degrees of fredom of the nodes [tx, ty, yz, rx, ry, rz]
    :type fixed_directions: list[int]
    :param color: The color of instrument used for visualization [r, g, b, alpha]
    :type color: list[float]
    :param `*args`: The variable arguments are passed to the SofaCoreController
    :param `**kwargs`: The keyword arguments arguments are passed to the SofaCoreController
    """

    def __init__(
        self,
        root_node,
        magnets,
        name="mag_instrument",
        length_body=0.5,
        length_tip=0.0034,
        outer_diam=0.00133,
        inner_diam=0.0008,
        young_modulus_body=170e6,
        young_modulus_tip=21e6,
        num_elem_body=30,
        num_elem_tip=3,
        nume_nodes_viz=600,
        T_start_sim=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        fixed_directions=[0, 0, 0, 0, 0, 0],
        color=[0.2, 0.8, 1.0, 1.0],
        *args,
        **kwargs
    ):

        # These are needed (and the normal way to override from a python class)
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root_node = root_node

        self.magnets = magnets
        self.index_mag = np.nonzero(self.magnets)[0]
        self.outer_diam = outer_diam
        self.inner_diam = inner_diam
        self.num_elem_body = num_elem_body
        self.num_elem_tip = num_elem_tip

        self.color = color

        # the inner diameter of the beam is not accounted for to
        # compute the stiffness:
        # compute the outer diameter of a plain circular section with
        # equivalent moment of area
        self.outer_diam_qu = ((outer_diam / 2.0) ** 4.0 - (inner_diam / 2.0) ** 4.0) ** (1.0 / 4.0)

        # index of the magnets for visual object
        self.index_mag_visu = [int(nume_nodes_viz * (length_body + length_tip - magnets[0].length)), int(nume_nodes_viz * (length_body + length_tip))]

        self.fixed_directions = fixed_directions

        topoLines_guide = self.root_node.addChild(name + "_topo_lines")
        topoLines_guide.addObject(
            "WireRestShape",
            name="InstrRestShape",
            straightLength=length_body,
            length=length_body + length_tip,
            numEdges=nume_nodes_viz,
            youngModulus=young_modulus_body,
            spireDiameter=250.0,
            numEdgesCollis=[self.num_elem_body, self.num_elem_tip],
            printLog=True,
            template="Rigid3d",
            spireHeight=0.0,
            radius=self.outer_diam_qu / 2.0,
            radiusExtremity=self.outer_diam_qu / 2.0,
            densityOfBeams=[self.num_elem_body, self.num_elem_tip],
            youngModulusExtremity=young_modulus_tip,
        )
        topoLines_guide.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Dynamic")
        topoLines_guide.addObject("EdgeSetTopologyContainer", name="meshLinesGuide")
        topoLines_guide.addObject("EdgeSetTopologyModifier", name="Modifier")
        topoLines_guide.addObject("EdgeSetGeometryAlgorithms", name="GeomAlgo", template="Rigid3d")
        topoLines_guide.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")
        topoLines_guide.addObject("MechanicalObject", name="dofTopo2", template="Rigid3d")

        self.InstrumentCombined = self.root_node.addChild(name)
        self.InstrumentCombined.addObject("RequiredPlugin", name="Sofa.Component.ODESolver.Backward")
        self.InstrumentCombined.addObject("EulerImplicitSolver", rayleighStiffness=0.2, printLog=False, rayleighMass=0.0)
        self.InstrumentCombined.addObject("RequiredPlugin", name="Sofa.Component.LinearSolver.Direct")
        self.InstrumentCombined.addObject("BTDLinearSolver", verification=False, subpartSolve=False, verbose=False)
        self.InstrumentCombined.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Grid")
        self.RG = self.InstrumentCombined.addObject("RegularGrid", name="meshLinesCombined", zmax=1, zmin=1, nx=self.num_elem_body + self.num_elem_tip, ny=1, nz=1, xmax=0.2, xmin=0, ymin=0, ymax=0)
        self.InstrumentCombined.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")
        self.MO = self.InstrumentCombined.addObject("MechanicalObject", showIndices=False, name="DOFs", template="Rigid3d")

        self.MO.init()
        restPos = []
        indicesAll = []
        i = 0
        for pos in self.MO.rest_position.value:
            restPos.append(T_start_sim)
            indicesAll.append(i)
            i = i + 1

        forcesList = ""
        for i in range(0, self.num_elem_body + self.num_elem_tip):
            forcesList += " 0 0 0 0 0 0 "

        indicesList = list(range(0, self.num_elem_body + self.num_elem_tip))

        self.MO.rest_position.value = restPos

        self.IC = self.InstrumentCombined.addObject("WireBeamInterpolation", WireRestShape="@../" + name + "_topo_lines" + "/InstrRestShape", radius=self.outer_diam_qu / 2.0, printLog=True, name="InterpolGuide")
        self.InstrumentCombined.addObject("AdaptiveBeamForceFieldAndMass", massDensity=155.0, name="GuideForceField", interpolation="@InterpolGuide")
        self.InstrumentCombined.addObject("RequiredPlugin", name="Sofa.Component.MechanicalLoad")
        self.CFF = self.InstrumentCombined.addObject("ConstantForceField", name="CFF", indices=indicesList, forces=forcesList, indexFromEnd=True)

        self.CFF_visu = self.InstrumentCombined.addObject("ConstantForceField", name="CFFVisu", indices=0, force="0 0 0 0 0 0", showArrowSize=1.0e2)

        self.IRC = self.InstrumentCombined.addObject(
            "InterventionalRadiologyController",
            xtip=[0.001],
            name="m_ircontroller",
            instruments="InterpolGuide",
            step=0.0007,
            printLog=True,
            listening=True,
            template="Rigid3d",
            startingPos=T_start_sim,
            rotationInstrument=[0.0],
            speed=1e-12,
            mainDirection=[0, 0, 1],
            threshold=5e-9,
            controlledInstrument=0,
        )
        self.InstrumentCombined.addObject("RequiredPlugin", name="Sofa.Component.Constraint.Lagrangian.Correction")
        self.InstrumentCombined.addObject("LinearSolverConstraintCorrection", wire_optimization="true", printLog=False)
        self.InstrumentCombined.addObject("RequiredPlugin", name="Sofa.Component.Constraint.Projective")
        self.InstrumentCombined.addObject("FixedConstraint", indices=0, name="FixedConstraint")
        self.InstrumentCombined.addObject("RequiredPlugin", name="Sofa.Component.SolidMechanics.Spring")
        self.InstrumentCombined.addObject("RestShapeSpringsForceField", points="@m_ircontroller.indexFirstNode", angularStiffness=1e8, stiffness=1e8)

        # restrict DOF of nodes
        self.InstrumentCombined.addObject("PartialFixedConstraint", indices=indicesAll, fixedDirections=self.fixed_directions, fixAll=True)

        # Collision model
        Collis = self.InstrumentCombined.addChild(name + "_collis")
        Collis.activated = True
        Collis.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Dynamic")
        Collis.addObject("EdgeSetTopologyContainer", name="collisEdgeSet")
        Collis.addObject("EdgeSetTopologyModifier", name="colliseEdgeModifier")
        Collis.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")
        Collis.addObject("MechanicalObject", name="CollisionDOFs")
        Collis.addObject("MultiAdaptiveBeamMapping", controller="../m_ircontroller", useCurvAbs=True, printLog=False, name="collisMap")
        Collis.addObject("RequiredPlugin", name="Sofa.Component.Collision.Geometry")
        Collis.addObject("LineCollisionModel", proximity=0.0, group=1)
        Collis.addObject("PointCollisionModel", proximity=0.0, group=1)

        # VISU ROS
        CathVisuROS = self.InstrumentCombined.addChild("CathVisuROS")
        CathVisuROS.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Grid")
        CathVisuROS.addObject("RegularGrid", name="meshLinesCombined", zmax=0.0, zmin=0.0, nx=nume_nodes_viz, ny=1, nz=1, xmax=1.0, xmin=0.0, ymin=0.0, ymax=0.0)
        CathVisuROS.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")
        self.MO_visu = CathVisuROS.addObject("MechanicalObject", name="ROSCatheterVisu", template="Rigid3d")
        CathVisuROS.addObject("AdaptiveBeamMapping", interpolation="@../InterpolGuide", printLog="1", useCurvAbs="1")

        # visualization sofa
        CathVisu = self.InstrumentCombined.addChild(name + "_viz")
        CathVisu.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")
        CathVisu.addObject("MechanicalObject", name="QuadsCatheter")
        CathVisu.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Dynamic")
        CathVisu.addObject("QuadSetTopologyContainer", name="ContainerCath")
        CathVisu.addObject("QuadSetTopologyModifier", name="Modifier")
        CathVisu.addObject("QuadSetGeometryAlgorithms", name="GeomAlgo", template="Vec3d")
        self.root_node.addObject("RequiredPlugin", name="Sofa.Component.Topology.Mapping")
        CathVisu.addObject("Edge2QuadTopologicalMapping", flipNormals="true", input="@../../" + name + "_topo_lines" + "/meshLinesGuide", nbPointsOnEachCircle="10", output="@ContainerCath", radius=self.outer_diam_qu / 2, tags="catheter")
        CathVisu.addObject("AdaptiveBeamMapping", interpolation="@../InterpolTube0", input="@../DOFs", isMechanical="false", name="VisuMapCath", output="@QuadsCatheter", printLog="1", useCurvAbs="1")
        VisuOgl = CathVisu.addChild("VisuOgl")
        VisuOgl.addObject("OglModel", quads="@../ContainerCath.quads", color=self.color, material="texture Ambient 1 0.2 0.2 0.2 0.0 Diffuse 1 1.0 1.0 1.0 1.0 Specular 1 1.0 1.0 1.0 1.0 Emissive 0 0.15 0.05 0.05 0.0 Shininess 1 20", name="VisualCatheter")
        VisuOgl.addObject("RequiredPlugin", name="Sofa.Component.Mapping.Linear")
        VisuOgl.addObject("IdentityMapping", input="@../QuadsCatheter", output="@VisualCatheter", name="VisuCathIM")
