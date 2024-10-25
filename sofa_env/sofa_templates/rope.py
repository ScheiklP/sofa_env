import numpy as np
from sofa_env.utils.math_helper import euler_to_rotation_matrix, multiply_quaternions, point_rotation_by_quaternion, quaternion_from_vectors, rotation_matrix_to_quaternion

from enum import Enum, unique
from typing import Optional, Union, Tuple, List

import Sofa.Core

from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.solver import ConstraintCorrectionType, SOLVER_PLUGIN_LIST, LinearSolverType, OdeSolverType, add_solver
from sofa_env.sofa_templates.visual import set_color


ROPE_PLUGIN_LIST = (
    [
        "Sofa.Component.Collision.Geometry",  # <- [LineCollisionModel, PointCollisionModel, TriangleCollisionModel, SphereCollisionModel]
        "Sofa.Component.Constraint.Lagrangian.Correction",  # <- [LinearSolverConstraintCorrection]
        "Sofa.Component.Constraint.Projective",  # <- [FixedProjectiveConstraint]
        "Sofa.Component.Mass",  # <- [UniformMass]
        "Sofa.Component.Mapping.Linear",  # <- [IdentityMapping, TubularMapping]
        "Sofa.Component.SolidMechanics.FEM.Elastic",  # <- [BeamFEMForceField]
        "Sofa.Component.Topology.Container.Constant",  # <- Needed to use components [CubeTopology]
        "Sofa.Component.Topology.Container.Dynamic",  # <- [EdgeSetGeometryAlgorithms, EdgeSetTopologyModifier]
        "Sofa.Component.Topology.Mapping",  # <- [Edge2QuadTopologicalMapping]
        "Sofa.GL.Component.Rendering3D",  # <- [OglModel]
        "Sofa.Component.StateContainer",  # <- [MechanicalObject]
        "Sofa.Component.MechanicalLoad",  # <- [UniformVelocityDampingForceField]
    ]
    + SCENE_HEADER_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
)


@unique
class RopeCollisionType(Enum):
    LINES_AND_POINTS = 0
    TRIANGULATED_CUBES = 1  # <- this is currently buggy due to https://github.com/sofa-framework/sofa/issues/3047
    SPHERES = 2
    NONE = 3


class Rope:
    """Creates a rope with a given number of segments and a given length.

    Args:
        parent_node (Sofa.Core.Node): The parent node of the rope.
        name (str): The name of the rope.
        radius (float): The radius of the rope.
        poses (Union[np.ndarray, List]): The poses of the rope's segments.
        total_mass (Optional[float]): The total mass of the rope.
        young_modulus (Union[float, int]): The Young's modulus of the rope.
        poisson_ratio (float): The Poisson's ratio of the rope.
        beam_radius (Optional[float]): The mechanical beam radius of the rope.
        fix_start (bool): Whether to add a fixed constraint to the start of the rope.
        fix_end (bool): Whether to add a fixed constraint to the end of the rope.
        collision_type (RopeCollisionType): The type of collision model to use for the rope.
        animation_loop_type (AnimationLoopType): The type of animation loop to use for the rope.
        rope_color (Tuple[int, int, int]): The color of the rope's visual model with RGB values in [0, 255].
        ode_solver_rayleigh_mass (float): The Rayleigh mass coefficient of the rope's OdeSolver.
        ode_solver_rayleigh_stiffness (float): The Rayleigh stiffness coefficient of the rope's OdeSolver.
        visual_resolution (int): The resolution of the rope's visual model.
        mechanical_damping (float): The mechanical damping of the rope.
        show_object (bool): Whether to render the rope's poses.
        show_object_scale (float): The rendered scale of the rope's poses.
        collision_group (Optional[int]): The collision group of the rope.
        collision_model_indices (Optional[List[int]]): The indices in the list of poses to use for the rope's collision model.
        collision_contact_stiffness (Optional[Union[float, int]]): The contact stiffness of the rope's collision model.
        check_self_collision (bool): Whether to check for self-collisions.
        use_beam_adapter_plugin (bool): Whether to use the BeamAdapter plugin.
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        radius: float,
        poses: Union[np.ndarray, List],
        total_mass: Optional[float] = 1.0,
        young_modulus: Union[float, int] = 5e7,
        poisson_ratio: float = 0.49,
        beam_radius: Optional[float] = None,
        fix_start: bool = False,
        fix_end: bool = False,
        collision_type: RopeCollisionType = RopeCollisionType.SPHERES,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        rope_color: Tuple[int, int, int] = (255, 0, 0),
        ode_solver_rayleigh_mass: float = 0.1,
        ode_solver_rayleigh_stiffness: float = 0.0,
        visual_resolution: int = 10,
        mechanical_damping: float = 0.2,
        show_object: bool = False,
        show_object_scale: float = 1.0,
        collision_group: Optional[int] = None,
        collision_model_indices: Optional[List[int]] = None,
        collision_contact_stiffness: Optional[Union[float, int]] = None,
        check_self_collision: bool = False,
        use_beam_adapter_plugin: bool = False,
    ) -> None:
        if use_beam_adapter_plugin:
            ROPE_PLUGIN_LIST.append("BeamAdapter")

        # Node information
        self.parent_node = parent_node
        self.name = name
        self.node = self.parent_node.addChild(name + "_node")
        self.animation_loop_type = animation_loop_type
        self.constraint_correction_type = ConstraintCorrectionType.LINEAR
        self.collision_group = collision_group

        # Rope parameters
        self.radius = radius
        self.start_position = poses[0][:3]
        self.start_poses = poses.copy()
        self.num_points = len(poses)
        self.length = np.linalg.norm(poses[-1][:3] - poses[0][:3])

        # Material properties
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        if beam_radius is None:
            self.beam_radius = radius
        else:
            self.beam_radius = beam_radius

        # Solver
        self.time_integration = self.node.addObject("EulerImplicitSolver", rayleighStiffness=ode_solver_rayleigh_stiffness, rayleighMass=ode_solver_rayleigh_mass)
        self.linear_solver = self.node.addObject("BTDLinearSolver", template="BTDMatrix6d")

        # MechanicalObject with one Rigid3d per segment
        self.mechanical_object = self.node.addObject(
            "MechanicalObject",
            template="Rigid3d",
            position=poses,
            showObject=show_object,
            showObjectScale=show_object_scale,
        )

        # Topology to connects the segments
        self.topology_container = self.node.addObject(
            "EdgeSetTopologyContainer",
            edges=[[x, x + 1] for x in range(len(poses) - 1)],
        )
        self.node.addObject("EdgeSetTopologyModifier")
        self.node.addObject("EdgeSetGeometryAlgorithms")

        # Optionally fixing the first/last index
        if fix_start or fix_end:
            indices = []
            if fix_start:
                indices.append(0)
            if fix_end:
                indices.append(1)
            self.node.addObject("FixedProjectiveConstraint", indices=indices)

        if use_beam_adapter_plugin:
            if total_mass is None:
                total_mass = 1.0

            BEAM_INTERPOLATION_KWARGS = {
                "radius": self.beam_radius,
                "defaultYoungModulus": [young_modulus] * (len(poses) - 1),
                "defaultPoissonRatio": [poisson_ratio] * (len(poses) - 1),
            }
            self.node.addObject("BeamInterpolation", **BEAM_INTERPOLATION_KWARGS)

            rope_length = 0.0
            for i in range(len(poses) - 1):
                rope_length += np.linalg.norm(poses[i][:3] - poses[i + 1][:3])
            rope_volume = rope_length * np.pi * self.beam_radius**2
            mass_density = total_mass / rope_volume
            ADAPTIVE_BEAM_FORCE_FIELD_KWARGS = {
                "massDensity": mass_density,
                "reinforceLength": True,
                "shearStressComputation": True,
                "rayleighMass": 0.1,
                "rayleighStiffness": 0.1,
                "computeMass": True,
            }
            self.node.addObject("AdaptiveBeamForceFieldAndMass", **ADAPTIVE_BEAM_FORCE_FIELD_KWARGS)

            if mechanical_damping > 0.0:
                self.node.addObject("UniformVelocityDampingForceField", dampingCoefficient=mechanical_damping)

            # Constraint correction for the FreeMotionAnimationLoop
            if animation_loop_type == AnimationLoopType.FREEMOTION:
                self.constraint_correction = self.node.addObject("LinearSolverConstraintCorrection", wire_optimization=True)

        else:
            # Optional mass
            if total_mass is not None:
                self.node.addObject("UniformMass", totalMass=total_mass)

            # FEM forcefield to describe the deformation
            self.node.addObject("BeamFEMForceField", radius=self.beam_radius, youngModulus=young_modulus, poissonRatio=poisson_ratio)
            if mechanical_damping > 0.0:
                self.node.addObject("UniformVelocityDampingForceField", dampingCoefficient=mechanical_damping)

            # Constraint correction for the FreeMotionAnimationLoop
            if animation_loop_type == AnimationLoopType.FREEMOTION:
                self.constraint_correction = self.node.addObject(self.constraint_correction_type.value)

        # Collision models
        self.collision_node = self.node.addChild("collision")

        collision_model_kwargs = {}
        if collision_group is not None:
            collision_model_kwargs["group"] = collision_group
        if check_self_collision:
            collision_model_kwargs["selfCollision"] = True
        if collision_contact_stiffness is not None:
            if animation_loop_type == AnimationLoopType.DEFAULT:
                collision_model_kwargs["contactStiffness"] = collision_contact_stiffness

        if collision_type == RopeCollisionType.LINES_AND_POINTS:
            self.collision_node.addObject("MechanicalObject", template="Vec3d", size=len(poses))
            self.collision_node.addObject("PointCollisionModel", **collision_model_kwargs)
            self.collision_node.addObject("LineCollisionModel")
            self.collision_node.addObject("IdentityMapping")

        elif collision_type == RopeCollisionType.TRIANGULATED_CUBES:
            raise NotImplementedError("RopeCollisionType.TRIANGULATED_CUBES currently only implemented for straight ropes and thus commented out.")
            # cube_corner_distance = radius / np.sqrt(2)
            # # TODO: Is there a "CubeSetTopologyContainer" such that we can construct the topology like the EdgeSetTopologyContainer?
            # self.collision_node.addObject(
            #     "CubeTopology",
            #     nx=num_points,
            #     ny=2,
            #     nz=2,
            #     min=[start_position[0], -cube_corner_distance, -cube_corner_distance],
            #     max=[start_position[0] + length, cube_corner_distance, cube_corner_distance],
            # )
            # self.collision_node.addObject("MechanicalObject")
            # self.collision_node.addObject("PointCollisionModel")
            # self.collision_node.addObject("LineCollisionModel")
            # self.collision_node.addObject("TriangleCollisionModel")
            # self.collision_node.addObject("BeamLinearMapping", isMechanical=True)

        elif collision_type == RopeCollisionType.SPHERES:
            if collision_model_indices is None:
                collision_model_indices = list(range(len(poses)))

            self.collision_node.addObject("MechanicalObject", template="Rigid3d")
            self.collision_node.addObject("SubsetMapping", indices=collision_model_indices)

            collision_model_node = self.collision_node.addChild("models")

            collision_model_node.addObject("MechanicalObject", template="Vec3d")
            self.sphere_collision_models = collision_model_node.addObject("SphereCollisionModel", radius=radius, **collision_model_kwargs)
            collision_model_node.addObject("IdentityMapping")

        else:
            pass

        # Tube for visual model
        tube_node = self.node.addChild("tube")
        tube_mechanical_object = tube_node.addObject("MechanicalObject")
        tube_topology = tube_node.addObject("QuadSetTopologyContainer")
        tube_node.addObject("QuadSetTopologyModifier")
        tube_node.addObject(
            "Edge2QuadTopologicalMapping",
            nbPointsOnEachCircle=visual_resolution,
            radius=radius,
            input=self.topology_container.getLinkPath(),
            output=tube_topology.getLinkPath(),
        )
        tube_node.addObject(
            "TubularMapping",
            nbPointsOnEachCircle=visual_resolution,
            radius=radius,
            input=self.mechanical_object.getLinkPath(),
            output=tube_mechanical_object.getLinkPath(),
        )

        triangle_tube_node = tube_node.addChild("triangle_tube")
        triangle_topology = triangle_tube_node.addObject("TriangleSetTopologyContainer")
        triangle_tube_node.addObject("TriangleSetTopologyModifier")
        triangle_tube_node.addObject(
            "Quad2TriangleTopologicalMapping",
            input=tube_topology.getLinkPath(),
            output=triangle_topology.getLinkPath(),
        )

        # OGL visual model
        visual_node = tube_node.addChild("visual")
        self.ogl_model = visual_node.addObject("OglModel", color=[color / 255 for color in rope_color], triangles=triangle_topology.triangles.getLinkPath())
        visual_node.addObject("IdentityMapping", input=tube_mechanical_object.getLinkPath(), output=self.ogl_model.getLinkPath())

    def get_state(self) -> np.ndarray:
        """Returns the current state of the Rope's MechanicalObject."""
        return self.mechanical_object.position.array()

    def get_reset_state(self) -> np.ndarray:
        """Returns the reset state of the Rope's MechanicalObject."""
        return self.mechanical_object.reset_position.array()

    def get_positions(self) -> np.ndarray:
        """Returns the Cartesian positions of the current state of the Rope's MechanicalObject."""
        return self.mechanical_object.position.array()[:, :3]

    def get_velocities(self) -> np.ndarray:
        """Returns the Cartesian positions of the current state of the Rope's MechanicalObject."""
        return self.mechanical_object.velocity.array()[:, :3]

    def get_reset_positions(self) -> np.ndarray:
        """Returns the Cartesian positions of the reset state of the Rope's MechanicalObject."""
        return self.mechanical_object.reset_position.array()[:, :3]

    def get_center_of_mass(self) -> np.ndarray:
        """Calculates the center of mass' Cartesian position by returning the mean position on the rope (uniform mass)."""
        return np.mean(self.get_positions(), axis=0)

    def get_reset_center_of_mass(self) -> np.ndarray:
        """Calculates the center of mass' Cartesian position by returning the mean reset position on the rope (uniform mass)."""
        return np.mean(self.get_reset_positions(), axis=0)

    def set_state(self, state) -> None:
        """Sets the state of the Rope's MechanicalObject.

        TODO:
            This does nothing...
        """
        with self.mechanical_object.position.writeable() as sofa_state:
            sofa_state[:] = state

    def set_reset_state(self, state) -> None:
        """Sets the reset state of the Rope's MechanicalObject."""
        with self.mechanical_object.reset_position.writeable() as sofa_state:
            sofa_state[:] = state

    def set_color(self, new_color: Tuple[int, int, int]):
        """Sets the color of the rope's visual model with RGB values in [0, 255]."""
        set_color(self.ogl_model, color=tuple(color / 255 for color in new_color))


def poses_for_circular_rope(radius: float, num_points: int, start_position: np.ndarray, min_angle: float = 0, max_angle: float = 360.0, euler_angle_rotation: Optional[np.ndarray] = None) -> np.ndarray:
    """Create poses along a circular shape that can be passed to ``Rope``.

    Args:
        radius (float): Radius of the circle.
        num_points (num_points): Number of points to generate along the circle.
        start_position (np.ndarray): Cartesian position of the first pose.
        max_angle (float): Angle in (0, 360] of the circle to create.
        euler_angle_rotation (Optional[np.ndarray]): Optional XYZ euler angles to rotate the rope.

    Returns:
        poses List(np.ndarray): A list of poses along the circle.
    """

    if 0.0 >= max_angle > 360:
        raise ValueError(f"Cannot create poses outside of (0, 360]. Received {max_angle=}")

    if 0.0 >= min_angle > max_angle:
        raise ValueError(f"Cannot create poses outside of (0, max_angle]. Received {max_angle=} and {min_angle=}")

    phis = np.linspace(min_angle, max_angle, num_points)
    poses = []
    for phi in phis:
        pose = np.zeros(7)
        pose[0] = radius * np.cos(np.deg2rad(phi))
        pose[1] = radius * np.sin(np.deg2rad(phi))
        pose[2] = 0.0
        pose[3:] = rotation_matrix_to_quaternion(euler_to_rotation_matrix(np.array([0.0, 0.0, 90.0 + phi])))

        poses.append(pose)

    if euler_angle_rotation is not None:
        if not len(euler_angle_rotation) == 3:
            raise ValueError(f"Expected 3 euler angles for XYZ euler rotation. Received {len(euler_angle_rotation)} as {euler_angle_rotation=}")
        transformation_quaternion = rotation_matrix_to_quaternion(euler_to_rotation_matrix(euler_angle_rotation))
        poses = [np.append(point_rotation_by_quaternion(pose[:3], transformation_quaternion), multiply_quaternions(transformation_quaternion, pose[3:])) for pose in poses]

    poses = np.asarray(poses)

    poses[:, :3] += start_position

    return poses


def poses_for_linear_rope(length: float, num_points: int, start_position: np.ndarray, vector: np.ndarray = np.array([1.0, 0.0, 0.0])) -> List[np.ndarray]:
    """Create poses along a vector with a given length that can be passed to ``Rope``.

    Args:
        num_points (num_points): Number of points to generate along the line.
        start_position (np.ndarray): Cartesian position of the first pose.
        vector (np.ndarray): Cartesian vector pointing along the line.

    Returns:
        poses List(np.ndarray): A list of poses along the line.
    """

    # Find the quaternion that rotates the X-Axis into the vector
    x_axis = np.array([1.0, 0.0, 0.0])
    if not np.allclose(x_axis, vector):
        orientation = quaternion_from_vectors(x_axis, vector)
    else:
        orientation = np.array([0.0, 0.0, 0.0, 1.0])

    # Find where to end the poses
    unit_vector = vector / np.linalg.norm(vector)
    end_position = start_position + unit_vector * length

    # Find the vector between each of the points
    delta_position = end_position - start_position
    position_increment = delta_position / num_points

    # Construct the poses by incrementing along the line, starting from start_position
    poses = [np.concatenate([start_position + position_increment * i, orientation]) for i in range(num_points)]

    return poses


def poses_for_rope_between_points(start_position: np.ndarray, end_position: np.ndarray, num_points: int) -> List[np.ndarray]:
    """Create poses between two Cartesian positions that can be passed to ``Rope``.

    Args:
        num_points (num_points): Number of points to generate along the line.
        start_position (np.ndarray): Cartesian position of the first pose.
        end_position (np.ndarray): Cartesian position of the last pose.

    Returns:
        poses List(np.ndarray): A list of poses between start and end position.
    """

    # Find the vector between each of the points
    delta_position = end_position - start_position
    position_increment = delta_position / (num_points - 1)

    # Find the quaternion that rotates the X-Axis into the vector that connects start and end position
    x_axis = np.array([1.0, 0.0, 0.0])
    if not np.allclose(x_axis, delta_position):
        orientation = quaternion_from_vectors(x_axis, delta_position)
    else:
        orientation = np.array([0.0, 0.0, 0.0, 1.0])

    # Construct the poses by incrementing along the line, starting from start_position
    poses = [np.concatenate([start_position + position_increment * i, orientation]) for i in range(num_points)]

    return poses


@unique
class CosseratBending(Enum):
    LINE = 0
    CIRCLE = 1
    FLIPPEDCIRCLE = 2


class CosseratRope:
    """A variant of the rope class, using the Cosserat Plugin."""

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        radius: float = 2.0,
        total_mass: Optional[float] = 0.22,
        young_modulus: Union[float, int] = 1e6,
        poisson_ratio: float = 0.4,
        beam_radius: Optional[float] = 0.15,
        fix_start: bool = True,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        rope_color: Tuple[int, int, int] = (255, 0, 0),
        visual_resolution: int = 10,
        mechanical_damping: float = 0.2,
        show_object: bool = False,
        show_object_scale: float = 1.0,
        collision_group: Optional[int] = None,
        collision_model_indices: Optional[List[int]] = None,
        collision_contact_stiffness: Optional[Union[float, int]] = None,
        check_self_collision: bool = False,
        length: float = 100.0,
        number_of_segments: int = 10,
        number_of_frames: int = 20,
        start_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
        rest_bend_state: CosseratBending = CosseratBending.LINE,
    ) -> None:
        # Node information
        self.parent_node = parent_node
        self.name = name
        self.node = parent_node.addChild("cosserat_rope")

        self.radius = radius
        self.collision_group = collision_group
        self.length = length
        self.number_of_segments = number_of_segments + 1
        self.number_of_frames = number_of_frames + 1
        self.start_position = start_position

        # If there is no extra argument passed for the mechanical radius of the cosserat beam,
        # use the rope's radius
        if beam_radius is None:
            beam_radius = radius

        # Add solvers to the node
        self.time_integration, self.linear_solver = add_solver(
            self.node,
            ode_solver_type=OdeSolverType.IMPLICITEULER,
            ode_solver_rayleigh_stiffness=0.2,
            ode_solver_rayleigh_mass=0.1,
            linear_solver_type=LinearSolverType.SPARSELDL,
            linear_solver_kwargs={"template": "CompressedRowSparseMatrixd"},
        )

        if animation_loop_type == AnimationLoopType.FREEMOTION:
            self.node.addObject("GenericConstraintCorrection")

        # Add a rigid mechanical object as the base of the rope / beam
        self.rigid_base_node = self.node.addChild("rigid_base")
        self.rigid_base_node.addObject("MechanicalObject", template="Rigid3d", position=np.append(start_position, [0.0, 0.0, 0.0, 1.0]))

        if fix_start:
            self.rigid_base_node.addObject("FixedProjectiveConstraint")

        ################
        # Cosserat state
        ################
        self.cosserat_description_node = self.node.addChild("cosserat_description")
        segment_length = length / number_of_segments
        absolute_segment_x_positions = [start_position[0] + segment_length * i for i in range(number_of_segments + 1)]
        # Used to describe the state of the cosserat beam
        # Not actual cartesian positions
        segment_lengths = [segment_length for _ in range(number_of_segments + 1)]
        if rest_bend_state == CosseratBending.LINE:
            # straight line, no bending on each segment
            segment_bending_state = [[0.0, 0.0, 0.0] for _ in range(number_of_segments + 1)]
        elif rest_bend_state == CosseratBending.CIRCLE:
            # over the whole length, turn 360°
            segment_bending_state = [[0.0, 0.0, -2 * np.pi / length] for _ in range(number_of_segments + 1)]
        elif rest_bend_state == CosseratBending.FLIPPEDCIRCLE:
            # over the whole length, turn 360° in the other direction
            segment_bending_state = [[0.0, 0.0, 2 * np.pi / length] for _ in range(number_of_segments + 1)]
        else:
            raise ValueError(f"Unsupported option for rest_bend_state: {rest_bend_state}")

        # MechanicalObject that holds the state of the cosserat beam (in cosserat coordinates, not Cartesian coordinates)
        self.cosserat_description_node.addObject("MechanicalObject", template="Vec3d", position=segment_bending_state)

        self.cosserat_description_node.addObject(
            "BeamHookeLawForceField",
            crossSectionShape="circular",
            length=segment_lengths,
            radius=beam_radius,
            youngModulus=young_modulus,
            poissonRatio=poisson_ratio,
            rayleighStiffness=0.0,
            lengthY=beam_radius,
            lengthZ=beam_radius,
        )

        self.cosserat_description_node.addObject("UniformVelocityDampingForceField", dampingCoefficient=mechanical_damping)

        #####################
        # Rope frames / poses
        #####################
        frame_length = length / number_of_frames
        self.frame_length = frame_length
        frame_poses = [[start_position[0] + frame_length * i, start_position[1], start_position[2], 0.0, 0.0, 0.0, 1.0] for i in range(number_of_frames + 1)]
        absolute_frame_x_positions = [start_position[0] + frame_length * i for i in range(number_of_frames + 1)]

        # The node that holds the poses of the rope is a child both of the rigid base, as well as the cosserat state
        self.cosserat_frame_node = self.rigid_base_node.addChild("cosserat_frame")
        self.cosserat_description_node.addChild(self.cosserat_frame_node)

        self.topology_container = self.cosserat_frame_node.addObject(
            "EdgeSetTopologyContainer",
            edges=[[x, x + 1] for x in range(number_of_frames)],
        )
        self.cosserat_frame_node.addObject("EdgeSetTopologyModifier")
        self.cosserat_frame_node.addObject("EdgeSetGeometryAlgorithms")
        self.mechanical_object = self.cosserat_frame_node.addObject("MechanicalObject", template="Rigid3d", position=frame_poses, showObject=show_object, showObjectScale=show_object_scale)

        if total_mass is not None:
            self.cosserat_frame_node.addObject("UniformMass", totalMass=total_mass)

        # TODO look at min and max values
        self.cosserat_frame_node.addObject(
            "DiscreteCosseratMapping",
            curv_abs_input=absolute_segment_x_positions,
            curv_abs_output=absolute_frame_x_positions,
            input1=self.cosserat_description_node.MechanicalObject.getLinkPath(),
            input2=self.rigid_base_node.MechanicalObject.getLinkPath(),
            output=self.cosserat_frame_node.MechanicalObject.getLinkPath(),
            debug=False,
            radius=beam_radius,
            max=100,
        )

        if total_mass is not None:
            self.node.addObject(
                "MechanicalMatrixMapper",
                template="Vec3,Rigid3",
                object1=self.cosserat_description_node.MechanicalObject.getLinkPath(),
                object2=self.rigid_base_node.MechanicalObject.getLinkPath(),
                nodeToParse=self.cosserat_frame_node.getLinkPath(),
            )

        #################
        # Collision model
        #################
        self.collision_node = self.cosserat_frame_node.addChild("collision")

        collision_model_kwargs = {}
        if collision_group is not None:
            collision_model_kwargs["group"] = collision_group
        if check_self_collision:
            collision_model_kwargs["selfCollision"] = True
        if collision_contact_stiffness is not None:
            collision_model_kwargs["contactStiffness"] = collision_contact_stiffness

        if collision_model_indices is None:
            collision_model_indices = list(range(number_of_frames + 1))

        self.collision_node.addObject("MechanicalObject", template="Rigid3d")
        self.collision_node.addObject("SubsetMapping", indices=collision_model_indices)

        collision_model_node = self.collision_node.addChild("models")

        collision_model_node.addObject("MechanicalObject", template="Vec3d")
        self.sphere_collision_models = collision_model_node.addObject("SphereCollisionModel", radius=radius, **collision_model_kwargs)
        collision_model_node.addObject("IdentityMapping")

        ##############
        # Visual model
        ##############
        # Tube for visual model
        tube_node = self.cosserat_frame_node.addChild("tube")
        tube_mechanical_object = tube_node.addObject("MechanicalObject")
        tube_topology = tube_node.addObject("QuadSetTopologyContainer")
        tube_node.addObject("QuadSetTopologyModifier")
        tube_node.addObject(
            "Edge2QuadTopologicalMapping",
            nbPointsOnEachCircle=visual_resolution,
            radius=radius,
            input=self.topology_container.getLinkPath(),
            output=tube_topology.getLinkPath(),
        )
        tube_node.addObject(
            "TubularMapping",
            nbPointsOnEachCircle=visual_resolution,
            radius=radius,
            input=self.mechanical_object.getLinkPath(),
            output=tube_mechanical_object.getLinkPath(),
        )

        # OGL visual model
        visual_node = tube_node.addChild("visual")
        self.ogl_model = visual_node.addObject("OglModel", color=[color / 255 for color in rope_color])
        visual_node.addObject("IdentityMapping", input=tube_mechanical_object.getLinkPath(), output=self.ogl_model.getLinkPath())
