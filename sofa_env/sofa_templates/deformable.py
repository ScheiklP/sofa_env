from enum import Enum
import Sofa.Core
import numpy as np

from pathlib import Path
from typing import Callable, Optional, Union, Tuple, List

from sofa_env.sofa_templates.topology import TopologyTypes, TOPOLOGY_PLUGIN_LIST
from sofa_env.sofa_templates.solver import LINEAR_SOLVER_DEFAULT_KWARGS, LinearSolverType, add_solver, ConstraintCorrectionType, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.loader import add_loader, check_file_path, LOADER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST
from sofa_env.sofa_templates.collision import add_collision_model, COLLISION_PLUGIN_LIST
from sofa_env.sofa_templates.materials import Material, add_fem_force_field_from_material, MATERIALS_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.motion_restriction import add_fixed_constraint_in_bounding_box, add_fixed_constraint_to_indices, add_rest_shape_spring_force_field_in_bounding_box, add_rest_shape_spring_force_field_to_indices, MOTION_RESTRICTION_PLUGIN_LIST

DEFORMABLE_PLUGIN_LIST = (
    [
        "Sofa.Component.Constraint.Lagrangian.Correction",  # <- [PrecomputedConstraintCorrection]
    ]
    + MOTION_RESTRICTION_PLUGIN_LIST
    + TOPOLOGY_PLUGIN_LIST
    + SOLVER_PLUGIN_LIST
    + LOADER_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
    + COLLISION_PLUGIN_LIST
    + MATERIALS_PLUGIN_LIST
    + SCENE_HEADER_PLUGIN_LIST
)

FIXTURE_FUNCTIONS = (add_fixed_constraint_to_indices, add_fixed_constraint_in_bounding_box, add_rest_shape_spring_force_field_to_indices, add_rest_shape_spring_force_field_in_bounding_box)


class DeformableObject:
    """Combines all the sofa components to describe a deformable object.

    Notes:
        Parameterizable components such as solvers and collision models are added through add functions (e.g. ``add_solver``).
        This way we avoid having a large number of parameters in the init function where no one remebers which parameter influences which component.
        To change the parameters of a function simply make a partial version of it or write a new one.

    Args:
        parent_node (Sofa.Core.Node): Parent node of the object.
        name (str): Name of the object.
        volume_mesh_path (Union[str, Path]): Path to the volume mesh.
        total_mass (float): Total mass of the deformable object.
        visual_mesh_path (Optional[Union[str, Path]]): Path to the visual surface mesh.
        collision_mesh_path (Optional[Union[str, Path]]): Path to the collision surface mesh.
        rotation (Tuple[float, float, float]): RPY rotations in degrees of the collision model in relation to the parent node. Order is X*Y*Z.
        translation (Tuple[float, float, float]): XYZ translation of the collision model in relation to the parent node.
        scale: (Union[float, Tuple[float, float, float]]): Scale factor for loading the meshes.
        volume_mesh_type (TopologyTypes): Type of the volume mesh (e.g. ``TopologyTypes.TETRA`` for Tetraeder).
        material (Optional[Material]): Description of the material behavior.
        add_deformation_model_func (Callable): Function that can be used to add other deformation forcefields than the ``add_fem_force_field_from_material`` that uses the ``material`` to create an FEM force field.
        add_solver_func (Callable): Function that adds the numeric solvers to the object.
        add_collision_model_func (Callable): Function that defines how the collision surface from ``collision_mesh_path`` is added to the object.
        add_visual_model_func (Callable): Function that defines how the visual surface from ``visual_mesh_path`` is added to the object.
        animation_loop_type (AnimationLoopType): The animation loop of the scene. Required to determine if objects for constraint correction should be added.
        constraint_correction_type (ConstraintCorrectionType): Type of constraint correction for the object, when ``animation_loop_type`` is ``AnimationLoopType.FREEMOTION``.
        show_object (bool): Whether to render the vertices of the volume mesh.
        show_object_scale (float): Render size of the nodes of the volume mesh if ``show_object`` is ``True``.
        show_object_color (Tuple[int, int, int]): RGB values of the nodes of the volume mesh if ``show_object`` is ``True``.

    Examples:
        Changing the parameterizable functions with partial

        >>> from sofa_env.sofa_templates.solver import add_solver
        >>> from functools import partial
        >>> add_solver = partial(add_solver, rayleigh_mass=0.02)
        >>> deformable = DeformableObject(..., add_solver_func=add_solver)

        Defining the material properties of the deformable object

        >>> material = Material(constitutive_model=ConstitutiveModel.COROTATED, poisson_ratio=0.2, young_modulus=5000)
        >>> deformable = DeformableObject(..., material=material)

        Replacing the FEM deformation model with a ``"MeshSpringForceField"``

        >>> from functools import partial
        >>> primitives = [MeshSpringPrimitive.LINE, MeshSpringPrimitive.TRIANGLE]
        >>> stiffness = [1000.0, 500.0]
        >>> damping = [0.0, 5.0]
        >>> mesh_spring_deformation_func = partial(add_mesh_spring_force_field, primitives=primitives, stiffness=stiffness, damping=damping)
        >>> deformable = DeformableObject(..., add_deformation_model_func=mesh_spring_deformation_func)

    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        volume_mesh_path: Union[str, Path],
        total_mass: float,
        visual_mesh_path: Optional[Union[str, Path]] = None,
        collision_mesh_path: Optional[Union[str, Path]] = None,
        rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        volume_mesh_type: TopologyTypes = TopologyTypes.TETRA,
        material: Optional[Material] = None,
        add_deformation_model_func: Callable = add_fem_force_field_from_material,
        add_solver_func: Callable = add_solver,
        add_collision_model_func: Callable = add_collision_model,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        constraint_correction_type: ConstraintCorrectionType = ConstraintCorrectionType.PRECOMPUTED,
        show_object: bool = False,
        show_object_scale: float = 7.0,
        show_object_color: Tuple[int, int, int] = (255, 0, 255),
    ) -> None:

        self.parent_node = parent_node
        self.name = name
        self.node = self.parent_node.addChild(name)

        # Add a loader for the volume mesh
        self.volume_mesh_loader = add_loader(
            attached_to=self.node,
            file_path=volume_mesh_path,
            scale=scale,
            loader_kwargs={
                "translation": translation,
                "rotation": rotation,
            },
        )

        # Add the solvers
        self.time_integration, self.linear_solver = add_solver_func(attached_to=self.node)

        # Add the topology container
        self.topology_container = self.node.addObject(f"{volume_mesh_type.value}SetTopologyContainer", src=self.volume_mesh_loader.getLinkPath())

        # Add the mechanical object that holds the mechanical state of the deformable object
        self.mechanical_object = self.node.addObject(
            "MechanicalObject",
            template="Vec3d",
            showObject=show_object,
            showObjectScale=show_object_scale,
            showColor=show_object_color,
        )

        # Add mass to the object
        self.mass = self.node.addObject("UniformMass", totalMass=total_mass)

        # Add a force field to the object that determines its FEM behavior
        if material is None:
            self.material = Material()
        else:
            self.material = material

        if add_deformation_model_func is add_fem_force_field_from_material:
            self.deformation_force_field = add_deformation_model_func(
                attached_to=self.node,
                material=self.material,
                topology_type=volume_mesh_type,
            )
        else:
            self.deformation_force_field = add_deformation_model_func(attached_to=self.node)

        self.animation_loop_type = animation_loop_type
        self.constraint_correction_type = constraint_correction_type

        # If using the FreeMotionAnimationLoop add a constraint correction object
        if animation_loop_type == AnimationLoopType.FREEMOTION:
            if not constraint_correction_type != ConstraintCorrectionType.UNCOUPLED:
                raise ValueError("Applying UncoupledConstraintCorrection to a deformable object does likely not make sense because (from the official SOFA documentation): makes the approximation that the compliance matrix is diagonal. This is as strong assumption since a diagonal matrix means that all constraints are independent from each other.")
            self.constraint_correction = self.node.addObject(constraint_correction_type.value)

        # Add collision models to the object
        if collision_mesh_path is not None:
            self.collision_model_node = add_collision_model_func(attached_to=self.node, surface_mesh_file_path=check_file_path(collision_mesh_path), scale=scale, translation=translation, rotation=rotation)

        # Add a visual model to the object
        if visual_mesh_path is not None:
            self.visual_model_node = add_visual_model_func(attached_to=self.node, surface_mesh_file_path=check_file_path(visual_mesh_path), scale=scale, translation=translation, rotation=rotation)

    def fix_indices(
        self,
        indices: Union[List[int], np.ndarray],
        fixed_degrees_of_freedom: Tuple[bool, bool, bool] = (True, True, True),
        fixture_func: Callable = add_fixed_constraint_to_indices,
        fixture_func_kwargs: dict = {},
        attach_to_collision_mesh: bool = False,
    ):
        """Fixes the indices of the DeformableObject's mechanical object with a given fixture function.

        Args:
            indices (Union[List[int]), np.ndarray]: Indices of the ``MechanicalObject`` that should be fixed.
            fixed_degrees_of_freedom (Tuple[bool, bool, bool]): Which Cartesian degrees of freedom to fix.
            fixture_func (Callable): Function that adds an object to the SOFA graph that fixes the positions. By default ``FixedProjectiveConstraint``.
            fixture_func_kwargs (dict): Additional kwargs to pass to the ``fixture_func``.
            attach_to_collision_mesh (bool): Whether to add the component to the collision model (surface) or the node that contains the force field (volume).
        """

        if attach_to_collision_mesh:
            if not hasattr(self, "collision_model_node"):
                raise ValueError("Could not find attribute collision_model_node in the deformable object.")
            attached_to = self.collision_model_node
        else:
            attached_to = self.node

        if fixture_func is add_fixed_constraint_to_indices:
            self.fixed_constraint = fixture_func(
                attached_to=attached_to,
                indices=indices,
                fixed_degrees_of_freedom=fixed_degrees_of_freedom,
                **fixture_func_kwargs,
            )

        else:
            self.fixed_constraint = fixture_func(
                attached_to=attached_to,
                indices=indices,
                **fixture_func_kwargs,
            )

    def fix_indices_in_bounding_box(
        self,
        min: Tuple[float, float, float],
        max: Tuple[float, float, float],
        fixed_degrees_of_freedom: Tuple[bool, bool, bool] = (True, True, True),
        fixture_func: Callable = add_fixed_constraint_in_bounding_box,
        fixture_func_kwargs: dict = {},
        show_bounding_box: bool = False,
        attach_to_collision_mesh: bool = False,
    ):
        """Fixes the indices of the DeformableObject's mechanical object inside a bounding box with a given fixture function.

        Args:
            min (Tuple[float), float, float]: Cartesian minimum values of the bounding box in which to fix the positions.
            max (Tuple[float), float, float]: Cartesian maximum values of the bounding box in which to fix the positions.
            fixed_degrees_of_freedom (Tuple[bool, bool, bool]): Which Cartesian degrees of freedom to fix.
            fixture_func (Callable): Function that adds an object to the SOFA graph that fixes the positions. By default ``FixedProjectiveConstraint``.
            fixture_func_kwargs (dict): Additional kwargs to pass to the ``fixture_func``.
            show_bounding_box (bool): Whether to render the bounding box.
            attach_to_collision_mesh (bool): Whether to add the component to the collision model (surface) or the node that contains the force field (volume).
        """

        if attach_to_collision_mesh:
            if not hasattr(self, "collision_model_node"):
                raise ValueError("Could not find attribute collision_model_node in the deformable object.")
            attached_to = self.collision_model_node
        else:
            attached_to = self.node

        if fixture_func is add_fixed_constraint_in_bounding_box:
            self.fixed_constraint = fixture_func(
                attached_to=attached_to,
                min=min,
                max=max,
                fixed_degrees_of_freedom=fixed_degrees_of_freedom,
                show_bounding_box=show_bounding_box,
                **fixture_func_kwargs,
            )
        else:
            self.fixed_constraint = fixture_func(
                attached_to=attached_to,
                min=min,
                max=max,
                show_bounding_box=show_bounding_box,
                **fixture_func_kwargs,
            )


class CuttableDeformableObject:
    """Combines all the sofa components to describe a cuttable deformable object.

    Notes:
        Parameterizable components such as solvers are added through add functions (e.g. ``add_solver``).
        This way we avoid having a large number of parameters in the init function where no one remebers which parameter influences which component.
        To change the parameters of a function simply make a partial version of it or write a new one.

    Args:
        parent_node (Sofa.Core.Node): Parent node of the collision model.
        name (str): Name of the object.
        volume_mesh_path (Union[str, Path]): Path to the volume mesh.
        total_mass (float): Total mass of the deformable object.
        rotation (Tuple[float, float, float]): RPY rotations in degrees of the collision model in relation to the parent node. Order is X*Y*Z.
        translation (Tuple[float, float, float]): XYZ translation of the collision model in relation to the parent node.
        scale: (Union[float, Tuple[float, float, float]]): Scale factor for loading the meshes.
        volume_mesh_type (TopologyTypes): Type of the volume mesh (e.g. TopologyTypes.TETRA for Tetraeder).
        material (Optional[Material]): Description of the material behavior.
        add_deformation_model_func (Callable): Function that can be used to add other deformation forcefields than the ``add_fem_force_field_from_material`` that uses the ``material`` to create an FEM force field.
        add_solver_func (Callable): Function that adds the numeric solvers to the object.
        animation_loop_type (AnimationLoopType): The animation loop of the scene. Required to determine if objects for constraint correction should be added.
        contact_stiffness (Union[float, int]): Stiffness of the collision reaction.

    Examples:
        Changing the parameterizable functions with partial

        >>> from sofa_env.sofa_templates.solver import add_solver
        >>> from functools import partial
        >>> add_solver = partial(add_solver, rayleigh_mass=0.02)
        >>> cuttable_object = CuttableDeformableObject(..., add_solver_func=add_solver)
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        volume_mesh_path: Union[str, Path],
        total_mass: float,
        rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        volume_mesh_type: TopologyTypes = TopologyTypes.TETRA,
        material: Optional[Material] = None,
        add_deformation_model_func: Callable = add_fem_force_field_from_material,
        add_solver_func: Callable = add_solver,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        contact_stiffness: Union[float, int] = 100,
    ) -> None:

        self.parent_node = parent_node
        self.name = name
        self.node = self.parent_node.addChild(name)

        if volume_mesh_type != TopologyTypes.TETRA:
            raise NotImplementedError

        # add a loader for the volume mesh
        self.volume_mesh_loader = add_loader(
            attached_to=self.node,
            file_path=volume_mesh_path,
            scale=scale,
            loader_kwargs={
                "translation": translation,
                "rotation": rotation,
            },
        )

        # Add the solvers
        self.time_integration, self.linear_solver = add_solver_func(attached_to=self.node)

        # Add the topology container
        self.topology_container = self.node.addObject("TetrahedronSetTopologyContainer", src=self.volume_mesh_loader.getLinkPath())
        self.node.addObject("TetrahedronSetTopologyModifier")
        self.node.addObject("TetrahedronSetGeometryAlgorithms")

        # Add the mechanical object that holds the mechanical state of the deformable object
        self.mechanical_object = self.node.addObject("MechanicalObject", template="Vec3d")

        # Add mass to the object. Does not work with UniformMass for fome reason...
        self.mass = self.node.addObject("DiagonalMass", totalMass=total_mass)

        # Add a force field to the object that determines its FEM behavior
        if material is None:
            self.material = Material()

        if add_deformation_model_func is add_fem_force_field_from_material:
            self.deformation_force_field = add_deformation_model_func(
                attached_to=self.node,
                material=self.material,
                topology_type=volume_mesh_type,
            )
        else:
            self.deformation_force_field = add_deformation_model_func(attached_to=self.node)

        self.animation_loop_type = animation_loop_type
        self.constraint_correction_type = ConstraintCorrectionType.GENERIC

        # If using the FreeMotionAnimationLoop add a constraint correction object
        if animation_loop_type == AnimationLoopType.FREEMOTION:
            self.constraint_correction = self.node.addObject(self.constraint_correction_type.value)

        # Add collision models to the object
        self.surface_node = self.node.addChild("surface")

        self.surface_topology_container = self.surface_node.addObject("TriangleSetTopologyContainer")
        self.surface_node.addObject("TriangleSetTopologyModifier")
        self.surface_node.addObject("TriangleSetGeometryAlgorithms")
        self.surface_node.addObject("Tetra2TriangleTopologicalMapping", input=self.topology_container.getLinkPath(), output=self.surface_topology_container.getLinkPath())
        collision_model_types = ["TriangleCollisionModel", "PointCollisionModel", "LineCollisionModel"]
        collision_model_kwargs = {"tags": "CarvingSurface"}
        if animation_loop_type == AnimationLoopType.DEFAULT:
            collision_model_kwargs["contactStiffness"] = contact_stiffness
        for collision_model_type in collision_model_types:
            self.surface_node.addObject(collision_model_type, **collision_model_kwargs)

        self.visual_node = self.surface_node.addChild("visual")
        self.ogl_model = self.visual_node.addObject("OglModel")
        self.visual_node.addObject("IdentityMapping", input=self.mechanical_object.getLinkPath(), output=self.ogl_model.getLinkPath())

    def fix_indices(
        self,
        indices=Union[List[int], np.ndarray],
        fixed_degrees_of_freedom: Tuple[bool, bool, bool] = (True, True, True),
        fixture_func: Callable = add_fixed_constraint_to_indices,
        fixture_func_kwargs: dict = {},
    ):
        """Fixes the indices of the DeformableObject's mechanical object with a given fixture function."""

        if fixture_func in FIXTURE_FUNCTIONS:
            self.fixed_constraint = fixture_func(attached_to=self.node, indices=indices, fixed_degrees_of_freedom=fixed_degrees_of_freedom, **fixture_func_kwargs)
        else:
            self.fixed_constraint = fixture_func(attached_to=self.node, **fixture_func_kwargs)

    def fix_indices_in_bounding_box(
        self,
        min: Tuple[float, float, float],
        max: Tuple[float, float, float],
        fixed_degrees_of_freedom: Tuple[bool, bool, bool] = (True, True, True),
        fixture_func: Callable = add_fixed_constraint_in_bounding_box,
        fixture_func_kwargs: dict = {},
        show_bounding_box: bool = True,
    ) -> List[int]:
        """Fixes the indices of the DeformableObject's mechanical object inside a bounding box with a given fixture function."""

        if fixture_func in FIXTURE_FUNCTIONS:
            self.fixed_constraint, indices = fixture_func(attached_to=self.node, min=min, max=max, fixed_degrees_of_freedom=fixed_degrees_of_freedom, show_bounding_box=show_bounding_box, **fixture_func_kwargs)
        else:
            self.fixed_constraint, indices = fixture_func(attached_to=self.node, **fixture_func_kwargs)

        return indices


def rigidify(
    deformable_object: DeformableObject,
    rigidification_indices: Union[List[int], List[List[int]]],
    rigid_object_pose: Optional[Union[List[int], List[List[int]]]] = None,
) -> Sofa.Core.Node:
    """Rigidify parts of a deformable object
    Adapted version of https://github.com/SofaDefrost/STLIB/blob/master/python3/src/stlib3/physics/mixedmaterial/rigidification.py

    Uses the SubsetMultiMapping component of SOFA to split the behavior of the deformable tissue into a partly rigid object.
    A total of three new MechanicalObjects are createt.
    1. a Vec3d MechanicalObject that holds the points that are to remain deformable
    2. a Rigid3d MechanicalObject that holds one Rigid3d pose (position + orientation) per specified subset of rigidified parts as a reference.
    3. A Vec3d MechanicalObject that holds the points that are to be rigidified

    MechanicalObjects 2 and 3 are mapped with a RigidMapping, so that they move together.

    Args:
        deformable_object (DeformableObject): The deformable object you want to rigidify.
        rigidification_indices (Union[List[int], List[List[int]]]): A list of indices on the volume mesh of the deformable object that should be rigidified. Can also be split into multiple subsets by passing a list of lists of indices.
        rigid_object_pose (Optional[Union[List[int], List[List[int]]]]): Optional reference poses for the rigidified parts. If none, the reference pose will be set to the barycenters of the rigidified subsets.

    Returns:
        mixedmaterial_node (Sofa.Core.Node): The sofa node that contains the sofa node of the original deformable tissue, as well as new nodes for rigidified and deformable parts of the object.
    """

    mixedmaterial_node = deformable_object.parent_node.addChild(f"rigidified_{deformable_object.name.value if isinstance(deformable_object.name, Sofa.Core.DataString) else deformable_object.name}")

    volume_mesh_positions = np.array([list(position) for position in deformable_object.topology_container.position.array()], dtype=np.float64)
    volume_mesh_indices = np.array(list(range(deformable_object.topology_container.nbPoints.value)), dtype=np.int64)

    # Make sure the indices are a list of lists
    if not isinstance(rigidification_indices[0], list):
        rigidification_indices = [rigidification_indices]

    # Make sure the dimensions match if poses for the rigid objects were passed
    # Should pass one pose for each subset of indices
    if rigid_object_pose is not None:
        # make sure the poses are a list of lists
        if not isinstance(rigid_object_pose[0], list):
            rigid_object_pose = [rigid_object_pose]

        if not len(rigid_object_pose) == len(rigidification_indices):
            raise ValueError(f"If you want to specify the reference pose for rigidified parts of the deformable object, please pass as many poses, as you passed index subsets. Found {len(rigidification_indices)} index subsets and {len(rigid_object_pose)} reference poses.")

    subset_reference_poses = []
    flat_rigidification_indices = []
    subset_map = []

    # For each list (subset) of indices
    for subset_number in range(len(rigidification_indices)):

        subset_indices = rigidification_indices[subset_number]
        if not len(subset_indices) > 0:
            print(f"[WARNING]: Got an empty list of indices to rigidify for subset {subset_number}. Will skip this subset.")
            continue

        # If there are no reference poses for the rigidified parts, create them from the barycenters of the indices that will be rigidified
        if rigid_object_pose is None:
            subset_positions = [volume_mesh_positions[index] for index in subset_indices]
            subset_reference_orientation = np.array([0.0, 0.0, 0.0, 1.0])
            subset_reference_position = np.mean(subset_positions, axis=0)
        else:
            if len(rigid_object_pose[subset_number]) == 3:
                subset_reference_orientation = np.array([0.0, 0.0, 0.0, 1.0])
                subset_reference_position = rigid_object_pose[subset_number]
            elif len(rigid_object_pose[subset_number]) == 7:
                subset_reference_orientation = rigid_object_pose[subset_number][3:]
                subset_reference_position = rigid_object_pose[subset_number][:3]
            else:
                raise ValueError(f"[ERROR]: Please pass the reference poses for the rigidified subsets either as position (3 values) or pose (7 values). Got {len(rigid_object_pose[subset_number])} for subset {subset_number}.")

        subset_reference_poses.append(list(subset_reference_position.tolist()) + list(subset_reference_orientation.tolist()))
        flat_rigidification_indices.extend(subset_indices)
        subset_map.extend([subset_number] * len(subset_indices))

    deformable_indices = list(filter(lambda x: x not in flat_rigidification_indices, volume_mesh_indices))

    kd_tree = {index: None for index in volume_mesh_indices}

    # Map the indices of the original volume mesh to [object 0 (deformable), index in object 0 (locally)]
    kd_tree.update({global_index: [0, local_index] for local_index, global_index in enumerate(deformable_indices)})

    # Map the indices of the original volume mesh to [object 1 (rigidified), index in object 1 (locally)]
    kd_tree.update({global_index: [1, local_index] for local_index, global_index in enumerate(flat_rigidification_indices)})

    # Flatten out the list to [object, local_index, object, local_index, ...] so that the index of the list corresponds to the global index in the volume mesh
    flat_index_pairs = [value for pair in kd_tree.values() for value in pair]

    deformable_parts = mixedmaterial_node.addChild("deformable")
    deformable_mechanical_object = deformable_parts.addObject("MechanicalObject", template="Vec3d", position=[list(volume_mesh_positions[index].tolist() for index in deformable_indices)])

    rigid_parts = mixedmaterial_node.addChild("rigid")
    rigid_parts.addObject("MechanicalObject", template="Rigid3d", position=subset_reference_poses)

    rigid_subsets = rigid_parts.addChild("rigid_subsets")
    rigid_mechanical_object = rigid_subsets.addObject("MechanicalObject", template="Vec3d", position=[list(volume_mesh_positions[index].tolist() for index in flat_rigidification_indices)])
    rigid_subsets.addObject("RigidMapping", globalToLocalCoords=True, rigidIndexPerPoint=subset_map)

    deformable_object.node.addObject(
        "SubsetMultiMapping",
        template="Vec3d,Vec3d",
        input=[deformable_mechanical_object.getLinkPath(), rigid_mechanical_object.getLinkPath()],
        output=deformable_object.mechanical_object.getLinkPath(),
        indexPairs=flat_index_pairs,
    )

    # Moves solvers and constraint correction from the original node to the new mixedmaterial node
    deformable_object.node.removeObject(deformable_object.time_integration)
    deformable_object.node.removeObject(deformable_object.linear_solver)
    mixedmaterial_node.addObject(deformable_object.time_integration)
    mixedmaterial_node.addObject(deformable_object.linear_solver)
    if deformable_object.animation_loop_type == AnimationLoopType.FREEMOTION:
        deformable_object.node.removeObject(deformable_object.constraint_correction)
        # A new precomputed constraint correction for the deformable parts
        deformable_parts.addObject(deformable_object.constraint_correction_type.value)
        # Uncoupled constraint correction for the rigidified parts
        rigid_parts.addObject(ConstraintCorrectionType.UNCOUPLED.value)

    # Add the original node as children to both deformable and rigid nodes
    rigid_subsets.addChild(deformable_object.node)
    deformable_parts.addChild(deformable_object.node)

    return mixedmaterial_node


class MeshSpringPrimitive(Enum):
    """Primitive elements that can be used in a ``"MeshSpringForceField"``"""

    LINE = "lines"
    TRIANGLE = "triangles"
    QUAD = "quads"
    TETRA = "tetras"
    CUBE = "cubes"


def add_mesh_spring_force_field(attached_to: Sofa.Core.Node, primitives: List[MeshSpringPrimitive], stiffness: List[float], damping: List[float]) -> Sofa.Core.Object:
    """Adds a ``"MeshSpringForceField"`` to a node.

    Args:
        primitives (List[MeshSpringPrimitive]): A list of primitives for which stiffness and damping should be set in the ``"MeshSpringForceField"``.
        stiffness (List[float]): Stiffness values for the springs for each primitive.
        damping (List[float]): Damping values for the springs for each primitive.

    Returns:
        force_field (Sofa.Core.Object): The ``"MeshSpringForceField"`` object.
    """

    if not (len(primitives) == len(stiffness) and len(primitives) == len(damping)):
        raise ValueError(f"Received differnt list sizes for primitives, stiffness, and damping. {len(primitives)=}, {len(stiffness)=}, {len(damping)=}")

    kwargs = {}
    for index in range(len(primitives)):
        kwargs[f"{primitives[index].value}Stiffness"] = stiffness[index]
        kwargs[f"{primitives[index].value}Damping"] = damping[index]

    force_field = attached_to.addObject("MeshSpringForceField", **kwargs)

    return force_field


class SimpleDeformableObject:
    """A simplified deformable object that is created from a list of positions and tetrahedra indices.

    Args:
        parent_node (Sofa.Core.Node): Parent node of the object.
        name (str): Name of the object.
        positions (np.ndarray): Cartesian positions (N, 3) of the deformable object.
        tetrahedra (np.ndarray): Indices that specify the tetrahedra of the deformable object.
        total_mass (float): Total mass of the deformable object.
        young_modulus (float): Young modulus of the deformable object.
        poisson_ratio (float): Poisson ratio of the deformable object.
        animation_loop_type (AnimationLoopType): The animation loop of the scene. Required to determine if objects for constraint correction should be added.
        ode_solver_rayleigh_stiffness (float): See documentation of ``OdeSolverType``.
        ode_solver_rayleigh_mass (float): See documentation of ``OdeSolverType``.
        linear_solver_kwargs (Optional[dict]): Additional keyword arguments to the LinearSolverType. If ``None``, read from ``LINEAR_SOLVER_DEFAULT_KWARGS``.
        show_object (bool): Whether to render the vertices of the volume mesh.
        show_object_scale (float): Render size of the nodes of the volume mesh if ``show_object`` is ``True``.
        color (Tuple[float, float, float]): RGB values of the nodes of the volume mesh if ``show_object`` is ``True``.
        collision_group (int): The group for which collisions with this object should be ignored. Value has to be set since the jaws and shaft must belong to the same group.
        cuttable (bool): Whether to add this object to the cuttable objects.
    """

    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        name: str,
        positions: np.ndarray,
        tetrahedra: np.ndarray,
        total_mass: float,
        young_modulus: float,
        poisson_ratio: float,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        ode_solver_rayleigh_stiffness: float = 0.1,
        ode_solver_rayleigh_mass: float = 0.1,
        linear_solver_kwargs: Optional[dict] = None,
        show_object: bool = False,
        show_object_scale: float = 1.0,
        color: Optional[Tuple[float, float, float]] = None,
        collision_group: Optional[int] = None,
        cuttable: bool = False,
    ) -> None:

        self.parent_node = parent_node
        self.name = name
        self.node = self.parent_node.addChild(name)

        # Ode solver
        self.node.addObject("EulerImplicitSolver", rayleighMass=ode_solver_rayleigh_mass, rayleighStiffness=ode_solver_rayleigh_stiffness)

        # Linear solver
        if animation_loop_type == AnimationLoopType.FREEMOTION:
            if linear_solver_kwargs is None:
                linear_solver_kwargs = LINEAR_SOLVER_DEFAULT_KWARGS[LinearSolverType.SPARSELDL]
            self.node.addObject("SparseLDLSolver", **linear_solver_kwargs)
        else:
            if linear_solver_kwargs is None:
                linear_solver_kwargs = LINEAR_SOLVER_DEFAULT_KWARGS[LinearSolverType.CG]
            self.node.addObject("CGLinearSolver", **linear_solver_kwargs)

        # Mechanical behavior
        self.topology_container = self.node.addObject("TetrahedronSetTopologyContainer", tetrahedra=tetrahedra)
        self.node.addObject("TetrahedronSetTopologyModifier")
        self.mechanical_object = self.node.addObject("MechanicalObject", position=positions, showObject=show_object, showObjectScale=show_object_scale)
        self.node.addObject("FastTetrahedralCorotationalForceField", youngModulus=young_modulus, poissonRatio=poisson_ratio)
        self.node.addObject("UniformMass", totalMass=total_mass)

        if animation_loop_type == AnimationLoopType.FREEMOTION:
            self.node.addObject("LinearSolverConstraintCorrection")

        # Collision
        self.collision_node = self.node.addChild("collision")
        self.collision_node.addObject("TriangleSetTopologyContainer")
        self.collision_node.addObject("TriangleSetTopologyModifier")
        self.collision_node.addObject("Tetra2TriangleTopologicalMapping")
        collision_model_kwargs = {}
        if collision_group is not None:
            collision_model_kwargs["group"] = collision_group
        if cuttable:
            collision_model_kwargs["tags"] = "CarvingSurface"
        self.collision_node.addObject("TriangleCollisionModel", **collision_model_kwargs)

        # Visual
        self.visual_node = self.node.addChild("visual")
        if color is not None:
            self.visual_node.addObject("OglModel", color=color)
        else:
            self.visual_node.addObject("OglModel")
        self.visual_node.addObject("IdentityMapping")
