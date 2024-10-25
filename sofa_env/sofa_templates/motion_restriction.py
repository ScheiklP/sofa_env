from typing import Tuple, List, Union, Optional

import numpy as np

import Sofa.Core

from sofa_env.utils.math_helper import euler_to_rotation_matrix

MOTION_RESTRICTION_PLUGIN_LIST = [
    "Sofa.Component.MechanicalLoad",
    "Sofa.Component.Engine.Select",
    "Sofa.Component.Constraint.Projective",  # <- [FixedProjectiveConstraint]
    "Sofa.Component.SolidMechanics.Spring",  # <- [RestShapeSpringsForceField]
    "Sofa.Component.Engine.Select",  # <- [BoxROI]
]


def add_bounding_box(
    attached_to: Sofa.Core.Node,
    min: Union[Tuple[float, float, float], np.ndarray],
    max: Union[Tuple[float, float, float], np.ndarray],
    translation: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
    rotation: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
    show_bounding_box: bool = False,
    show_bounding_box_scale: float = 1.0,
    name: Optional[str] = None,
    extra_kwargs: Optional[dict] = None,
) -> Sofa.Core.Object:
    """Adds a bounding box to a node. Can be used to filter the indices of a mechanical object that lie within a box.

    Notes:
        The node ``attached_to`` must have a mechanical object.

    Args:
        attached_to (Sofa.Core.Node): Parent node of the bounding box.
        min (Union[Tuple[float, float, float], np.ndarray]): Lower limits of the bounding box.
        max (Union[Tuple[float, float, float], np.ndarray]): Upper limit of the bounding box.
        translation (Optional[Union[Tuple[float, float, float], np.ndarray]]): Optional translation applied to the (rotated) bounding box.
        rotation (Optional[Union[Tuple[float, float, float], np.ndarray]]): Optional rotation applied to the bounding box specified by XYZ euler angles in degrees.
        show_bounding_box (bool): Whether to render the bounding box.
        show_bounding_box_scale (float): Size of the rendered bounding box if ``show_bounding_box`` is ``True``.
        name (Optional[str]): Optional name of the bounding box.
        extra_kwargs (Optional[dict]): Optional keyword arguments passed to the bounding box component.
    """
    box = np.concatenate([min, max])
    if box.ndim != 1:
        raise ValueError(f"Invalid ndim of min/max argument(s). Expected ndim of concatenation [min, max] to be 1 but got: {box.ndim}.")
    if box.shape[0] != 6:
        raise ValueError(f"Invalid shape of min/max argument(s). Expected concatenation of [min, max] to have shape (6,) but got: {box.shape}")

    box_min = box[0:3]
    box_max = box[3:6]
    if not np.all(box_min < box_max):
        raise ValueError(f"Invalid min/max argument(s). Expected min < max (elementwise) but got: {box_min < box_max = }")

    kwargs = {
        "drawBoxes": show_bounding_box,
        "drawSize": show_bounding_box_scale,
    }
    if name is not None:
        kwargs["name"] = name

    if translation is None and rotation is None:
        kwargs["box"] = box
    else:
        # see: https://github.com/sofa-framework/sofa/blob/master/Sofa/Component/Engine/Select/src/sofa/component/engine/select/BoxROI.inl
        # OBB is defined by 3 points (p0, p1, p2) and a depth distance.
        # A parallelogram will be defined by (p0, p1, p2, p3 = p0 + (p2-p1)).
        # The box will finally correspond to the parallelogram extrusion of depth/2 along its normal
        # and depth/2 in the opposite direction.
        # OBB in local SOFA frame (x right, y out, z down):
        #
        #  p0+-----------+p1
        #    |           |
        #    |           |
        #    |     +-----+-->x
        #    |     |     |
        #    |     |     |
        #  p3+-----+-----+p2
        #          |
        #          v
        #          z
        offset = np.zeros(3) if translation is None else np.array(translation)
        rotation_angles = np.zeros(3) if rotation is None else np.array(rotation)
        box_to_global = lambda v: euler_to_rotation_matrix(rotation_angles) @ v + offset
        y_depth = box_max[1] - box_min[1]
        y_center = box_min[1] + 0.5 * y_depth
        p0 = box_to_global(np.array([box_min[0], y_center, box_min[2]]))
        p1 = box_to_global(np.array([box_max[0], y_center, box_min[2]]))
        p2 = box_to_global(np.array([box_max[0], y_center, box_max[2]]))
        # p3 = p0 + (p2 - p1)
        obb = np.hstack([p0, p1, p2, y_depth])
        kwargs["orientedBox"] = obb

    if extra_kwargs is not None:
        kwargs.update(extra_kwargs)

    return attached_to.addObject("BoxROI", **kwargs)


def add_fixed_constraint_to_indices(
    attached_to: Sofa.Core.Node,
    indices: Union[List[int], np.ndarray],
    fixed_degrees_of_freedom: Tuple[bool, bool, bool] = (True, True, True),
) -> Sofa.Core.Object:
    """Fixes the given indices of the given node's mechanical object to their initial position.

    Notes:
        Technically fixes the initial velocity of the points. So if the velocity is non-zero in time step 0, the indices will continue travelling at that velocity. You can add ``projectVelocity=True`` to the FixedProjectiveConstraint and PartialFixedConstraint.


    Args:
        attached_to (Sofa.Core.Node): Parent node of the bounding box.
        indices (Union[List[int], np.ndarray]): Which indices of the object should be fixed.
        fixed_degrees_of_freedom (Tuple[bool, bool, bool]): Which of the axis to restrict. XYZ.
    """

    if all(fixed_degrees_of_freedom):
        return attached_to.addObject("FixedProjectiveConstraint", indices=indices)
    else:
        return attached_to.addObject("PartialFixedConstraint", indices=indices, fixedDirections=fixed_degrees_of_freedom)


def add_fixed_constraint_in_bounding_box(
    attached_to: Sofa.Core.Node,
    min: Union[Tuple[float, float, float], np.ndarray],
    max: Union[Tuple[float, float, float], np.ndarray],
    translation: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
    rotation: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
    show_bounding_box: bool = False,
    show_bounding_box_scale: float = 1.0,
    bounding_box_name: Optional[str] = None,
    fixed_degrees_of_freedom: Tuple[bool, bool, bool] = (True, True, True),
) -> Sofa.Core.Object:
    """Finds the indices of the given node's mechanical object in a bounding box and fixes them to their initial position.

    Notes:
        Technically fixes the initial velocity of the points. So if the velocity is non-zero in time step 0, the indices will continue travelling at that velocity. You can add ``projectVelocity=True`` to the FixedProjectiveConstraint and PartialFixedConstraint.

    Args:
        attached_to (Sofa.Core.Node): Parent node of the bounding box.
        min (Union[Tuple[float, float, float], np.ndarray]): Lower limits of the bounding box.
        max (Union[Tuple[float, float, float], np.ndarray]): Upper limit of the bounding box.
        translation (Optional[Union[Tuple[float, float, float], np.ndarray]]): Optional translation applied to the (rotated) bounding box.
        rotation (Optional[Union[Tuple[float, float, float], np.ndarray]]): Optional rotation applied to the bounding box specified by XYZ euler angles in degrees.
        show_bounding_box (bool): Whether to render the bounding box.
        show_bounding_box_scale (float): Size of the rendered bounding box if show_bounding_box is True.
        bounding_box_name (Optional[str]): Optional name of the bounding box.
        fixed_degrees_of_freedom (Tuple[bool, bool, bool]): Which of the axis to restrict. XYZ.
    """

    bounding_box = add_bounding_box(
        attached_to=attached_to,
        min=min,
        max=max,
        translation=translation,
        rotation=rotation,
        show_bounding_box=show_bounding_box,
        show_bounding_box_scale=show_bounding_box_scale,
        name=bounding_box_name,
    )

    if all(fixed_degrees_of_freedom):
        return attached_to.addObject("FixedProjectiveConstraint", indices=f"{bounding_box.getLinkPath()}.indices"), bounding_box.indices.toList()
    else:
        return attached_to.addObject("PartialFixedConstraint", indices=f"{bounding_box.getLinkPath()}.indices", fixedDirections=fixed_degrees_of_freedom)


def add_rest_shape_spring_force_field_to_indices(
    attached_to: Sofa.Core.Node,
    indices: Union[List[int], np.ndarray],
    stiffness: float = 1e4,
    angular_stiffness: float = 1e4,
    show_springs: bool = False,
) -> Sofa.Core.Object:
    """Adds springs between indices of the given node's mechanical object and their initial positions.

    Args:
        attached_to (Sofa.Core.Node): Parent node of the bounding box.
        indices (Union[List[int], np.ndarray]): Which indices of the object should be fixed.
        stiffness (float): Spring stiffness in lenght.
        angular_stiffness (float): Angular stiffness of the springs.
        show_springs (bool): Whether to render the springs.
    """

    return attached_to.addObject("RestShapeSpringsForceField", stiffness=stiffness, angularStiffness=angular_stiffness, points=indices, drawSpring=show_springs)


def add_rest_shape_spring_force_field_in_bounding_box(
    attached_to: Sofa.Core.Object,
    min: Union[Tuple[float, float, float], np.ndarray],
    max: Union[Tuple[float, float, float], np.ndarray],
    translation: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
    rotation: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
    show_bounding_box: bool = False,
    show_bounding_box_scale: float = 1.0,
    bounding_box_name: Optional[str] = None,
    stiffness: float = 1e4,
    angular_stiffness: float = 1e4,
    show_springs: bool = False,
) -> Sofa.Core.Object:
    """Finds the indices of the given node's mechanical object in a bounding box and adds springs between them and their initial positions.

    Args:
        attached_to (Sofa.Core.Node): Parent node of the bounding box.
        min (Union[Tuple[float, float, float], np.ndarray]): Lower limits of the bounding box.
        max (Union[Tuple[float, float, float], np.ndarray]): Upper limit of the bounding box.
        translation (Optional[Union[Tuple[float, float, float], np.ndarray]]): Optional translation applied to the (rotated) bounding box.
        rotation (Optional[Union[Tuple[float, float, float], np.ndarray]]): Optional rotation applied to the bounding box specified by XYZ euler angles in degrees.
        show_bounding_box (bool): Whether to render the bounding box.
        show_bounding_box_scale (float): Size of the rendered bounding box if ``show_bounding_box`` is ``True``.
        bounding_box_name (Optional[str]): Optional name of the bounding box.
        stiffness (float): Spring stiffness in length.
        angular_stiffness (float): Angular stiffness of the springs.
        show_springs (bool): Whether to render the springs.
    """

    bounding_box = add_bounding_box(
        attached_to=attached_to,
        min=min,
        max=max,
        translation=translation,
        rotation=rotation,
        show_bounding_box=show_bounding_box,
        show_bounding_box_scale=show_bounding_box_scale,
        name=bounding_box_name,
    )

    return attached_to.addObject(
        "RestShapeSpringsForceField",
        stiffness=stiffness,
        angularStiffness=angular_stiffness,
        points=f"{bounding_box.getLinkPath()}.indices",
        drawSpring=show_springs,
    )
