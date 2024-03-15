from enum import Enum

MAPPING_PLUGIN_LIST = [
    "Sofa.Component.Mapping.Linear",  # <- [BarycentricMapping, RigidMapping, IdentityMapping]
]


class MappingType(Enum):
    """SOFA names for mapping types.

    **RigidMapping** maps the child's Vec3d points to the parent's Rigid3d points.


    **BarycentricMapping** calculates the closest connections between the parent node's points,
    and the barycenters of the child node's points. This mostly makes sense when mapping
    two different Vec3d objects.


    **IdentityMapping** maps the points between child and parent and assumes that they are exactly
    the same number.

    **SubsetMapping** maps the points between child and parent and assumes that the child is a subset
    """

    RIGID = "RigidMapping"
    BARYCENTRIC = "BarycentricMapping"
    IDENTITY = "IdentityMapping"
    SUBSET = "SubsetMapping"
