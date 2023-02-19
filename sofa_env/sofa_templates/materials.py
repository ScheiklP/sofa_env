import Sofa.Core
from enum import Enum
from sofa_env.sofa_templates.topology import TopologyTypes, TOPOLOGY_PLUGIN_LIST

MATERIALS_PLUGIN_LIST = [
    "Sofa.Component.SolidMechanics.FEM.Elastic",  # <- [FastTetrahedralCorotationalForceField]
] + TOPOLOGY_PLUGIN_LIST


class ConstitutiveModel(Enum):
    """Constitutive models for linear and non-linear elastic materials.

    Linear and Corotated material describe a linear relationship between strain and stress.
    [StVenantKirchhoff](https://en.wikipedia.org/wiki/Hyperelastic_material#Saint_Venant%E2%80%93Kirchhoff_model) and [NeoHookean](https://en.wikipedia.org/wiki/Neo-Hookean_solid) are hyperelastic materials -> non-linear relationship between strain and stress.
    """

    LINEAR = "Linear"
    COROTATED = "Corotated"
    STVENANTKIRCHHOFF = "StVenantKirchhoff"
    NEOHOOKEAN = "NeoHookean"


class Material:
    """Describes a material based on its constitutive model.

    Sets ``fem_parameters`` that can be used to create FEM force fields.
    """

    def __init__(
        self,
        constitutive_model: ConstitutiveModel = ConstitutiveModel.COROTATED,
        poisson_ratio: float = 0.3,
        young_modulus: int = 4000,
    ):

        assert isinstance(constitutive_model, ConstitutiveModel), f"Expected a ConstitutiveModel, but received <<{type(constitutive_model)}: {constitutive_model}>>."

        self.constitutive_model = constitutive_model
        self.poisson_ratio = poisson_ratio
        self.young_modulus = young_modulus

        if self.constitutive_model == ConstitutiveModel.LINEAR:
            self.fem_parameters = {
                "youngModulus": young_modulus,
                "poissonRatio": poisson_ratio,
                "method": "small",
            }
        elif self.constitutive_model == ConstitutiveModel.COROTATED:
            self.fem_parameters = {
                "youngModulus": young_modulus,
                "poissonRatio": poisson_ratio,
                "method": "large",
            }
        elif self.constitutive_model == ConstitutiveModel.NEOHOOKEAN:
            # convert to lame parameters
            lame_parameter_mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
            lame_parameter_k = young_modulus / (3 * (1.0 - 2.0 * poisson_ratio))

            self.fem_parameters = {
                "ParameterSet": [lame_parameter_mu, lame_parameter_k],
                "materialName": constitutive_model.value,
            }
        elif self.constitutive_model == ConstitutiveModel.STVENANTKIRCHHOFF:
            # convert to lame parameters
            lame_parameter_mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
            lame_parameter_l = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))

            self.fem_parameters = {
                "ParameterSet": [lame_parameter_mu, lame_parameter_l],
                "materialName": constitutive_model.value,
            }
        else:
            raise NotImplementedError


def add_fem_force_field_from_material(attached_to: Sofa.Core.Node, material: Material, topology_type: TopologyTypes = TopologyTypes.TETRA) -> Sofa.Core.Object:
    """Adds a FEM force field based on the passed material type and the object topology.

    Linear and Corotated materials will create a <Tetrahedron>FEMForceField (FastTetrahedralCorotationalForceField for Corotated and Tetrahedron), while StVenantKirchhoff and NeoHookean materials will create a <Tetrahedron>HyperelasticityFEMForceField.
    Linear materials are suited for small displacements, while Corotated materials allow large displacements.
    """

    if material.constitutive_model in (ConstitutiveModel.LINEAR, ConstitutiveModel.COROTATED):
        if topology_type == TopologyTypes.TETRA:
            force_field = attached_to.addObject("FastTetrahedralCorotationalForceField", **material.fem_parameters)
        else:
            force_field = attached_to.addObject(f"{topology_type.value}FEMForceField", **material.fem_parameters)
    elif material.constitutive_model in (ConstitutiveModel.NEOHOOKEAN, ConstitutiveModel.STVENANTKIRCHHOFF):
        force_field = attached_to.addObject(f"{topology_type.value}HyperelasticityFEMForceField", **material.fem_parameters)
    else:
        raise NotImplementedError

    return force_field
