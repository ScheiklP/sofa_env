# SOFA_ENV
Gym environments for reinforcement learning based on the [SOFA Simulation Framework](https://www.sofa-framework.org/).
This repository is part of "LapGym - An Open Source Framework for Reinforcement Learning in Robot-Assisted Laparoscopic Surgery".
See [LapGym](https://www.jmlr.org/papers/v24/23-0207.html) for the paper and [lap_gym](https://github.com/ScheiklP/lap_gym) for the top level repository.

## Getting Started
Tested under Ubuntu {18.04, 20.04, 22.04}, Fedora {36, 37}, and Windows (WSL).
MacOS (on Intel and Apple Silicon) is technically supported, however
it currently does not work due to a cmake problem in the pybind11 part of [SofaPython3](https://github.com/sofa-framework/SofaPython3).

1. Follow the [instructions](docs/source/setting_up_sofa.rst) to install SOFA and SofaPython3
2. Install the `sofa_env` package
```bash
conda activate sofa
pip install -e .
```
3. Test the installation
```
python3 sofa_env/scenes/controllable_object_example/controllable_env.py
```

## Environments
This repository currently contains 14 reinforcement learning environments.
- 12 robot-assited laparoscopy environments from [LapGym](https://www.jmlr.org/papers/v24/23-0207.html)
- a reinforcement learning environment around the scene from [A Simulation Framework for Magnetic Continuum Robots](https://github.com/ethz-msrl/mCR_simulator)
- the reinforcement learning environment from [Sim-to-Real Transfer for Visual Reinforcement Learning of Deformable Object Manipulation for Robot-Assisted Surgery](https://ieeexplore.ieee.org/abstract/document/9976185)

| Video                                                                            | Name                     |
|----------------------------------------------------------------------------------|:-------------------------|
| <img src="docs/source/images/environments/reach.gif" width="100"/>               | Reach                    |
| <img src="docs/source/images/environments/deflect_spheres.gif" width="100"/>     | Deflect Spheres          |
| <img src="docs/source/images/environments/search_for_point.gif" width="100"/>    | Search for Point         |
| <img src="docs/source/images/environments/tissue_manipulation.gif" width="100"/> | Tissue Manipulation      |
| <img src="docs/source/images/environments/pick_and_place.gif" width="100"/>      | Pick and Place           |
| <img src="docs/source/images/environments/grasp_lift_touch.gif" width="100"/>    | Grasp Lift and Touch     |
| <img src="docs/source/images/environments/rope_cutting.gif" width="100"/>        | Rope Cutting             |
| <img src="docs/source/images/environments/precision_cutting.gif" width="100"/>   | Precision Cutting        |
| <img src="docs/source/images/environments/tissue_dissection.gif" width="100"/>   | Tissue Dissection        |
| <img src="docs/source/images/environments/thread_in_hole.gif" width="100"/>      | Thread in Hole           |
| <img src="docs/source/images/environments/rope_threading.gif" width="100"/>      | Rope Threading           |
| <img src="docs/source/images/environments/ligating_loop.gif" width="100"/>       | Ligating Loop            |
| <img src="docs/source/images/environments/tissue_retraction.gif" width="100"/>   | Tissue Retraction        |
| <img src="docs/source/images/environments/mcr.gif" width="100"/>                 | Magnetic Continuum Robot |

## Adding new Environments
A ``SofaEnv`` base class and ``sofa_templates`` are provided to simplify implementing new reinforcement environments.
Please refer to the included documentation for a tutorial on how to build new scenes in SOFA and implement reinforcement environments with ``sofa_env``.
Also visit [sofa_godot](https://github.com/ScheiklP/sofa_godot) for SofaGodot, a [Godot](https://github.com/godotengine/godot) plugin to visually create new SOFA scenes.

## Documentation
You can either access the documentation [here](https://scheiklp.github.io/sofa_env/) or build it locally with

```bash
cd docs
make html
firefox build/html/index.html
```

## Citing
If you use the project in your work, please consider citing it with:
```bibtex
@article{JMLR:v24:23-0207,
  author  = {Paul Maria Scheikl and Balázs Gyenes and Rayan Younis and Christoph Haas and Gerhard Neumann and Martin Wagner and Franziska Mathis-Ullrich},
  title   = {LapGym - An Open Source Framework for Reinforcement Learning in Robot-Assisted Laparoscopic Surgery},
  journal = {Journal of Machine Learning Research},
  year    = {2023},
  volume  = {24},
  number  = {368},
  pages   = {1--42},
  url     = {http://jmlr.org/papers/v24/23-0207.html}
}
```

## Notes
The Freemotion Animation Loop of SOFA solves collisions and other constraints with Lagrange Multipliers.
This method requires the computation of a compliance matrix for each object.
This compliance matrix can be precomputed and is valid as long as the object does not change too much for example through cutting.
Precomputing, handling, and loading the compliance matrix is done by the `PrecomputedConstraintCorrection` object.
If your scene has one or more of them, the object will try to load the matrix from storage, and compute it, when it cannot find the appropriate file.
This can be seen by the terminal output of SOFA when creating the scene.
You may see a `[ERROR]   [FileRepository] File tissue-1479-0.1.comp NOT FOUND in: ...` output which is actually not an error but SOFA telling you it is now generating the precomputed compliance matrix.
This may take up to a few minutes.

## Getting Started with SOFA - Reading List
- To get a general understanding of how SOFA works, read all pages under *Simulation Principles* in the [official documentation](https://www.sofa-framework.org/community/doc/simulation-principles/scene-graph/)
- To get a deeper understanding of the underlying principles, like mappings, read the [SOFA Paper](https://hal.inria.fr/hal-00681539/document).
But be aware, that some terms like *MechanicalState* (now called *MechanicalObject*) in the paper have changed in the actual API.
Use it more as a "how does SOFA work?" not as a "how is SOFA implemented?"
- Look through the [SOFA Tutorials](https://github.com/sofa-framework/sofa/tree/master/examples/Tutorials).
Some of the Tutorials might not work.
That is generally not your fault.
You can run them via the command line. For example

```bash
$FOLDER_TARGET/install/bin/runSofa $FOLDER_SRC/examples/Tutorials/Basic/TutorialBasicCube.scn
```

## Acknowledgements
This work is supported by the Helmholtz Association under the joint research school "HIDSS4Health – Helmholtz Information and Data Science School for Health".
