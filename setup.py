import importlib.util

from setuptools import setup

SOFA_MODULE = "Sofa"
assert importlib.util.find_spec(SOFA_MODULE), f"Could not find {SOFA_MODULE} module. \n Please install SOFA with the SofaPython3 plugin."

setup(
    name="sofa_env",
    version="0.0.1",
    description="OpenAI Gym wrapper for SOFA simulations",
    author="Paul Maria Scheikl",
    author_email="paul.scheikl@kit.edu",
    packages=["sofa_env"],
    install_requires=[
        "numpy",
        "gymnasium",
        "pyglet==1.5.21",
        "pygame",
        "PyOpenGL==3.1.7",
        "PyOpenGL-accelerate",
        "Sphinx",
        "sphinx-tabs",
        "sphinx_rtd_theme",
        "sphinx-autobuild",
        "autodocsumm",
        "opencv-python",
        "matplotlib",
        "tqdm",
        "requests",
        "numba",
        "inputs",
        "numpy-stl",
        "open3d",
        "pytest",
        "filelock",
    ],
    python_requires=">=3.9",
)
