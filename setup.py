import os
import subprocess
import platform
import logging
import shutil

from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SOFA_DOWNLOAD_URLS = {
    "Linux": "https://github.com/sofa-framework/sofa/releases/download/v24.06.00/SOFA_v24.06.00_Linux.zip",
    "Darwin": "https://github.com/sofa-framework/sofa/releases/download/v24.06.00/SOFA_v24.06.00_MacOS.zip",
    "Windows": "https://github.com/sofa-framework/sofa/releases/download/v24.06.00/SOFA_v24.06.00_Win64.zip",
}

PYTHON_VERSION = "3.10"

INSTALL_COMMANDS = {
    "wget": {
        "Linux": "sudo apt install wget",
        "Darwin": "brew install wget",
        "Windows": "choco install wget",
    },
    "unzip": {
        "Linux": "sudo apt install unzip",
        "Darwin": "brew install unzip",
        "Windows": "choco install unzip",
    },
}


class SOFAInstallCommand(install):
    """Customized install command to optionally install SOFA."""

    def run(self):

        # Check if manual sofa install environment variable (SKIP_SOFA) is set
        skip_sofa = os.environ.get("SKIP_SOFA", "0")

        if skip_sofa == "1":
            logger.warning("Skipping SOFA installation. Make sure to install SOFA manually.")
        else:
            download_and_install_sofa()

        # Proceed with normal installation
        install.run(self)


class SOFADevelopCommand(develop):
    """Customized develop command to optionally install SOFA."""

    def run(self):

        # Check if manual sofa install environment variable (SKIP_SOFA) is set
        skip_sofa = os.environ.get("SKIP_SOFA", "0")

        if skip_sofa == "1":
            logger.warning("Skipping SOFA installation. Make sure to install SOFA manually.")
        else:
            download_and_install_sofa()

        # Proceed with normal installation
        develop.run(self)


def download_and_install_sofa():
    # Check if the required Python version is installed
    python_version = platform.python_version()
    main_python_version = ".".join(python_version.split(".")[:2])
    if main_python_version != PYTHON_VERSION:
        raise Exception(
            f"Python version {PYTHON_VERSION} is required. Found {python_version}."
        )

    # Check if the CPU architecture is x86_64
    cpu_arch = platform.machine()
    if cpu_arch != "x86_64":
        raise Exception(
            f"CPU architecture x86_64 is required. Found {cpu_arch}. Please install SOFA manually."
        )

    # Determine platform and get the corresponding download URL
    download_url = SOFA_DOWNLOAD_URLS[platform.system()]
    file_name = download_url.split("/")[-1]
    org_dir_name = file_name.split(".zip")[0]
    sofa_dir_name = "SOFA"

    which_command = "which" if platform.system() != "Windows" else "where"

    # Assert that wget, unzip, and rm are available
    if subprocess.run([which_command, "wget"]).returncode != 0:
        raise Exception(
            f"wget is not installed. Please install it with {INSTALL_COMMANDS['wget'][platform.system()]}"
        )

    if subprocess.run([which_command, "unzip"]).returncode != 0:
        raise Exception(
            f"unzip is not installed. Please install it with {INSTALL_COMMANDS['unzip'][platform.system()]}"
        )

    # Download SOFA and extract it
    logger.info(f"Downloading SOFA from {download_url}")
    subprocess.run(["wget", download_url])

    logger.info(f"Extracting SOFA")
    subprocess.run(["unzip", file_name])
    os.remove(file_name)

    # Move extracted directory to "SOFA", overwrite if necessary
    if os.path.exists(sofa_dir_name):
        shutil.rmtree(sofa_dir_name)
    os.rename(org_dir_name, sofa_dir_name)

    # Set environment variables
    sofa_root = os.path.abspath(sofa_dir_name)
    sofa_python_root = os.path.join(sofa_root, "plugins", "SofaPython3")
    sofa_python_libs = os.path.join(sofa_python_root, "lib", "python3", "site-packages")
    python_pkg_path = (
        subprocess.check_output(
            [
                "python3",
                "-c",
                'import sysconfig; print(sysconfig.get_paths()["purelib"])',
            ]
        )
        .strip()
        .decode("utf-8")
    )

    # Check if the SOFA Python libraries are already installed, and remove them if necessary,
    # then create symbolic links.
    for lib in ["Sofa", "SofaRuntime", "SofaTypes", "splib"]:
        if os.path.exists(os.path.join(python_pkg_path, lib)):
            os.remove(os.path.join(python_pkg_path, lib))
        os.symlink(
            os.path.join(sofa_python_libs, lib), os.path.join(python_pkg_path, lib)
        )

    logger.info(f"SOFA installed successfully at {sofa_root}")


setup(
    name="sofa_env",
    version="1.0.0",
    description="Gymnasium wrapper for SOFA simulations",
    author="Paul Maria Scheikl",
    author_email="pscheik1@jhu.edu",
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
        "scipy",
    ],
    python_requires=">=3.9",
    cmdclass={
        "install": SOFAInstallCommand,
        "develop": SOFADevelopCommand,
    },
)
