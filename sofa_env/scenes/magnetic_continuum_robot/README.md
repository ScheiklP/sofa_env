# Magnetic Continuum Robot Scene
This scene is a port from [A Simulation Framework for Magnetic Continuum Robots](https://github.com/ethz-msrl/mCR_simulator) into `sofa_env`.
The scene components were slightly adapted to conform with SOFA version 22.06.

## Installation
- Tested with Python 3.9.12 on Ubuntu 22.10
- Compared to the installation described in the README.md of the `sofa_env` repository, three additonal plugins have to be installed during the build process.
- Important: When cloning the plugin repositories, checkout the correct branch/version afterwards. Do not use the master branch! Wrong plugin versions can lead to crashes during the build process.
    - BeamAdapter: v22.06
    - SoftRobots: v22.06
    - STLIB: v22.06
    - SOFA and SofaPython3: See commit IDs in code blocks below
- The `ln -s` commands at the end are needed when using the newest Ubuntu version (22.10), as the links were missing.


### Install build dependencies

```
sudo apt -y install build-essential software-properties-common python3-software-properties
sudo apt -y install libboost-all-dev
sudo apt -y install libpng-dev libjpeg-dev libtiff-dev libglew-dev zlib1g-dev
sudo apt -y install libeigen3-dev
sudo apt -y install libcanberra-gtk-module libcanberra-gtk3-module
sudo apt -y install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
```

### Preparing Python

1. Install Conda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3*.sh
conda init
```

2. Create a Conda environment and install `pybind11`

```
conda create -n sofa python=3.9
conda activate sofa
conda install -c conda-forge pybind11
```

### Environment variables for building SOFA including needed Plugins

1. Set source and build paths

```
FOLDER_SRC=$HOME/sofa/src
FOLDER_TARGET=$HOME/sofa/build
FOLDER_SP3=$FOLDER_SRC/applications/plugins/SofaPython3
FOLDER_BEAM=$FOLDER_SRC/applications/plugins/BeamAdapter
FOLDER_SOFT=$FOLDER_SRC/applications/plugins/SoftRobots
FOLDER_STLIB=$FOLDER_SRC/applications/plugins/STLIB
```

3. Set variables for building SofaPython3

```
PYTHON_PKG_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
PYTHON_EXE=$(which python3)
PYTHON_ROOT_DIR=$CONDA_PREFIX
```

### Clone the SOFA and plugin repositories

```
mkdir -p $FOLDER_SRC
mkdir -p $FOLDER_TARGET
mkdir -p $FOLDER_BEAM
mkdir -p $FOLDER_SOFT
mkdir -p $FOLDER_STLIB
git clone git@github.com:sofa-framework/sofa.git $FOLDER_SRC
cd $FOLDER_SRC
git checkout cfaeadccb418337c48d63e05adbf7d09e412d375
cd $FOLDER_SP3
git init
git remote add origin git@github.com:sofa-framework/SofaPython3.git
git pull origin master
git checkout b5e28c15305cb8cc21a7dc04d51db47730962f61
git clone git@github.com:sofa-framework/BeamAdapter.git $FOLDER_BEAM
cd $FOLDER_BEAM
git checkout v22.06
git clone git@github.com:SofaDefrost/SoftRobots.git $FOLDER_SOFT
cd $FOLDER_SOFT
git checkout v22.06
git clone git@github.com:SofaDefrost/STLIB.git $FOLDER_STLIB
cd $FOLDER_STLIB
git checkout v22.06
```

### CMake Build commands and plugin copying + env setting

1. Build sofa and plugins

```
cmake -Wno-dev \
-S $FOLDER_SRC -B $FOLDER_TARGET \
-DCMAKE_BUILD_TYPE=Release \
-DSOFA_FETCH_SOFAPYTHON3=OFF \
-DPLUGIN_SOFAPYTHON3=ON \
-DPython_EXECUTABLE=$PYTHON_EXE \
-DPython_ROOT_DIR=$PYTHON_ROOT_DIR \
-DSP3_LINK_TO_USER_SITE=ON \
-DSP3_PYTHON_PACKAGES_LINK_DIRECTORY=$PYTHON_PKG_PATH \
-DPLUGIN_SOFACARVING=ON \
-DPLUGIN_SOFAIMPLICITFIELD=ON \
-DPLUGIN_SOFADISTANCEGRID=ON \
-DPLUGIN_PLUGINEXAMPLE=ON \
-DSOFA_EXTERNAL_DIRECTORIES="$FOLDER_BEAM;$FOLDER_STLIB;$FOLDER_SOFT" \
-DSP3_BUILD_TEST=OFF \
-DSOFA_BUILD_TESTS=OFF

cmake --build $FOLDER_TARGET -j --target install
```

2. Move built plugin libraries into install directory

```
cp -r $FOLDER_TARGET/external_directories/BeamAdapter $FOLDER_TARGET/plugins
cp -r $FOLDER_TARGET/external_directories/SoftRobots $FOLDER_TARGET/plugins
cp -r $FOLDER_TARGET/external_directories/STLIB $FOLDER_TARGET/plugins
```

3. Set conda environment variables

```
conda env config vars set SOFA_ROOT=$FOLDER_TARGET
conda env config vars set SOFAPYTHON3_ROOT=$FOLDER_TARGET/lib
```

### Manually linking SofaPython3 to Python

```
ln -s $FOLDER_TARGET/lib/python3/site-packages/Sofa $PYTHON_ROOT_DIR/lib/python3.9/site-packages/Sofa
ln -s $FOLDER_TARGET/lib/python3/site-packages/SofaRuntime $PYTHON_ROOT_DIR/lib/python3.9/site-packages/SofaRuntime
ln -s $FOLDER_TARGET/lib/python3/site-packages/SofaTypes $PYTHON_ROOT_DIR/lib/python3.9/site-packages/SofaTypes
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 $PYTHON_ROOT_DIR/lib/libstdc++.so.6
ln -s $FOLDER_TARGET/lib/python3/site-packages/splib $PYTHON_ROOT_DIR/lib/python3.9/site-packages/splib
```
