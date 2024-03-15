.. _installation:

Installation
=============

This was tested with Python3.7 to Python3.10 on Ubuntu 18.04 to 22.04, Fedora 36 and 37, WSL on Windows.

Install build dependencies
##########################

.. code-block:: bash

    sudo apt -y install build-essential software-properties-common python3-software-properties
    sudo apt -y install libboost-all-dev
    sudo apt -y install libpng-dev libjpeg-dev libtiff-dev libglew-dev zlib1g-dev
    sudo apt -y install libeigen3-dev
    sudo apt -y install libcanberra-gtk-module libcanberra-gtk3-module
    sudo apt -y install qt5-default
    sudo apt -y install libtinyxml2-dev

.. note::
   For Ubuntu 22.04 and later, replace ``sudo apt -y install qt5-default`` with ``sudo apt -y install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools libqt5charts5-dev``.

Preparing python
################

You can either use a Conda environment, a virtualenv, or use the system's python.
We recommend using Conda.

* Option A: **Conda**

   1. Install Conda

   .. code-block:: bash

      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
      bash Miniconda3*.sh
      conda init

   2. Create a Conda environment and install pybind11

   .. code-block:: bash

      conda create -n sofa python=3.10
      conda activate sofa
      conda install -c conda-forge pybind11

* Option B: **VirtualEnv**

   .. code-block:: bash

      python3 -m venv $HOME/sofa/venv
      source $HOME/sofa/venv/bin/activate
      pip3 install pybind11

* Option C: **System**

   .. code-block:: bash

      sudo apt install python3-pybind11

.. _env_variables:

Environment variables for building SOFA
#######################################

   1. Set source and build paths

   .. code-block:: bash

      FOLDER_SRC=$HOME/sofa/src
      FOLDER_TARGET=$HOME/sofa/build
      FOLDER_SP3=$FOLDER_SRC/applications/plugins/SofaPython3

   2. Set variables for building SofaPython3

      * Option A: **Conda**

      .. code-block:: bash

         PYTHON_PKG_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
         PYTHON_EXE=$(which python3)
         PYTHON_ROOT_DIR=$CONDA_PREFIX

      * Option B: **VirtualEnv**

      .. code-block:: bash

         PYTHON_PKG_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
         PYTHON_EXE=$(which python3)
         PYTHON_ROOT_DIR=$VIRTUAL_ENV


      * Option C: **System**

      .. code-block:: bash

         PYTHON_PKG_PATH=$(python3 -m site --user-site)
         PYTHON_EXE=$(which python3)
         PYTHON_ROOT_DIR=/usr/local

.. _cloning:

Clone the SOFA repository
#########################

.. code-block:: bash

   mkdir -p $FOLDER_SRC
   mkdir -p $FOLDER_TARGET
   git clone https://github.com/sofa-framework/sofa.git $FOLDER_SRC
   cd $FOLDER_SRC
   git checkout v23.12
   cd $FOLDER_SP3
   git init
   git remote add origin https://github.com/sofa-framework/SofaPython3.git
   git pull origin master
   git checkout f1ac0f03efd6f6e7c30df8b18259e16da523f0b2

.. _cmake:

Run ``cmake``
#############

.. code-block:: bash

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
   -DSP3_BUILD_TEST=OFF \
   -DSOFA_BUILD_TESTS=OFF

For debugging the SOFA build itself, the following two CMake arguments are also helpful:

.. code-block:: bash

   -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=1

.. _compile:

Compile SOFA
############

1. Build SOFA

.. code-block:: bash

   cmake --build $FOLDER_TARGET -j --target install

.. warning::
   Using the `-j` flag tells cmake to build multiple targets in parallel. If you run out of memory, the compilation will fail. You can either reduce the number of parallel processes by passing a number to cmake (e.g. `-j 2`) or by increasing the size of your swapfile.


2. Add SofaPython3 to the list of default plugins so that SOFA loads it by default.

.. code-block:: bash

   echo "" | tee -a $FOLDER_TARGET/install/lib/plugin_list.conf.default
   echo "$FOLDER_TARGET/install/plugins/SofaPython3/lib/libSofaPython3.so 1.0" | tee -a $FOLDER_TARGET/install/lib/plugin_list.conf.default


.. warning::

   Empty the build folder after every change in code / commit. You never know...

   .. code-block:: bash

     rm -rf $FOLDER_TARGET
     mkdir -p $FOLDER_TARGET

   After that you can run cmake to configure and build.


Setting environment variables
#############################

Setting environment variables for python to let SOFA and SofaPython3 know where to find the relevant stuff

.. warning::

   On the last checked commit, SofaPython3 might have troubles finding the correct ``libpython3.9.so.1.0``. If that happens, add the directory
   that holds the ``libpython3.9.so.1.0`` that you used to compile to the ``LD_LIBRARY_PATH`` environment variable.
   E.g.:

   .. code-block:: bash

     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PYTHON_ROOT_DIR/lib

   We do not add this environment variable to conda, because conda will hard code the variable to whatever you set it to, not evaluating ``LD_LIBRARY_PATH=$LD_LIBRARY_PATH:...`` again. Furthermore this is a bug, introduced in recent commits and should hopefully be resolved soon.

* Option A: **Conda**

.. code-block:: bash

   conda env config vars set SOFA_ROOT=$FOLDER_TARGET/install
   conda env config vars set SOFAPYTHON3_ROOT=$FOLDER_TARGET/install/plugins/SofaPython3

.. note::
   For Ubuntu 22.04: If SOFA is missing ``GLIBCXX_3.4.30``, install it with ``conda install -c conda-forge gcc=12.1.0``

* Option B: **VirtualEnv**

You can also do this stuff with the virtualenvwrapper https://virtualenvwrapper.readthedocs.io/en/latest/

* Option C: **System**

Export the environment variables through your ``~/.bashrc`` or ``~/.zshrc``

.. code-block:: bash

   echo export SOFA_ROOT=$FOLDER_TARGET/install >> ~/.bashrc
   echo export SOFAPYTHON3_ROOT=$FOLDER_TARGET/install/plugins/SofaPython3 >> ~/.bashrc


Adding additional SOFA Plugins
##############################

If you want to compile SOFA with additional plugins, such as BeamAdapter or Cosserat, you will have to do a few additional steps

   1. In step :ref:`Environment variables for building SOFA <env_variables>` export additional environment variables for each plugin

   .. code-block:: bash

      FOLDER_BEAM=$FOLDER_SRC/applications/plugins/BeamAdapter
      FOLDER_COSSERAT=$FOLDER_SRC/applications/plugins/Cosserat

   2. In step :ref:`Clone the SOFA repository <cloning>` clone the additional plugins

   .. code-block:: bash

      git clone git@github.com:sofa-framework/BeamAdapter.git $FOLDER_BEAM
      git clone git@github.com:SofaDefrost/plugin.Cosserat.git $FOLDER_COSSERAT

   3. in step :ref:`Run cmake <cmake>` add this flag to the cmake command

   .. code-block:: bash

      -DSOFA_EXTERNAL_DIRECTORIES="$FOLDER_BEAM;$FOLDER_COSSERAT" \

   4. After :ref:`compiling <compile>`, move the built libraries into the install directory

   .. code-block:: bash

      cp -r $FOLDER_TARGET/external_directories/BeamAdapter $FOLDER_TARGET/install/plugins
      cp -r $FOLDER_TARGET/external_directories/Cosserat $FOLDER_TARGET/install/plugins


Manually Linking SofaPython3 to Python
######################################

If for some reason installing SofaPython3 does not work (cannot import Sofa in Python), you will probably just need to correctly link the modules compiled in SofaPython3 to your environment.
To import a module, python will look for it in the site-packages dir. For Conda, that is most likely in ``$HOME/miniconda3/envs/<env_name>/lib/python3.9/site-packages``.
First, locate where the SofaPython3 modules were compiled to (e.g. ``$HOME/sofa/build/install/plugins/SofaPython3/lib/python3/site-packages``) and then create soft links from all the modules into site-packages.

For example:

   .. code-block:: bash

      ln -s $HOME/sofa/build/install/plugins/SofaPython3/lib/python3/site-packages/Sofa $HOME/miniconda3/envs/<env_name>/lib/python3.9/site-packages/Sofa
      ln -s $HOME/sofa/build/install/plugins/SofaPython3/lib/python3/site-packages/SofaRuntime $HOME/miniconda3/envs/<env_name>/lib/python3.9/site-packages/SofaRuntime
      ln -s $HOME/sofa/build/install/plugins/SofaPython3/lib/python3/site-packages/SofaTypes $HOME/miniconda3/envs/<env_name>/lib/python3.9/site-packages/SofaTypes
      ln -s $HOME/sofa/build/install/plugins/SofaPython3/lib/python3/site-packages/splib $HOME/miniconda3/envs/<env_name>/lib/python3.9/site-packages/splib
