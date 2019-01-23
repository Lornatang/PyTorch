![PyTorch Logo](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)

--------------------------------------------------------------------------------

PyTorch is a Python package that provides two high-level features:
- Tensor computation (like NumPy) with strong GPU acceleration
- Deep neural networks built on a tape-based autograd system

You can reuse your favorite Python packages such as NumPy, SciPy and Cython to extend PyTorch when needed.

You can write new neural network layers in Python using the torch API
[or your favorite NumPy-based libraries such as SciPy](https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html).

If you want to write your layers in C/C++, we provide a convenient extension API that is efficient and with minimal boilerplate.
There is no wrapper code that needs to be written. You can see [a tutorial here](https://pytorch.org/tutorials/advanced/cpp_extension.html) and [an example here](https://github.com/pytorch/extension-cpp).


## Installation

### Binaries
Commands to install from binaries via Conda or pip wheels are on our website:
[https://pytorch.org](https://pytorch.org)

### From Source

If you are installing from source, we highly recommend installing an [Anaconda](https://www.continuum.io/downloads) environment.
You will get a high-quality BLAS library (MKL) and you get a controlled compiler version regardless of your Linux distro.

Once you have [Anaconda](https://www.continuum.io/downloads) installed, here are the instructions.

If you want to compile with CUDA support, install
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 7.5 or above
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v6.x or above

If you want to disable CUDA support, export environment variable `NO_CUDA=1`.
Other potentially useful environment variables may be found in `setup.py`.

If you want to build on Windows, Visual Studio 2017 14.11 toolset and NVTX are also needed.
Especially, for CUDA 8 build on Windows, there will be an additional requirement for VS 2015 Update 3 and a patch for it.
The details of the patch can be found out [here](https://support.microsoft.com/en-gb/help/4020481/fix-link-exe-crashes-with-a-fatal-lnk1000-error-when-you-use-wholearch).

#### Install Dependencies

Common
```
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
```

On Linux
```bash
# Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda92 # or [magma-cuda80 | magma-cuda91] depending on your cuda version
```

#### Get the PyTorch Source
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
```

#### Install PyTorch
On Linux
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

On macOS
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
```

On Windows
```cmd
set "VS150COMNTOOLS=C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build"
set CMAKE_GENERATOR=Visual Studio 15 2017 Win64
set DISTUTILS_USE_SDK=1
REM The following two lines are needed for Python 2.7, but the support for it is very experimental.
set MSSdk=1
set FORCE_PY27_BUILD=1
REM As for CUDA 8, VS2015 Update 3 is also required to build PyTorch. Use the following line.
set "CUDAHOSTCXX=%VS140COMNTOOLS%\..\..\VC\bin\amd64\cl.exe"

call "%VS150COMNTOOLS%\vcvarsall.bat" x64 -vcvars_ver=14.11
python setup.py install
```

### Docker Image

Dockerfile is supplied to build images with cuda support and cudnn v7. You can pass `-e PYTHON_VERSION=x.y` flag to specify which Python version is to be used by Miniconda, or leave it unset to use the default. Build as usual
```
docker build -t pytorch -f docker/pytorch/Dockerfile .
```

You can also pull a pre-built docker image from Docker Hub and run with nvidia-docker,
but this is not currently maintained and will pull PyTorch 0.2.
```
nvidia-docker run --rm -ti --ipc=host pytorch/pytorch:latest
```
Please note that PyTorch uses shared memory to share data between processes, so if torch multiprocessing is used (e.g.
for multithreaded data loaders) the default shared memory segment size that container runs with is not enough, and you
should increase shared memory size either with `--ipc=host` or `--shm-size` command line options to `nvidia-docker run`.

### Building the Documentation

To build documentation in various formats, you will need [Sphinx](http://www.sphinx-doc.org) and the
readthedocs theme.

```
cd docs/
pip install -r requirements.txt
```
You can then build the documentation by running ``make <format>`` from the
``docs/`` folder. Run ``make`` to get a list of all available output formats.

### Previous Versions

Installation instructions and binaries for previous PyTorch versions may be found
on [website](https://pytorch.org/previous-versions).

### Visit the [official website](https://pytorch.org/) for more information.

### You can also visit [Google](https://www.google.com/) for online answers.