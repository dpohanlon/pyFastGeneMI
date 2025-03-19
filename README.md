# pyFastGeneMI

A Python interface to the maximum likelihood mutual information calculation from [fastGeneMI](https://github.com/jonathanishhorowicz/fastGeneMI).

Building
========

When building from source, this package requires CMake, OpenMP, Armadillo, PyBind11, and CARMA. PyBind11 and CARMA are included with the package, however the rest are best installed using the package manager for your system. On Mac these can be installed using `brew`:

```bash
brew install cmake
brew install armadillo
brew install libomp
```

and OpenMP requires the path to be set:

``` bash
export OpenMP_ROOT=$(brew --prefix)/opt/libomp
```

On Ubuntu, these can be installed using `apt-get`:

```bash
sudo apt-get install cmake
sudo apt-get install python3-dev
sudo apt-get install libarmadillo-dev

```

To checkout the repository, as well as the PyBind11 and CARMA submodule:

``` bash

git clone --recurse-submodules git@github.com/dpohanlon/pyFastGeneMI.git
cd pyFastGeneMI
```

C++
---

To build the C++ library, you can build with CMake:

```bash
mkdir build
cd build
cmake ../
make -j4
```

Python
-----

With the repository checked out, build using pip:

```bash
pip install .
```
