# parallelproj Backend

parallelproj (backend) is a high-performance library for 3D forward and backward projection, 
supporting both CUDA and non-CUDA builds and a minimal python interface.

## Table of Contents
- [Requirements](#requirements)
- [Building the Project](#building-the-project)
  - [Non-CUDA Build](#non-cuda-build)
  - [CUDA Build](#cuda-build)
- [Running Tests](#running-tests)
- [Python Interface](#python-interface)

---

## Requirements

### General Requirements
- **CMake** (version 3.18 or higher)
- **C++17** compatible compiler
- **OpenMP** (for non-CUDA builds)

### CUDA-Specific Requirements
- **CUDA Toolkit** (if building with CUDA support)

---

## Building the Project

### Non-CUDA Build

To build the project without CUDA support:

1. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```

2. Configure the project with CMake:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF
   ```

3. Build the project:
   ```bash
   cmake --build .
   ```

### CUDA Build

To build the project with CUDA support:

1. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```

2. Configure the project with CMake:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
   ```

3. Build the project:
   ```bash
   cmake --build .
   ```

---

## Running Tests

To run the tests after building the project:

1. Navigate to the `build` directory:
   ```bash
   cd build
   ```

2. Run the tests using `ctest`:
   ```bash
   ctest --output-on-failure
   ```

---

## Python Interface

If you want to build the Python interface, ensure that `pybind11` and `DLPack` are installed. 
Then, configure the project with the `BUILD_PYTHON` option:

1. Configure the project:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON
   ```

2. Build the Python interface:
   ```bash
   cmake --build .
   ```

---

## Notes

- For CUDA builds, ensure that the CUDA Toolkit is installed and properly configured.
- For non-CUDA builds, OpenMP is required for parallelization.