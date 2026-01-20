cmake_minimum_required(VERSION 3.10...3.18)
project(fastGeneMI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)

# -------------------------
# Dependencies
# -------------------------

find_package(Threads REQUIRED)

find_package(Armadillo REQUIRED)
message(STATUS "Armadillo include dirs: ${ARMADILLO_INCLUDE_DIRS}")
message(STATUS "Armadillo libraries: ${ARMADILLO_LIBRARIES}")

# If building inside conda, force CMake to use that interpreter unless user overrides
if(NOT DEFINED Python_EXECUTABLE AND DEFINED ENV{CONDA_PREFIX})
    set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python" CACHE FILEPATH "" FORCE)
endif()

# Force pybind11 (added via add_subdirectory) to use CMake's FindPython results
set(PYBIND11_FINDPYTHON ON)

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

if(NOT TARGET pybind11::module)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11)
endif()

if(NOT TARGET carma::carma)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/carma)
endif()


# OpenMP (optional)
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "Found OpenMP")
    endif()
endif()

# -------------------------
# Core library
# -------------------------
add_library(fastGeneMI_lib STATIC
    fastGeneMI/mi_estimators_main.cpp
)

target_include_directories(fastGeneMI_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/fastGeneMI>
    $<INSTALL_INTERFACE:include/fastGeneMI>
    ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(fastGeneMI_lib PUBLIC
    ${ARMADILLO_LIBRARIES}
    Threads::Threads
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(fastGeneMI_lib PRIVATE OpenMP::OpenMP_CXX)
endif()

# -------------------------
# Standalone executable
# -------------------------
add_executable(fgMI fastGeneMI/mi_estimators_main.cpp)
target_link_libraries(fgMI PRIVATE fastGeneMI_lib)

if(OpenMP_CXX_FOUND)
    target_link_libraries(fgMI PRIVATE OpenMP::OpenMP_CXX)
endif()

# -------------------------
# Python extension module
# -------------------------
pybind11_add_module(fastGeneMI fastGeneMI/bindings.cpp)

target_include_directories(fastGeneMI PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fastGeneMI
    ${ARMADILLO_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
)

target_link_libraries(fastGeneMI PRIVATE
    fastGeneMI_lib
    carma::carma
    ${ARMADILLO_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(fastGeneMI PRIVATE OpenMP::OpenMP_CXX)
endif()

# -------------------------
# Optimization flags
# -------------------------
if(DEFINED ENV{GITHUB_ACTIONS} AND "$ENV{GITHUB_ACTIONS}" STREQUAL "true")
    message(STATUS "Building on GitHub Actions: Using generic flags")
    target_compile_options(fastGeneMI_lib PRIVATE -O3 -ffast-math)
else()
    message(STATUS "Building locally: Using -march=native")
    target_compile_options(fastGeneMI_lib PRIVATE -O3 -march=native -ffast-math)
endif()

# -------------------------
# Install
# -------------------------
install(TARGETS fastGeneMI
    LIBRARY DESTINATION fastGeneMI
)

install(TARGETS fgMI
    RUNTIME DESTINATION bin
)

install(TARGETS fastGeneMI_lib
    ARCHIVE DESTINATION lib
)
