cmake_minimum_required(VERSION 3.10...3.18)
project(fgMI LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Find Threads
find_package(Threads REQUIRED)

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Find Python (for building the Python module)
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# Find pybind11. If not found via package, add from our extern/pybind11 subdirectory.
if(NOT TARGET pybind11::module)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/carma)
endif()

# -------------------------
# Standalone Executable Target
# -------------------------
# Create the executable from mi_estimators.hpp (which contains main())
add_executable(fgMI mi_estimators_main.cpp)

# Link Armadillo and OpenMP (if available)
target_link_libraries(fgMI PRIVATE ${ARMADILLO_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(fgMI PRIVATE OpenMP::OpenMP_CXX)
endif()

# -------------------------
# Python Module Target
# -------------------------
# Create the Python module using pybind11 from bindings.cpp
pybind11_add_module(fgmi_module bindings.cpp)

# Ensure the module can find our headers
target_include_directories(fgmi_module PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link against Armadillo and OpenMP as needed
target_link_libraries(fgmi_module PRIVATE ${ARMADILLO_LIBRARIES})
target_link_libraries(fgmi_module PRIVATE carma::carma)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fgmi_module PRIVATE OpenMP::OpenMP_CXX)
endif()
