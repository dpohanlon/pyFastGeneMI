cmake_minimum_required(VERSION 3.10...3.18)
project(fastGeneMI LANGUAGES CXX)

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
find_package(Python 3.9 COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# Find pybind11. If not found via package, add from our extern/pybind11 subdirectory.
if(NOT TARGET pybind11::module)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/carma)
endif()

# Create a library for the core functionality
add_library(fastGeneMI_lib STATIC fastGeneMI/mi_estimators_main.cpp)

# Specify include directories for the library
target_include_directories(fastGeneMI_lib PUBLIC fastGeneMI)

# -------------------------
# Standalone Executable Target
# -------------------------
# Create the executable from mi_estimators.hpp (which contains main())
add_executable(fgMI fastGeneMI/mi_estimators_main.cpp)

# Link Armadillo and OpenMP (if available)
target_link_libraries(fgMI PRIVATE ${ARMADILLO_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(fgMI PRIVATE OpenMP::OpenMP_CXX)
endif()

# -------------------------
# Python Module Target
# -------------------------
# Create the Python module using pybind11 from fastGeneMI/bindings.cpp
pybind11_add_module(fastGeneMI fastGeneMI/bindings.cpp)

target_link_libraries(fastGeneMI PRIVATE fastGeneMI_lib)

# Ensure the module can find our headers
target_include_directories(fastGeneMI PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link against Armadillo and OpenMP as needed
target_link_libraries(fastGeneMI PRIVATE ${ARMADILLO_LIBRARIES})
target_link_libraries(fastGeneMI PRIVATE carma::carma)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fastGeneMI PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(fastGeneMI_lib PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS fastGeneMI
        LIBRARY DESTINATION fastGeneMI)

# Install the executable
install(TARGETS fgMI
    RUNTIME DESTINATION bin
)

# Install the Python module
install(TARGETS fastGeneMI
    LIBRARY DESTINATION lib/python_modules
)
