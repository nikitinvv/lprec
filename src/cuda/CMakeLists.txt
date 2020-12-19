enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# These policies are related to SWIG python libraries naming
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# SWIG requires the Python header
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})
# TODO: Is NumPy required?
#find_package(NumPy REQUIRED)
#include_directories(${NumPy_INCLUDE_DIRS})

find_package(OpenMP)

# Must tell CMake that SWIG interface files are CXX files
set_source_files_properties(lpRgpu.i PROPERTIES CPLUSPLUS ON)
swig_add_library(lpRgpu LANGUAGE python SOURCES lpRgpu.i lpRgpu.cu gridStorage.cu params.cu callerr.cu)
swig_link_libraries(lpRgpu cudart cufft OpenMP::OpenMP_CXX)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fopenmp")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/lpRgpu.py" DESTINATION lprec)
install(TARGETS lpRgpu LIBRARY DESTINATION lprec)
