cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
project(
  "tensorflow_cpu_capi"
  VERSION 0.2
)

set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install")

# -------------
# CMake Options
# -------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
include(CMakePackageConfigHelpers)
set(CMAKECFG_INSTALL_DIR lib/cmake/TensorflowCPUCAPI)

# --------------------------------------------------------
# Include External Projects for Shared / Static Tensorflow
# --------------------------------------------------------

include(TensorflowBase)
include(TensorflowShared)

# ----------------------------------
# Define Shared Tensorflow Interface
# ----------------------------------

add_library(tensorflow_cpu_capi INTERFACE)
target_compile_options(
  tensorflow_cpu_capi INTERFACE
  "-std=c++11"
)
add_dependencies(
  tensorflow_cpu_capi
  tensorflow_shared
)
target_include_directories(
  tensorflow_cpu_capi INTERFACE
  "${CMAKE_INSTALL_PREFIX}/include/tensorflow/"
)
target_link_libraries(
  tensorflow_cpu_capi INTERFACE
  "${CMAKE_INSTALL_PREFIX}/lib/tensorflow_cpu_capi/libtensorflow.so"
  # static protobuf is used from the contrib/makefile
  dl pthread
)

# ----------------------------------------
# Configure CMake Config and Version Files
# ----------------------------------------

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/TensorflowCPUCAPIConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TensorflowCPUCAPIConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/TensorflowCPUCAPIConfig.cmake"
  INSTALL_DESTINATION "${CMAKECFG_INSTALL_DIR}"
  NO_SET_AND_CHECK_MACRO # TensorflowCC only uses interface libraries
  NO_CHECK_REQUIRED_COMPONENTS_MACRO # TensorflowCC does not have components
)

# -------
# Install
# -------

# install all header files
install(
  DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tensorflow/"
  DESTINATION include/tensorflow
  FILES_MATCHING PATTERN "c_api.h"
)
# shared library specific
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/tensorflow/bazel-bin/tensorflow/libtensorflow.so"
  DESTINATION lib/tensorflow_cpu_capi
)
# --------------------------
# Install CMake targets file
# --------------------------

set_target_properties(
  tensorflow_cpu_capi PROPERTIES EXPORT_NAME Shared
)
install(
  TARGETS tensorflow_cpu_capi
  EXPORT TensorflowCPUCAPITargets
)
install(
  EXPORT TensorflowCPUCAPITargets
  FILE TensorflowCPUCAPITargets.cmake
  NAMESPACE TensorflowCPUCAPI::
  DESTINATION "${CMAKECFG_INSTALL_DIR}"
)

# install config and version files
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TensorflowCPUCAPIConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TensorflowCPUCAPIConfigVersion.cmake"
  DESTINATION "${CMAKECFG_INSTALL_DIR}"
)
