cmake_minimum_required(VERSION 2.8)
project(tfdetect)

set(LIB_NAME ${PROJECT_NAME})

IF(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
ENDIF(NOT DEFINED CMAKE_BUILD_TYPE)
set(CMAKE_VERBOSE_MAKEFILE false)

# Add compiler flags to use C++11
IF(NOT WIN32)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -std=c++11 -march=native")
ENDIF(NOT WIN32)

set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install")

# OpenCV for image handling - should come with PVT, how about include dirs??
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("OpenCV version ${OpenCV_VERSION}:")
message(${OpenCV_INCLUDE_DIRS})

set(HEADERS
  inc/tfdetect.h
)
set(SOURCES
  src/tfdetect.cpp
)
include_directories("inc")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/c_api/install/lib/cmake")
find_package(TensorflowCPUCAPI REQUIRED)

# dynamic library
set(LIBS ${LIB_NAME})
add_library(${LIB_NAME} SHARED ${HEADERS} ${SOURCES})
target_link_libraries(${LIB_NAME}
  ${OpenCV_LIBRARIES}
  TensorflowCPUCAPI::Shared
)

# static library
set(LIBS ${LIBS} ${LIB_NAME}_static)
add_library(${LIB_NAME}_static STATIC ${HEADERS} ${SOURCES})
target_link_libraries(${LIB_NAME}_static
  ${OpenCV_LIBRARIES}
  TensorflowCPUCAPI::Shared
)

install(FILES ${HEADERS}
  DESTINATION include/${LIB_NAME}
  COMPONENT headers
)

install(TARGETS ${LIBS}
  DESTINATION lib
  EXPORT ${LIB_NAME}
  COMPONENT libraries
)
