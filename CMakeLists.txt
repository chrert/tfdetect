cmake_minimum_required(VERSION 2.8)
project(tfdetect)

set(LIB_NAME ${PROJECT_NAME})
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install")

add_subdirectory(cmake_tensorflow_c)

# OpenCV for image handling - should come with PVT, how about include dirs??
find_package(OpenCV REQUIRED)
message("OpenCV version ${OpenCV_VERSION}:")
message(${OpenCV_INCLUDE_DIRS})

set(HEADERS
  inc/tfdetect.h
)
set(SOURCES
  src/tfdetect.cpp
)

set(LIBS ${LIB_NAME})
add_library(${LIB_NAME} ${HEADERS} ${SOURCES})
target_include_directories(${LIB_NAME} PUBLIC "inc")
target_include_directories(${LIB_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${LIB_NAME}
  ${OpenCV_LIBRARIES}
  tensorflow_c
)
# important to fix the dynamic linking of the libraries
set_target_properties(${LIB_NAME} PROPERTIES
                      INSTALL_RPATH '${CMAKE_INSTALL_PREFIX}/lib:$ORIGIN/')

install(FILES ${HEADERS}
  DESTINATION include/${LIB_NAME}
  COMPONENT headers
)

install(TARGETS ${LIBS}
  DESTINATION lib
  EXPORT ${LIB_NAME}
  COMPONENT libraries
)
