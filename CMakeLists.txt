cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)

include("${CMAKE_CURRENT_SOURCE_DIR}/external/madrona/cmake/madrona_init.cmake")

project(MadronaBenchmark LANGUAGES C CXX)

include(setup)
include(dependencies)

add_subdirectory(external)
add_subdirectory(common)
add_subdirectory(habitat)
add_subdirectory(hideseek)
add_subdirectory(mjx)
add_subdirectory(glb)
