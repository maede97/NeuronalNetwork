cmake_minimum_required(VERSION 3.1)

project(NeuronalNetwork)

set(CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
   message(SEND_ERROR "In-source builds are not allowed.")
endif()



# Example build by default ON
option(BUILD_EXAMPLES "Build examples" ON)
# Unit test build by default ON
option(ENABLE_TESTING "Build unit tests" ON)

if(BUILD_EXAMPLES)
  # Build examples
  add_subdirectory(examples)
endif()

if(ENABLE_TESTING) 
  # Enable testing
  enable_testing()
  add_subdirectory(test)
endif()

# set release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()