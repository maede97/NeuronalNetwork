cmake_minimum_required(VERSION 3.1)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_BINARY_DIR}
)

link_directories(
  ${PROJECT_BINARY_DIR}/NeuronalNetwork
)

set(TESTS
    Linker
    Layers
)

foreach(TEST_NAME ${TESTS})
  add_executable(${TEST_NAME} Test_${TEST_NAME}.cpp)
  add_test(${TEST_NAME} ${TEST_NAME})
endforeach()