cmake_minimum_required(VERSION 3.1)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_BINARY_DIR}
)

link_directories(
  ${PROJECT_BINARY_DIR}/graphlib
)

set(EXAMPLES
    Characters
    Example
)

foreach(EXAMPLE_NAME ${EXAMPLES})
  add_executable(${EXAMPLE_NAME} ${EXAMPLE_NAME}.cpp)
endforeach()
