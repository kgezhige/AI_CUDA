cmake_minimum_required(VERSION 3.10)
project(TestLibTorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(test test.cpp)
target_link_libraries(test "${TORCH_LIBRARIES}")
set_property(TARGET test PROPERTY CXX_STANDARD 17)