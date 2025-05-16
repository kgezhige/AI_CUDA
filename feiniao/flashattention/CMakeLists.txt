cmake_minimum_required(VERSION 3.10)
project(flashattn)
set(CMAKE_PREFIX_PATH "/home/xukaili/Document/work/libtorch/libtorch/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(flashattn flashattn.cpp flash.cu) 
target_link_libraries(flashattn "${TORCH_LIBRARIES}")
set_property(TARGET flashattn PROPERTY CXX_STANDARD 17)