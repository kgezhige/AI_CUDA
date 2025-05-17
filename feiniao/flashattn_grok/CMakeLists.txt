cmake_minimum_required(VERSION 3.10)
project(flashattn_grok LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(SOURCES
flashattn_grok.cu
    # 添加其他 .cu 或 .cpp 文件
)
add_executable(flashattn_grok ${SOURCES}) 

# 设置目标属性
set_target_properties(flashattn_grok PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON  # 启用可分离编译
    CUDA_ARCHITECTURES 75         # 指定GPU架构，例如Ampere（A100）为80
)

# 链接CUDA运行时库
target_link_libraries(flashattn_grok PRIVATE CUDA::cudart)

# # 包含目录（如果有头文件）
# target_include_directories(flash_attention PRIVATE
#     ${CMAKE_SOURCE_DIR}/include
#     ${CUDAToolkit_INCLUDE_DIRS}
# )

# 调试标志（为Nsight分析生成调试信息）
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(flashattn_grok PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G>  # 生成CUDA调试信息
    )
endif()

# 优化标志（Release模式）
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(flashattn_grok PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>  # 优化
    )
endif()

# sudo /opt/nvidia/nsight-compute/2023.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export "/home/xukaili/Document/NVIDIA Nsight Compute/block32_tread_64" --force-overwrite --set full /home/xukaili/Document/work/feiniao/flashattn_grok/build/flashattn_grok