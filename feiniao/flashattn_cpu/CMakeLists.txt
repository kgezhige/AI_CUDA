cmake_minimum_required(VERSION 3.10)
project(AttentionCPU C)

set(CMAKE_C_STANDARD 11)

# Add executable
add_executable(attention_cpu flashattn_cpu.c)

# Link math library
target_link_libraries(attention_cpu m)