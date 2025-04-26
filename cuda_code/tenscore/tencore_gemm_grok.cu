#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <iostream>
#include <iomanip> // 用于控制输出格式

// 检查 CUDA 错误的宏
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* func, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

// CUDA 核函数：使用 Tensor Core 进行矩阵乘法 D = A * B + C
__global__ void wmma_matrix_mult(half* a, half* b, float* c, float* d, int M, int N, int K) {
    // WMMA 碎片
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;

    // 计算当前线程块处理的子矩阵位置
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // 每个 warp 处理一个 16x16 子矩阵
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // 每个 warp 负责 M 和 N 方向上的一个 16x16 子矩阵
    int row_offset = warpM * 16;
    int col_offset = warpN * 16;

    // 初始化累加器
    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    // 沿 K 维度分块
    for (int k = 0; k < K; k += 16) {
        // 检查边界，确保不越界
        if (row_offset < M && col_offset < N && k < K) {
            // 加载 A 的子矩阵 (row_offset, k)
            nvcuda::wmma::load_matrix_sync(a_frag, a + row_offset * K + k, K);
            // 加载 B 的子矩阵 (k, col_offset)
            nvcuda::wmma::load_matrix_sync(b_frag, b + k * N + col_offset, N);
            // 执行矩阵乘法
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // 加载 C 的子矩阵并累加
    if (row_offset < M && col_offset < N) {
        nvcuda::wmma::load_matrix_sync(c_frag, c + row_offset * N + col_offset, N, nvcuda::wmma::mem_row_major);
        for (int i = 0; i < acc_frag.num_elements; i++) {
            acc_frag.x[i] += c_frag.x[i];
        }
        // 存储结果到 D
        nvcuda::wmma::store_matrix_sync(d + row_offset * N + col_offset, acc_frag, N, nvcuda::wmma::mem_row_major);
    }
}

// 主机函数
int main() {
    // 矩阵尺寸
    const int M = 1024;
    const int N = 128;
    const int K = 512;

    // 主机端矩阵
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];
    float* h_D = new float[M * N];

    // 初始化矩阵 A、B、C
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;
    for (int i = 0; i < M * N; i++) h_C[i] = 0.5f;

    // 转换为 FP16
    half* h_A_fp16 = new half[M * K];
    half* h_B_fp16 = new half[K * N];
    for (int i = 0; i < M * K; i++) h_A_fp16[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; i++) h_B_fp16[i] = __float2half(h_B[i]);

    // 设备端指针
    half *d_A, *d_B;
    float *d_C, *d_D;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_D, M * N * sizeof(float)));

    // 复制数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A_fp16, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B_fp16, K * N * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // 设置线程块和网格
    dim3 threadsPerBlock(32, 4); // 每个线程块有 32*4=128 个线程
    dim3 blocksPerGrid((M + 15) / 16, 2); // 每个 warp 处理 16x16 子矩阵

    // 调用核函数
    wmma_matrix_mult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 复制结果回主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 打印结果（前 4 个元素）
    printf("Result matrix D (first 4 elements):\n");
    // for (int i = 0; i < M*N; i++) {
    //     printf("%f ", h_D[]);
    // }
    // printf("\n");

    // 复制结果回主机后，打印h_D
    std::cout << "Result Matrix h_D:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // 设置输出格式：固定小数点，保留2位小数
            std::cout << std::fixed << std::setprecision(2) << h_D[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    

    // 释放内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_D;
    delete[] h_A_fp16;
    delete[] h_B_fp16;
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaFree(d_D));

    return 0;
}