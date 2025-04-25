#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <iostream>

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
    // 声明 WMMA 碎片
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;

    // 初始化累加器碎片为 0
    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    // 加载矩阵 A 和 B 到碎片
    nvcuda::wmma::load_matrix_sync(a_frag, a, 16); // 步长为 16
    nvcuda::wmma::load_matrix_sync(b_frag, b, 16); // 步长为 16

    // 执行矩阵乘法：acc = A * B
    nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    // 加载矩阵 C 到碎片
    nvcuda::wmma::load_matrix_sync(c_frag, c, 16, nvcuda::wmma::mem_row_major);

    // 累加 C 到 acc：acc += C
    for (int i = 0; i < acc_frag.num_elements; i++) {
        acc_frag.x[i] += c_frag.x[i];
    }

    // 存储结果到 D
    nvcuda::wmma::store_matrix_sync(d, acc_frag, 16, nvcuda::wmma::mem_row_major);
}

// 主机函数：初始化矩阵并调用核函数
int main() {
    // 矩阵尺寸
    const int M = 16;
    const int N = 16;
    const int K = 16;

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
    dim3 threadsPerBlock(32, 1); // 一个 warp 有 32 个线程
    dim3 blocksPerGrid(1, 1);    // 单块处理 16x16 矩阵

    // 调用核函数
    wmma_matrix_mult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 复制结果回主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 打印结果
    printf("Result matrix D (first 4 elements):\n");
    for (int i = 0; i < 4; i++) {
        printf("%f ", h_D[i]);
    }
    printf("\n");

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