#include <stdio.h>
#include <cuda_runtime.h>

// 基本矩阵乘法内核
__global__ void matrixMulBasic(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < M && col < N) {
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 主机端代码
void matrixMulCUDA(float* h_A, float* h_B, float* h_C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // 分配设备内存
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 复制数据到设备
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // 启动内核
    matrixMulBasic<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    // 复制结果回主机
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int M = 1024, N = 1024, K = 1024;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // 分配主机内存
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    // 初始化输入矩阵
    for (int i = 0; i < M * K; ++i) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = rand() / (float)RAND_MAX;

    // 调用CUDA矩阵乘法
    matrixMulCUDA(h_A, h_B, h_C, M, N, K);

    // 释放主机内存
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}