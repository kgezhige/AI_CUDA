#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// 优化矩阵乘法内核（使用共享内存）
__global__ void matrixMulOptimized(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE + 1]; // 添加填充避免Bank冲突
    __shared__ float sB[TILE_SIZE][TILE_SIZE + 1];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // 分块计算
    for (int m = 0; m < (K + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // 加载A和B的子矩阵到共享内存
        if (row < M && m * TILE_SIZE + threadIdx.x < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + m * TILE_SIZE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < N && m * TILE_SIZE + threadIdx.y < K)
            sB[threadIdx.y][threadIdx.x] = B[(m * TILE_SIZE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

        // 计算子矩阵乘法
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    // 写入结果
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// 主机端代码
void matrixMulCUDAOptimized(float* h_A, float* h_B, float* h_C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t pitch_A, pitch_B, pitch_C;

    // 分配对齐的设备内存
    cudaMallocPitch(&d_A, &pitch_A, K * sizeof(float), M);
    cudaMallocPitch(&d_B, &pitch_B, N * sizeof(float), K);
    cudaMallocPitch(&d_C, &pitch_C, N * sizeof(float), M);

    // 复制数据到设备
    cudaMemcpy2D(d_A, pitch_A, h_A, K * sizeof(float), K * sizeof(float), M, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_B, pitch_B, h_B, N * sizeof(float), N * sizeof(float), K, cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // 启动内核
    matrixMulOptimized<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    // 复制结果回主机
    cudaMemcpy2D(h_C, N * sizeof(float), d_C, pitch_C, N * sizeof(float), M, cudaMemcpyDeviceToHost);

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

    // 调用优化CUDA矩阵乘法
    matrixMulCUDAOptimized(h_A, h_B, h_C, M, N, K);

    // 释放主机内存
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}