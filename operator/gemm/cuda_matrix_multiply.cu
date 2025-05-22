#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// 定义线程块大小（选择 32x32 以匹配 warp 大小并优化共享内存使用）
#define TILE_WIDTH 32

// CUDA 错误检查宏
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// CUDA 核函数：使用共享内存的矩阵乘法
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int M, int N, int K) {
    // 分配共享内存
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    // 获取线程索引
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    // 循环处理分块
    for (int t = 0; t < (N - 1) / TILE_WIDTH + 1; ++t) {
        // 加载数据到共享内存
        if (row < M && t * TILE_WIDTH + threadIdx.x < N) {
            sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_WIDTH + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < K && t * TILE_WIDTH + threadIdx.y < N) {
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * K + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 同步线程以确保共享内存加载完成
        __syncthreads();

        // 计算当前分块的乘法
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        // 同步线程以确保共享内存可被下一轮重用
        __syncthreads();
    }

    // 写入结果到全局内存
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

// 主机函数：初始化矩阵
void initializeMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = (float)(rand() % 100) / 10.0f; // 随机值 0.0 到 9.9
    }
}

// 主机函数：验证结果
void verifyResult(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * K + j];
            }
            assert(fabs(C[i * K + j] - sum) < 1e-5);
        }
    }
}

int main() {
    // 矩阵维度
    int M = 512; // A 的行数
    int N = 512; // A 的列数，B 的行数
    int K = 512; // B 的列数

    // 分配主机内存
    float *h_A = (float *)malloc(M * N * sizeof(float));
    float *h_B = (float *)malloc(N * K * sizeof(float));
    float *h_C = (float *)malloc(M * K * sizeof(float));

    // 初始化矩阵
    srand(42);
    initializeMatrix(h_A, M, N);
    initializeMatrix(h_B, N, K);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, N * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * K * sizeof(float)));

    // 复制数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice));

    // 设置网格和线程块
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((K + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // 启动核函数
    matrixMultiplyShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 复制结果回主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost));

    // 验证结果
    verifyResult(h_A, h_B, h_C, M, N, K);
    printf("Matrix multiplication completed successfully!\n");

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    return 0;
}