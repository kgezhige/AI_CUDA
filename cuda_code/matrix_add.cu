#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

// 错误检查宏
#define CHECK_CUDA_ERROR(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// 矩阵加法内核
__global__ void matrixAdd(float *A, float *B, float *C, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // 行索引
    int j = blockIdx.x * blockDim.x + threadIdx.x; // 列索引
    if (i < rows && j < cols) {
        int idx = i * cols + j; // 一维数组索引
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // 矩阵维度
    const int rows = 1024*10;
    const int cols = 1024*10;
    const int size = rows * cols;
    const int bytes = size * sizeof(float);

    // 主机内存
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(bytes);
    h_B = (float *)malloc(bytes);
    h_C = (float *)malloc(bytes);

    // 初始化输入矩阵
    for (int i = 0; i < size; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // 设备内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, bytes));

    // 记录开始时间
    clock_t start = clock();

    // 主机到设备数据传输
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // 设置线程块和网格尺寸
    dim3 threadsPerBlock(16*2, 16*2); // 每个块 16x16 线程
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 启动内核
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError()); // 检查内核启动错误
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // 等待内核完成

    // 设备到主机数据传输
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // 记录结束时间
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    // 验证结果
    bool correct = true;
    for (int i = 0; i < size; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            printf("Error at index %d: Expected %f, Got %f\n", i, expected, h_C[i]);
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("Matrix addition successful!\n");
    }

    // 输出执行时间
    printf("Execution time: %f seconds\n", time_spent);

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    return 0;
}