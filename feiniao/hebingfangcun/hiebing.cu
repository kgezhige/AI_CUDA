#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void add1(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}
__global__ void add2(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x +1;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}
int main() {
    const int N = 32 * 1024; //* 1024;
    float *input_x = (float*)malloc(N * sizeof(float));
    float *input_y = (float*)malloc(N * sizeof(float));
    float *output = (float*)malloc(N * sizeof(float));

    // 初始化输入数据
    for (int i = 0; i < N; i++) {
        input_x[i] = 1.0f;
        input_y[i] = 2.0f;
    }

    float *d_x, *d_y, *d_output;
    cudaError_t err;

    // 分配设备内存
    err = cudaMalloc((void**)&d_x, N * sizeof(float)); CHECK_CUDA_ERROR(err);
    err = cudaMalloc((void**)&d_y, N * sizeof(float)); CHECK_CUDA_ERROR(err);
    err = cudaMalloc((void**)&d_output, N * sizeof(float)); CHECK_CUDA_ERROR(err);

    // 拷贝数据到设备
    err = cudaMemcpy(d_x, input_x, N * sizeof(float), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR(err);
    err = cudaMemcpy(d_y, input_y, N * sizeof(float), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR(err);

    // 配置网格和线程块
    dim3 Block(256);
    dim3 Grid((N + Block.x - 1) / Block.x); // 确保覆盖所有元素

    // 启动 kernel
    add1<<<Grid, Block>>>(d_x, d_y, d_output, N);
    add2<<<Grid, Block>>>(d_x, d_y, d_output, N);

    err = cudaGetLastError(); CHECK_CUDA_ERROR(err);

    // 等待 GPU 完成
    err = cudaDeviceSynchronize(); CHECK_CUDA_ERROR(err);

    // 拷贝结果回主机
    err = cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost); CHECK_CUDA_ERROR(err);

    // 验证结果（可选）
    for (int i = 0; i < 10; i++) {
        printf("output[%d] = %f\n", i, output[i]);
    }

    // 释放内存
    free(input_x);
    free(input_y);
    free(output);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_output);

    return 0;
}