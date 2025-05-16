#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

// 检查CUDA错误
#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// 简化版FlashAttention的CUDA核函数
__global__ void flashAttentionKernel(
    float* Q, float* K, float* V, float* O, // 输入Q、K、V和输出O
    int N, int d, // 序列长度N，头维度d
    int block_size // 分块大小
) {
    // 计算当前线程块处理的Q矩阵行索引
    int row = blockIdx.x * block_size;
    if (row >= N) return; // 超出序列长度，退出

    // 共享内存，用于存储Q、K、V的小块
    extern __shared__ float shared_mem[];
    float* s_Q = shared_mem; // 存储Q块
    float* s_K = s_Q + block_size * d; // 存储K块
    float* s_V = s_K + block_size * d; // 存储V块
    float* s_S = s_V + block_size * d; // 存储注意力分数S

    // 局部变量，用于在线Softmax
    float m = -1e10; // 最大值初始化为负无穷
    float l = 0.0; // 分母和初始化为0
    float o[d]; // 当前行的输出
    for (int i = 0; i < d; i++) o[i] = 0.0; // 初始化输出为0

    // 加载当前行的Q块到共享内存
    // block_size 指分块大小 64,
    
    for (int i = threadIdx.x; i < block_size * d; i += blockDim.x) {
        int r = i / d;
        int c = i % d;
        if (row + r < N) {
            s_Q[i] = Q[(row + r) * d + c];
        } else {
            s_Q[i] = 0.0; // 填充0
        }
    }
    __syncthreads();

    // 按块遍历K和V
    for (int col = 0; col < N; col += block_size) {
        // 加载K块到共享内存
        for (int i = threadIdx.x; i < block_size * d; i += blockDim.x) {
            int r = i / d;
            int c = i % d;
            if (col + r < N) {
                s_K[i] = K[(col + r) * d + c];
                s_V[i] = V[(col + r) * d + c];
            } else {
                s_K[i] = 0.0;
                s_V[i] = 0.0;
            }
        }
        __syncthreads();

        // 计算注意力分数S = Q @ K^T
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            if (row + i >= N) continue;
            for (int j = 0; j < block_size; j++) {
                float score = 0.0;
                for (int k = 0; k < d; k++) {
                    score += s_Q[i * d + k] * s_K[j * d + k];
                }
                score /= sqrtf((float)d); // 缩放点积
                s_S[i * block_size + j] = score;
            }
        }
        __syncthreads();

        // 在线Softmax
        float new_m = m;
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            for (int j = 0; j < block_size; j++) {
                new_m = fmaxf(new_m, s_S[i * block_size + j]);
            }
        }
        __syncthreads();

        // 更新m和l
        float old_l = l;
        l = 0.0;
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            for (int j = 0; j < block_size; j++) {
                s_S[i * block_size + j] = expf(s_S[i * block_size + j] - new_m);
                l += s_S[i * block_size + j];
            }
        }
        __syncthreads();

        // 更新输出O
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            if (row + i >= N) continue;
            for (int k = 0; k < d; k++) {
                float sum = 0.0;
                for (int j = 0; j < block_size; j++) {
                    sum += s_S[i * block_size + j] * s_V[j * d + k];
                }
                o[k] = o[k] * expf(m - new_m) * old_l + sum;
            }
        }
        m = new_m;
        __syncthreads();
    }

    // 写回全局内存
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
        if (row + i >= N) continue;
        for (int k = 0; k < d; k++) {
            O[(row + i) * d + k] = o[k] / l;
        }
    }
}

// 主函数：调用FlashAttention
void flashAttention(float* Q, float* K, float* V, float* O, int N, int d) {
    // 设置分块大小（可调）
    int block_size = 64;
    // 计算网格大小
    int grid_size = (N + block_size - 1) / block_size;
    // 计算共享内存大小
    size_t shared_mem_size = (3 * block_size * d + block_size * block_size) * sizeof(float);

    // 启动CUDA核函数
    flashAttentionKernel<<<grid_size, 256, shared_mem_size>>>(
        Q, K, V, O, N, d, block_size
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// 示例：主程序
int main() {
    int N = 512; // 序列长度
    int d = 64;  // 头维度
    size_t size = N * d * sizeof(float);

    // 分配主机内存
    float *h_Q = (float*)malloc(size);
    float *h_K = (float*)malloc(size);
    float *h_V = (float*)malloc(size);
    float *h_O = (float*)malloc(size);

    // 初始化输入（示例：填充1.0）
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = 1.0f;
        h_K[i] = 1.0f;
        h_V[i] = 1.0f;
    }

    // 分配设备内存
    float *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA_ERROR(cudaMalloc(&d_Q, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_K, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_V, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_O, size));

    // 复制数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice));

    // 调用FlashAttention
    flashAttention(d_Q, d_K, d_V, d_O, N, d);

    // 复制结果回主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_O, d_O, size, cudaMemcpyDeviceToHost));

    // 打印部分结果
    printf("Output (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_O[i]);
    }
    printf("\n");

    // 释放内存
    free(h_Q); free(h_K); free(h_V); free(h_O);
    CHECK_CUDA_ERROR(cudaFree(d_Q));
    CHECK_CUDA_ERROR(cudaFree(d_K));
    CHECK_CUDA_ERROR(cudaFree(d_V));
    CHECK_CUDA_ERROR(cudaFree(d_O));

    return 0;
}