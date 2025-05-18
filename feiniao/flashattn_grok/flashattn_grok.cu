#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

// 检查CUDA错误
#define CHECK_CUDA_ERROR(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (%d) at %s:%d\n", cudaGetErrorString(e), e, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// FlashAttention核函数
__global__ void flashAttentionKernel(
    float* Q, float* K, float* V, float* O,
    int N, int d, int block_size
) {
    int row = blockIdx.x * block_size;
    if (row >= N) return;

    extern __shared__ float shared_mem[];
    float* s_Q = shared_mem;
    float* s_K = s_Q + block_size * d;
    float* s_V = s_K + block_size * d;
    float* s_S = s_V + block_size * d;
    float* o = s_S + block_size * block_size;

    // 初始化 o
    if (threadIdx.x < block_size) {
        for (int k = 0; k < d; k++) {
            o[threadIdx.x * d + k] = 0.0f;
        }
    }
    __syncthreads();

    // 加载 Q
    for (int i = threadIdx.x; i < block_size * d; i += blockDim.x) {
        int r = i / d;
        int c = i % d;
        if (row + r < N) {
            s_Q[i] = Q[(row + r) * d + c];
        } else {
            s_Q[i] = 0.0;
        }
    }
    __syncthreads();

    float m = -1e10;

    // 按块遍历 K 和 V
    for (int col = 0; col < N; col += block_size) {
        // 加载 K 和 V
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

        // 初始化 s_S
        // 当 blockDim.x=64  block_size=32 
        // 子块大小为 32 32 ，线程 0,1,，，32 每个线程负责子块的一行32个元素。导致后32个线程没有用。
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            for (int j = 0; j < block_size; j++) {
                s_S[i * block_size + j] = 0.0f;
            }
        }
        __syncthreads();

        // 计算 S = Q @ K^T
        // 当 blockDim.x=64  block_size=32 Q，K 形状都为 32(长度），64 (d)
        // 线程 0,1,，，32 每个线程负责子块（Q或者K）的一行64个元素。导致后32个线程没有用。
        // 每个线程也是负责每个分数矩阵S的一行score,每个score元素是通过一行一纵相乘。
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {    
            if (row + i >= N) continue;
            for (int j = 0; j < block_size; j++) {
                float score = 0.0;
                for (int k = 0; k < d; k++) {
                    score += s_Q[i * d + k] * s_K[j * d + k];
                }
                score /= sqrtf((float)d);
                s_S[i * block_size + j] = score;
            }
        }
        __syncthreads();

        // 在线 Softmax
        __shared__ float shared_max[32];
        __shared__ float shared_sum[32];
        __shared__ float shared_new_m;
        __shared__ float maxold; //全局最大值
        shared_new_m = 0;
        maxold = 0;
        if (threadIdx.x < block_size) {
            shared_max[threadIdx.x] = -1e10;
            shared_sum[threadIdx.x] = 0.0;
        }
        __syncthreads();
        // 这里计算分块分数矩阵s(32,32)的每一行的最大值。每个线程负责一行32元素的最大值计算。
        // 即每一行都有一个寄存器变量row_max存储最大值，最后放在对应线程的 shared_max[threadIdx.x]
        // 线程 0：计算 s_S[0 * 32 + 0] 到 s_S[0 * 32 + 31] 的最大值，存入 shared_max[0]。
        // 线程 1：计算 s_S[1 * 32 + 0] 到 s_S[1 * 32 + 31] 的最大值，存入 shared_max[1]。 线程 32 到 63：空闲，不执行任何操作。
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            float row_max = -1e10;
            for (int j = 0; j < block_size; j++) {
                float val = s_S[i * block_size + j];
                if (isfinite(val)) {
                    row_max = fmaxf(row_max, val);
                }
            }
            shared_max[threadIdx.x] = row_max;
        }
        // 聚合 shared_max 中的 32 个值只需要一个线程完成。限制仅线程 0 执行，防止多个线程同时操作 new_m 和 shared_new_m，避免数据竞争
        // 比如score中32x32矩阵，虽然每一行的最大值可能不一样，但是总的来说，每个元素都除以了小块的最大值，softmax分子分母也都是同时除以了最大值。所以最终的softmax都是不变的。
        // 计算 s_S 每行的最大值（row_max），存储到 shared_max。 聚合所有行的最大值，更新全局最大值 new_m。将结果存储到 shared_new_m，供后续 softmax 使用。        
        __syncthreads();
        maxold = shared_new_m
        if (threadIdx.x == 0) {
            new_m = m;
            for (int i = 0; i < block_size; i++) {
                new_m = fmaxf(new_m, shared_max[i]);
            }
            shared_new_m = new_m;    //当前字块的最大值，也是全局的最大值
        }
        // 这里计算分块分数矩阵s(32,32)的每一行的求和。每个线程负责一行32元素取e^{val-new_m}后的求和。线程 32 到 63：空闲

        __syncthreads();
        // 
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            float row_sum = 0.0; //寄存器变量，存储中间求和结果
            for (int j = 0; j < block_size; j++) {
                float val = s_S[i * block_size + j];
                if (isfinite(val)) {
                    s_S[i * block_size + j] = expf(val - shared_new_m);
                    row_sum += s_S[i * block_size + j];  // 此时row sum 相当于lb 当前一块内的值求和
                } else {
                    s_S[i * block_size + j] = 0.0;
                }
            }
            // 更新全局累积和 l_i
            float l_old = shared_sum[threadIdx.x]; // 之前子块的累积和
            // float m_i = fmaxf(old_m, mb);       // 全局最大值
            // 计算 lg
            shared_sum[threadIdx.x] = l_old * expf(maxold - shared_new_m) + row_sum //* expf(mb - m_i);            
        }
        // __syncthreads();
        // // 归一化 s_S
        // // 每一行的元素除以此行的和。得到softmax
        // for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
        //     float row_sum = shared_sum[threadIdx.x];
        //     if (row_sum == 0.0) row_sum = 1.0;
        //     for (int j = 0; j < block_size; j++) {
        //         s_S[i * block_size + j] /= row_sum;
        //     }
        // }
        __syncthreads();

        // 更新 O
        // 这里计算分块分数矩阵输出O 的每一行的输出o。每个线程负责一行32元素的O输出计算。线程 32 到 63：空闲

        // 更新输出 o
        // float old_m = m; // 保存当前块前的全局最大值
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            if (row + i >= N) continue;
            float l_old = shared_sum[threadIdx.x]; // 获取当前累积和（包含之前块的信息）
            float m_i = fmaxf(old_m, new_m);       // 更新全局最大值
            for (int k = 0; k < d; k++) {
                float Ob = 0.0; // 当前子块的 O_b[i]
                for (int j = 0; j < block_size; j++) {
                    Ob += s_S[i * block_size + j] * s_V[j * d + k]; // 使用未归一化的 s_S
                }
                // 递推更新 o
                o[threadIdx.x * d + k] = (o[threadIdx.x * d + k] * expf(maxold - shared_new_m)) + (Ob) //* expf(new_m - m_i));
            }
        }

        // m = new_m;
        __syncthreads();

        // 调试输出
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("new_m=%f\n", new_m);
            for (int j = 0; j < 5; j++) {
                printf("s_S[0][%d]=%f\n", j, s_S[j]);
            }
            printf("o[0][0:4]=%f %f %f %f\n", o[0], o[1], o[2], o[3]);
        }
        __syncthreads();
    }

    __syncthreads();
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
        if (row + i >= N) continue;
        float l_i = shared_sum[threadIdx.x]; // 全局累积和
        for (int k = 0; k < d; k++) {
            O[(row + i) * d + k] = o[threadIdx.x * d + k] / l_i; // 写入最终输出
        }
    }
    // // 写回 O
    // for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
    //     if (row + i >= N) continue;
    //     for (int k = 0; k < d; k++) {
    //         O[(row + i) * d + k] = o[threadIdx.x * d + k];
    //     }
    // }
}

void flashAttention(float* Q, float* K, float* V, float* O, int N, int d) {
    int block_size = 32;
    int grid_size = (N + block_size - 1) / block_size;
    size_t shared_mem_size = (3 * block_size * d + block_size * block_size + block_size * d) * sizeof(float);

    printf("Launching kernel with grid_size=%d, block_size=%d, shared_mem=%zu\n", grid_size, 64, shared_mem_size);
    flashAttentionKernel<<<grid_size, 64, shared_mem_size>>>(Q, K, V, O, N, d, block_size);
    cudaError_t err = cudaGetLastError();
    printf("cudaGetLastError: %s (%d)\n", cudaGetErrorString(err), err);
    CHECK_CUDA_ERROR(err);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

int main() {
    int N = 512;
    int d = 64;
    size_t size = N * d * sizeof(float);

    float *h_Q = (float*)malloc(size);
    float *h_K = (float*)malloc(size);
    float *h_V = (float*)malloc(size);
    float *h_O = (float*)malloc(size);

    srand(42);
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f; // [-0.005, 0.005]
        h_K[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        h_V[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA_ERROR(cudaMalloc(&d_Q, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_K, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_V, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_O, size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice));

    flashAttention(d_Q, d_K, d_V, d_O, N, d);

    CHECK_CUDA_ERROR(cudaMemcpy(h_O, d_O, size, cudaMemcpyDeviceToHost));

    printf("Output (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_O[i]);
    }
    printf("\n");

    free(h_Q); free(h_K); free(h_V); free(h_O);
    CHECK_CUDA_ERROR(cudaFree(d_Q));
    CHECK_CUDA_ERROR(cudaFree(d_K));
    CHECK_CUDA_ERROR(cudaFree(d_V));
    CHECK_CUDA_ERROR(cudaFree(d_O));

    return 0;
}