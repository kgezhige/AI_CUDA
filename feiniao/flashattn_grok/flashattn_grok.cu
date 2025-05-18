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
    float* S_O = s_S + block_size * block_size;  // +上一个内存的长度

    float* max_rowi = S_O + block_size * d; //   max_rowi[block_size] 
    float* lg = max_rowi + block_size;      //   lg[block_size] 
    // float* max_g = lg + block_size;         // max;
    // float* maxold = max_g + block_size;         // max;

    printf("kernel %d ++ %d \n ",blockIdx.x,  threadIdx.x);
    // 初始化 S_O
    if (threadIdx.x < block_size) {
        for (int k = 0; k < d; k++) {
            S_O[threadIdx.x * d + k] = 0.0f;
        }
    }
    __syncthreads();

    // 加载 Q
    // 0号线程负责 搬运到 s_Q[0]
    // 1号线程负责 搬运到 s_Q[1],1+bD, 0+2 *bd] 即 [0, 64, 128]
    // 一轮循环后 
    // 0号线程负责 搬运到 s_Q[0 + bD] s_Q[0 + 64] 
    // 1号线程负责 搬运到 s_Q[1 + bD] s_Q[1 + 64] 
    // 所以 i 会遍历线程搬运到所有的 s_Q 
    // 并且由于 线程连续， 会合并访存和 避免warp conflict
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

    // float m = -1e10;
    __shared__ float max_g ;    // 全局最大值
    __shared__ float maxold ;   // 全局最大值上一轮未更新
    max_g = 0; maxold = 0;
    // 按块遍历 K 和 V
    if (threadIdx.x < block_size) {
        max_rowi[threadIdx.x] = -1e10;
        lg[threadIdx.x] = 0.0;
    }
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

        // Score 矩阵赋值为 0 
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
            if (row + i >= N) continue;     // 纵轴为行，也是 线程 的序列数
            for (int j = 0; j < block_size; j++) {
                float score = 0.0;
                for (int k = 0; k < d; k++) {
                    score += s_Q[i * d + k] * s_K[j * d + k];   // 不用转制
                }
                score /= sqrtf((float)d);
                s_S[i * block_size + j] = score;

            }
        }
        __syncthreads();

        // 这里计算分块分数矩阵s(32,32)的每一行的最大值。每个线程负责一行32元素的最大值计算。
        // 即每一行都有一个寄存器变量row_max存储最大值，最后放在对应线程的 max_rowi[threadIdx.x]
        // 线程 0：计算 s_S[0 * 32 + 0] 到 s_S[0 * 32 + 31] 的最大值，存入 max_rowi[0]。
        // 线程 1：计算 s_S[1 * 32 + 0] 到 s_S[1 * 32 + 31] 的最大值，存入 max_rowi[1]。 线程 32 到 63：空闲，不执行任何操作。
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            float row_max = -1e10;
            for (int j = 0; j < block_size; j++) {
                float val = s_S[i * block_size + j];
                if (isfinite(val)) {
                    row_max = fmaxf(row_max, val);
                }
            }
            max_rowi[threadIdx.x] = row_max;
            // printf("row max %f \n", row_max);
        }
        // if ((blockIdx.x == 0) && (threadIdx.x ==0))
        // printf(" %d ++ %d  score s%f  %f , %f \n ",blockIdx.x,  threadIdx.x, max_rowi[0],max_rowi[1] ,max_rowi[31]);

        // 聚合 max_rowi 中的 32 个值只需要一个线程完成。限制仅线程 0 执行，防止多个线程同时操作 new_m 和 shared_new_m，避免数据竞争
        // 比如score中32x32矩阵，虽然每一行的最大值可能不一样，但是总的来说，每个元素都除以了小块的最大值，softmax分子分母也都是同时除以了最大值。所以最终的softmax都是不变的。
        // 计算 s_S 每行的最大值（row_max），存储到 shared_max。 聚合所有行的最大值，更新全局最大值 new_m。将结果存储到 shared_new_m，供后续 softmax 使用。        
        __syncthreads();
        maxold = max_g;
        if (threadIdx.x == 0) {
            float new_m = 0;
            new_m =  -1e10;;
            for (int i = 0; i < block_size; i++) {
                new_m = fmaxf(new_m, max_rowi[i]);
            }
            max_g = new_m;    //当前字块的最大值，也是全局的最大值
        }

        // 这里计算分块分数矩阵s(32,32)的每一行的e^{val-new_m}求和。每个线程负责一行32元素取e^{val-new_m}后的求和 lb 以及修正的lg。线程 32 到 63：空闲
        __syncthreads();
        if ((threadIdx.x <1))
        printf(" max b %d t %d ___ col %d max s%f  %f  \n ",blockIdx.x,  threadIdx.x, col, maxold, max_g);
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            float lb = 0.0; //寄存器变量，存储子块的 lb
            for (int j = 0; j < block_size; j++) {
                float val = s_S[i * block_size + j];

                if (isfinite(val)) {
                    s_S[i * block_size + j] = expf(val - max_g);
                    lb += s_S[i * block_size + j];  // 此时row sum 相当于lb 当前一块内的值求和
                } else {
                    printf(" finite ");
                    s_S[i * block_size + j] = 0.0;
                }
                // if ((threadIdx.x ==0) && (blockIdx.x==0))
                // printf(" thd b %d t %d ___ col %d val %f maxg %f s %f lb %f j %d \n ",blockIdx.x,  threadIdx.x, col, val, max_g, s_S[i * block_size + j], lb, j );
            }
            // 计算 lg
            float lgold;
            lgold = lg[threadIdx.x];
            lg[threadIdx.x] = lgold * expf(maxold - max_g) + lb ;//* expf(mb - m_i);            
            if ((threadIdx.x ==0) && (blockIdx.x==1))
            printf(" bx, %d tx %d maxold %f, max_g %f ,lg %f lgo %f lb, %f col %d \n ",blockIdx.x,  threadIdx.x,maxold,max_g, lg[threadIdx.x],lgold, lb,col);
        }
        __syncthreads();

        // 更新输出 S_O
        // 这里计算分块分数矩阵输出O 的每一行的输出o。每个线程负责一行32元素的O输出计算。线程 32 到 63：空闲
        //每个线程计算输出o的某一行，的第一列元素（通过S与矩阵V相乘），然后计算第二列元素。直到一列都计算完成
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            if (row + i >= N) continue;
            float l_old = lg[threadIdx.x]; // 获取当前累积和（包含之前块的信息）
            // float m_i = fmaxf(old_m, new_m);       // 更新全局最大值
            for (int k = 0; k < d; k++) {
                float Ob = 0.0; // 当前子块的 O_b[i]
                for (int j = 0; j < block_size; j++) {
                    Ob += s_S[i * block_size + j] * s_V[j * d + k]; // 使用未归一化的 s_S
                }//           ^ 这里有 i ，表示每个线程负责 s_S 的不同的行。 i = [0,1,..32]  分别计算 s_S不同行的值
                // 递推更新 S_O
                S_O[threadIdx.x * d + k] = (S_O[threadIdx.x * d + k] * expf(maxold - max_g)) + (Ob); //* expf(new_m - m_i));
            }
        }
        __syncthreads();

    }
    // 之前的 S_O 都不是最终的输出。需要在一行计算完之后 写回 全局内存
    __syncthreads();
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
        if (row + i >= N) continue;
        float l_i = lg[threadIdx.x]; // 全局累积和
        for (int k = 0; k < d; k++) {
            O[(row + i) * d + k] = S_O[threadIdx.x * d + k] / l_i; // 写入最终输出
            // 这里因为 S_O 是 共享内存的矩阵， 在block 0, block 1中 ，其 位置是相等的。
            // 而 O 是全局内存的值， block 0, row = 0,要写在所以 O的行索引为 [0:31]
            // block 1, row = 32,要写在所以 O的行索引为 [32:63] 所以不能覆盖了 block 0 的输出值 O
        }
    }
    if ((threadIdx.x ==0))
    printf(" bx, %d tx %d row %d  \n ",blockIdx.x,  threadIdx.x,row);
 
}

void flashAttention(float* Q, float* K, float* V, float* O, int N, int d) {
    int block_size = 40;
    int grid_size = (N + block_size - 1) / block_size;  // grid size 为 序列长度 N 除以 block size 
    size_t shared_mem_size = (3 * block_size * d + block_size * block_size + block_size * d + block_size * 2) * sizeof(float);
    printf("grid %d, block %d shared_mem_size %d  ", grid_size, block_size, shared_mem_size);
    printf("Launching kernel with grid_size=%d, block_size=%d, shared_mem=%zu\n", grid_size, 64, shared_mem_size);
    flashAttentionKernel<<<grid_size, 64, shared_mem_size>>>(Q, K, V, O, N, d, block_size);
    cudaError_t err = cudaGetLastError();
    printf("cudaGetLastError: %s (%d)\n", cudaGetErrorString(err), err);
    CHECK_CUDA_ERROR(err);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Max shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Max shared memory per block (opt-in): %zu bytes\n", prop.sharedMemPerBlockOptin);
    int N = 128;
    int d = 64;
    size_t size = N * d * sizeof(float);

    float *h_Q = (float*)malloc(size);
    float *h_K = (float*)malloc(size);
    float *h_V = (float*)malloc(size);
    float *h_O = (float*)malloc(size);

    // srand(42);
    for (int i = 0; i < N ; i++) {
        for (int j = 0; j< d ; j++)
        {
            h_Q[i * d + j] = i * 0.02 + j * 0.02;
            h_K[i * d + j] = i * 0.02 + j * 0.02;
            h_V[i * d + j] = i * 0.02 + j * 0.02;
        }
    }    
    printf("Q:+ %f %f %f %f %f",h_Q[0],h_Q[2],h_Q[d+1],h_Q[2*d+1],h_Q[(N-1)*d+1]);
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

    printf("\n Output ho 0,10:\n");
    for (int i = 0; i < 64; i++) {
        printf("%d  %f \n ", i,h_O[i*64]);
    }
    // printf("\n Output h_o 64,74 :\n");
    // for (int i = 64; i < 74; i++) {
    //     printf("%f ", h_O[i]);
    // }
    // printf("\n");
    // for (int i = 64*2; i < 64*2; i++) {
    //     printf("%f ", h_O[i]);
    // }
    printf("\n");

    free(h_Q); free(h_K); free(h_V); free(h_O);
    CHECK_CUDA_ERROR(cudaFree(d_Q));
    CHECK_CUDA_ERROR(cudaFree(d_K));
    CHECK_CUDA_ERROR(cudaFree(d_V));
    CHECK_CUDA_ERROR(cudaFree(d_O));

    return 0;
}