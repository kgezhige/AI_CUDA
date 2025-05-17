// flashattn_grok_change_from_mimal.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Macro for CUDA error checking
#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Structure to hold attention configuration parameters
struct AttentionConfig {
    int Bc;            // Column block size for K and V
    int Br;            // Row block size for Q
    int Tc;            // Number of column tiles
    int Tr;            // Number of row tiles
    size_t shared_mem_size; // Shared memory size in bytes
};

// Configures block and tile sizes based on input dimensions and hardware constraints
AttentionConfig configure_attention(int N, int d, int max_shared_mem) {
    AttentionConfig config;
    
    // Initialize default block sizes (aim for square blocks for efficiency)
    config.Bc = 32; // Column block size
    config.Br = 32; // Row block size
    
    // Calculate shared memory needed: 3 * Bc * d (for Qi, Kj, Vj) + Bc * Br (for S)
    size_t sram_per_block = (3 * config.Bc * d + config.Bc * config.Br) * sizeof(float);
    
    // Reduce block sizes if shared memory exceeds hardware limit
    while (sram_per_block > max_shared_mem && config.Bc > 8) {
        config.Bc /= 2;
        config.Br /= 2;
        sram_per_block = (3 * config.Bc * d + config.Bc * config.Br) * sizeof(float);
    }
    
    // Compute number of tiles (ceiling division)
    config.Tc = (N + config.Bc - 1) / config.Bc;
    config.Tr = (N + config.Br - 1) / config.Br;
    config.shared_mem_size = sram_per_block;
    
    // Print configuration for debugging
    printf("Attention Config: Bc=%d, Br=%d, Tc=%d, Tr=%d, SharedMem=%zu bytes\n",
           config.Bc, config.Br, config.Tc, config.Tr, config.shared_mem_size);
    
    return config;
}

// CUDA kernel for computing scaled dot-product attention
__global__ void attention_kernel(
    const float* Q, const float* K, const float* V, // Input tensors
    int N, int d, int Tc, int Tr, int Bc, int Br,  // Dimensions and tile sizes
    float softmax_scale, float* l, float* m, float* O // Scaling factor and outputs
) {
    // Thread and block indices
    int tx = threadIdx.x; // Thread index within block
    int bx = blockIdx.x;  // Batch index
    int by = blockIdx.y;  // Head index

    // Compute offsets for Q, K, V, O (shape: [B, nh, N, d]) and l, m (shape: [B, nh, N])
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = num_heads
    int lm_offset = (bx * gridDim.y * N) + (by * N);

    // Allocate shared memory for Qi, Kj, Vj (Bc * d each) and S (Bc * Br)
    extern __shared__ float sram[];
    float* Qi = sram;                    // Query tile: Br * d
    float* Kj = sram + (Bc * d);         // Key tile: Bc * d
    float* Vj = sram + (2 * Bc * d);     // Value tile: Bc * d
    float* S = sram + (3 * Bc * d);      // Score matrix: Br * Bc

    // Loop over column tiles (process K and V in blocks of Bc)
    for (int j = 0; j < Tc; j++) {
        // Load Kj and Vj from global to shared memory
        for (int x = 0; x < d; x++) {
            int global_idx = qkv_offset + (j * Bc * d) + (tx * d) + x;
            // Boundary check to avoid out-of-bounds access
            Kj[(tx * d) + x] = (tx < Bc && global_idx < qkv_offset + N * d) ?
                               K[global_idx] : 0.0f;
            Vj[(tx * d) + x] = (tx < Bc && global_idx < qkv_offset + N * d) ?
                               V[global_idx] : 0.0f;
        }
        __syncthreads(); // Ensure Kj and Vj are fully loaded

        // Loop over row tiles (process Q in blocks of Br)
        for (int i = 0; i < Tr; i++) {
            // Load Qi from global to shared memory and l, m to registers
            for (int x = 0; x < d; x++) {
                int global_idx = qkv_offset + (i * Br * d) + (tx * d) + x;
                Qi[(tx * d) + x] = (tx < Br && global_idx < qkv_offset + N * d) ?
                                   Q[global_idx] : 0.0f;
            }
            // Load previous max (m) and sum (l) for online softmax
            float row_m_prev = (tx < Br) ? m[lm_offset + (i * Br) + tx] : -INFINITY;
            float row_l_prev = (tx < Br) ? l[lm_offset + (i * Br) + tx] : 0.0f;

            // Compute S = QK^T / sqrt(d) and find row-wise max
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0.0f;
                // Dot product between Qi[tx] and Kj[y]
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale; // Scale by 1/sqrt(d)
                S[(tx * Bc) + y] = sum; // Store score
                row_m = fmaxf(row_m, sum); // Update row max
            }

            // Compute P = exp(S - row_m) and row-wise sum for softmax
            float row_l = 0.0f;
            for (int y = 0; y < Bc; y++) {
                S[(tx * Bc) + y] = expf(S[(tx * Bc) + y] - row_m);
                row_l += S[(tx * Bc) + y];
            }

            // Update m and l using online softmax formula
            float row_m_new = fmaxf(row_m_prev, row_m);
            float row_l_new = (expf(row_m_prev - row_m_new) * row_l_prev) +
                             (expf(row_m - row_m_new) * row_l);

            // Compute output O = P * V and update incrementally
            for (int x = 0; x < d; x++) {
                float pv = 0.0f; // Compute Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(tx * Bc) + y] * Vj[(y * d) + x];
                }
                int out_idx = qkv_offset + (i * Br * d) + (tx * d) + x;
                if (tx < Br && out_idx < qkv_offset + N * d) {
                    float prev_o = O[out_idx];
                    // Update output: O = (prev_O * prev_l * exp(prev_m - new_m) + new_pv * exp(m - new_m)) / new_l
                    O[out_idx] = (1.0f / row_l_new) *
                                 ((row_l_prev * expf(row_m_prev - row_m_new) * prev_o) +
                                  (expf(row_m - row_m_new) * pv));
                }
            }

            // Write updated l and m to global memory
            if (tx < Br) {
                m[lm_offset + (i * Br) + tx] = row_m_new;
                l[lm_offset + (i * Br) + tx] = row_l_new;
            }
            __syncthreads(); // Ensure all threads finish before next iteration
        }
    }
}

// Launches the attention kernel with the specified configuration
void launch_attention_kernel(
    float* d_Q, float* d_K, float* d_V, float* d_O, // Device pointers
    int B, int nh, int N, int d, float softmax_scale, // Dimensions and scaling
    float* d_l, float* d_m, const AttentionConfig& config // Aux arrays and config
) {
    // Set grid and block dimensions
    dim3 grid_dim(B, nh); // One block per batch and head
    dim3 block_dim(config.Bc); // Bc threads per block

    // Launch kernel with dynamic shared memory
    attention_kernel<<<grid_dim, block_dim, config.shared_mem_size>>>(
        d_Q, d_K, d_V, N, d, config.Tc, config.Tr, config.Bc, config.Br,
        softmax_scale, d_l, d_m, d_O
    );
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
}

int main() {
    // Define problem dimensions
    const int B = 1;    // Batch size
    const int nh = 1;   // Number of attention heads
    const int N = 512;  // Sequence length
    const int d = 64;   // Head dimension
    
    // Compute softmax scaling factor
    const float softmax_scale = 1.0f / sqrtf((float)d);
    
    // Calculate memory sizes
    size_t qkv_size = B * nh * N * d * sizeof(float); // Size for Q, K, V, O
    size_t lm_size = B * nh * N * sizeof(float);      // Size for l, m
    
    // Allocate host memory
    float *h_Q = (float*)malloc(qkv_size);
    float *h_K = (float*)malloc(qkv_size);
    float *h_V = (float*)malloc(qkv_size);
    float *h_O = (float*)malloc(qkv_size);
    float *h_l = (float*)malloc(lm_size);
    float *h_m = (float*)malloc(lm_size);
    
    // Check for allocation failures
    if (!h_Q || !h_K || !h_V || !h_O || !h_l || !h_m) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize input data with random values
    srand(42);
    for (int i = 0; i < B * nh * N * d; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f; // Range: [-0.005, 0.005]
        h_K[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        h_V[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }
    // Initialize l and m for online softmax
    for (int i = 0; i < B * nh * N; i++) {
        h_l[i] = 0.0f;
        h_m[i] = -INFINITY;
    }
    
    // Print first 5 elements of Q, K, V for first batch and head
    printf("\nFirst 5 elements of Q[0,0,0:5,0]:\n");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", h_Q[i * d]);
    }
    printf("\nFirst 5 elements of K[0,0,0:5,0]:\n");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", h_K[i * d]);
    }
    printf("\nFirst 5 elements of V[0,0,0:5,0]:\n");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", h_V[i * d]);
    }
    printf("\n");
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m;
    CHECK_CUDA_ERROR(cudaMalloc(&d_Q, qkv_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_K, qkv_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_V, qkv_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_O, qkv_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_l, lm_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_m, lm_size));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_l, h_l, lm_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_m, h_m, lm_size, cudaMemcpyHostToDevice));
    
    // Get maximum shared memory per block
    int max_shared_mem;
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    
    // Configure attention parameters
    AttentionConfig config = configure_attention(N, d, max_shared_mem);
    
    // Launch attention kernel
    launch_attention_kernel(d_Q, d_K, d_V, d_O, B, nh, N, d, softmax_scale, d_l, d_m, config);
    
    // Copy output back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_O, d_O, qkv_size, cudaMemcpyDeviceToHost));
    
    // Print first 5 elements of O for first batch and head
    printf("First 5 elements of O[0,0,0:5,0]:\n");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", h_O[i * d]);
    }
    printf("\n");
    
    // Free device memory
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    cudaFree(d_l); cudaFree(d_m);
    
    // Free host memory
    free(h_Q); free(h_K); free(h_V); free(h_O);
    free(h_l); free(h_m);
    
    return 0;
}