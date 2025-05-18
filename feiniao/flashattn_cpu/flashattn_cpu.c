#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void softmax(float *input, int length) {
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        float val = expf(input[i] - max_val);
        sum += val;
        input[i] = val;
    }
    for (int i = 0; i < length; i++) {
        input[i] /= sum;
    }
}

void matmul_transposeB(const float *A, const float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int p = 0; p < n; p++) {
                sum += A[i * n + p] * B[j * n + p];
            }
            C[i * k + j] = sum;
        }
    }
}

void matmul(const float *A, const float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int p = 0; p < n; p++) {
                sum += A[i * n + p] * B[p * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

void attention_cpu(const float *Q, const float *K, const float *V, float *O, int N, int d) {
    float *scores = (float*)malloc(N * N * sizeof(float));
    if (!scores) {
        fprintf(stderr, "Memory allocation failed for scores\n");
        exit(1);
    }
    float scale = 1.0f / sqrtf((float)d);

    matmul_transposeB(Q, K, scores, N, d, N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            scores[i * N + j] *= scale;
        }
        softmax(&scores[i * N], N);
    }

    printf("Attention Scores (first 5 rows):\n");
    for (int i = 0; i < 5 && i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.4f ", scores[i * N + j]);
        }
        printf("\n");
    }

    matmul(scores, V, O, N, N, d);

    printf("\nOutput (O, first 5 rows):\n");
    for (int i = 0; i < N; i++) {
        printf(" out i %d O %f \n",i, O[i*d]);
        // for (int j = 0; j < d; j++) {
        //     printf("%.4f ", O[i * d + j]);
        // }
    }

    free(scores);
}

int main() {
    int N = 64;
    int d = 64;
    size_t size = N * d * sizeof(float);

    float *h_Q = (float*)malloc(size);
    float *h_K = (float*)malloc(size);
    float *h_V = (float*)malloc(size);
    float *h_O = (float*)malloc(size);
    if (!h_Q || !h_K || !h_V || !h_O) {
        fprintf(stderr, "Memory allocation failed\n");
        free(h_Q); free(h_K); free(h_V); free(h_O);
        return 1;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            h_Q[i * d + j] = i * 0.02 + j * 0.02;
            h_K[i * d + j] = i * 0.02 + j * 0.02;
            h_V[i * d + j] = i * 0.02 + j * 0.02;
            h_O[i * d + j] = 0.0f;
        }
    }

    attention_cpu(h_Q, h_K, h_V, h_O, N, d);

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);

    return 0;
}