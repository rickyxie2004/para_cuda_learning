#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>

const char* version_name = "Optimized implementation.";
#define TILE_SIZE 16
#define BLOCK_SIZE 128

__global__ void mat_QKT(float* Q, float* K, float* QK_T, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += Q[row * n + k] * K[col * n + k]; 
        }
        QK_T[row * n + col] = sum;
    }
}

__global__ void scale_QKT(float* QK_T, int n, float scale){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<n*n){
        QK_T[idx] *= scale;
    }
}

__global__ void softmax(float* QK_T, int n){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < n){
        float maxn = -INFINITY;
        for (int i=0;i<n/2;i++){
            float val = QK_T[row*n+i];
            if(val>maxn) maxn = val;
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < n; ++j) {
            float e = expf(QK_T[row * n + j] - maxn);
            QK_T[row * n + j] = e;
            sum_exp += e;
        }
        for (int j = 0; j < n; ++j) {
            QK_T[row * n + j] /= sum_exp;
        }

    }
}

__global__ void matmul_attention(float* QK_T, float* V, float* Y, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < n && col < n) {
        float sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += QK_T[row * n + k] * V[k * n + col];
        }
        Y[row * n + col] = sum;
    }
}

__global__ void tiled_matmul_QKT(float* Q, float* K, float* QK_T, int n){
    __shared__ float tile_Q[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_K[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for(int i=0; i<(n+TILE_SIZE-1)/TILE_SIZE; i++){
        if(row<n && i*TILE_SIZE + threadIdx.x<n){
            tile_Q[threadIdx.y][threadIdx.x] = Q[row*n + i*TILE_SIZE + threadIdx.x];
        } else {
            tile_Q[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if(col<n && i*TILE_SIZE + threadIdx.y<n){
            tile_K[threadIdx.y][threadIdx.x] = K[col * n + i * TILE_SIZE + threadIdx.y];
        } else {
            tile_K[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for(int k=0; k<TILE_SIZE; k++){
            sum += tile_Q[threadIdx.y][k] * tile_K[k][threadIdx.x];
        }
        __syncthreads();

    }
    if(row<n && col<n){
        QK_T[row*n + col] = sum;
    }
}

__global__ void tiled_matmul_attention(float* QK_T, float* V, float* Y, int n){
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < n && t * TILE_SIZE + threadIdx.x < n)
            tile_A[threadIdx.y][threadIdx.x] = QK_T[row * n + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0;

        if (col < n && t * TILE_SIZE + threadIdx.y < n)
            tile_B[threadIdx.y][threadIdx.x] = V[(t * TILE_SIZE + threadIdx.y) * n + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        Y[row * n + col] = sum;
}
void square_attention (int n, float* gpu_Q, float* gpu_K, float* gpu_V, float* gpu_Y)
{
    float* gpu_QK_T;
    cudaMalloc(&gpu_QK_T, n*n*sizeof(float));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    tiled_matmul_QKT<<<gridDim,blockDim>>>(gpu_Q,gpu_K,gpu_QK_T,n);

    int tot = n*n;
    scale_QKT<<<(tot+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(gpu_QK_T,n, 1.0f / sqrtf(float(n)));

    softmax<<<(n+BLOCK_SIZE)-1/BLOCK_SIZE,BLOCK_SIZE>>>(gpu_QK_T,n);

    tiled_matmul_attention<<<gridDim, blockDim>>>(gpu_QK_T, gpu_V, gpu_Y, n);

    cudaFree(gpu_QK_T);
}
