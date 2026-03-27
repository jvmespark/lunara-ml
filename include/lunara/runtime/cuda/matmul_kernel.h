#pragma once
#include <cuda_runtime.h>

namespace lunara::rt::cuda {

template <int TILE_M = 16, int TILE_N = 16, int TILE_K = 16>
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float acc = 0.f;

    for (int tile_idx = 0; tile_idx < (K + TILE_K - 1)/TILE_K; tile_idx++) {
        if (row < M && tile_idx*TILE_K + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row*K + tile_idx*TILE_K + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.f;
        }
        
        if (tile_idx*TILE_K + threadIdx.y < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile_idx*TILE_K + threadIdx.y)*N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_K; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row*N + col] = acc;
    }
}

}