#pragma once
#include "matmul_kernel.h"

namespace lunara::rt::cuda {

inline void launch_matmul(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream = 0) {

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1)/threads.x,
                (M + threads.y - 1)/threads.y);

    matmul_kernel<16,16,16><<<blocks, threads, 0, stream>>>(A,B,C,M,N,K);
}

}