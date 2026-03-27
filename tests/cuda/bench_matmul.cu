#include "lunara/runtime/cpu_ref.h"
#include "lunara/runtime/cuda/matmul.h"
#include "lunara/util/timer.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

void fill_matrix(std::vector<float>& mat, int rows, int cols) {
    for(int i=0;i<rows*cols;i++) {
        mat[i] = float(i%100)/100.f;
    }
}

int main() {
    std::vector<std::pair<int,int>> shapes = {
        {256, 256}, {512, 512}, {1024, 1024}, {1024, 2048}
    };

    for(auto [M,N] : shapes) {
        int K = N;

        std::vector<float> hA(M*K), hB(K*N), hC(M*N), hC_ref(M*N);

        fill_matrix(hA,M,K);
        fill_matrix(hB,K,N);

        float *dA,*dB,*dC;
        cudaMalloc(&dA,M*K*sizeof(float));
        cudaMalloc(&dB,K*N*sizeof(float));
        cudaMalloc(&dC,M*N*sizeof(float));

        cudaMemcpy(dA,hA.data(),M*K*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(dB,hB.data(),K*N*sizeof(float),cudaMemcpyHostToDevice);

        lunara::rt::cuda::launch_matmul(dA,dB,dC,M,N,K);
        cudaDeviceSynchronize();

        lunara::Timer t;
        t.start();
        lunara::rt::cuda::launch_matmul(dA,dB,dC,M,N,K);
        cudaDeviceSynchronize();
        t.stop();

        cudaMemcpy(hC.data(),dC,M*N*sizeof(float),cudaMemcpyDeviceToHost);

        if(M <= 512) {
            lunara::rt::cpu::matmul_ref(
                {hA.data(),{M,K}}, {hB.data(),{K,N}}, {hC_ref.data(),{M,N}}
            );

            float max_err = 0.f;
            for(int i=0;i<M*N;i++) {
                max_err = std::max(max_err, std::abs(hC[i]-hC_ref[i]));
            }
            std::cout << "Shape " << M << "x" << K << "x" << N << " max error: " << max_err << "\n";
        }

        std::cout << "Shape " << M << "x" << K << "x" << N << " GPU time: " << t.ms() << " ms\n";

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    return 0;
}