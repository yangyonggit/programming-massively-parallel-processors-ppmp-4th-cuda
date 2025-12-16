#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "cdiv.h"

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif
#ifndef COARSE_FACTOR
#define COARSE_FACTOR 4
#endif

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

static void checkCublas(cublasStatus_t stat, const char* msg) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << msg << " (code " << stat << ")" << std::endl;
        std::exit(1);
    }
}

// Thread coarsening tiled matrix multiplication
// Each thread computes COARSE_FACTOR output columns in the same row
__global__ void matrixMulKernelCoarsened(const float* __restrict__ M,
                                         const float* __restrict__ N,
                                         float* __restrict__ P,
                                         int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    float Pvalue[COARSE_FACTOR];
    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        Pvalue[c] = 0.0f;
    }

    // ph iterates over tiles along the K dimension
    for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {
        // Load current tiles into shared memory
        Mds[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];
        // For coarsened columns, we will load N tile per coarse step inside loop
        __syncthreads();

        // Compute contributions for COARSE_FACTOR columns
        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int col = colStart + c * TILE_WIDTH; // shift by a tile width per coarse column
            // Load N tile column corresponding to current coarse column
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();
        }
    }

    // Write results
    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int col = colStart + c * TILE_WIDTH;
        P[row * width + col] = Pvalue[c];
    }
}

void launchCoarsened(const float* h_M, const float* h_N, float* h_P,
                     int width, float* outTime, int numRuns = 1000) {
    const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(width) * sizeof(float);
    float *d_M = nullptr, *d_N = nullptr, *d_P = nullptr;
    checkCuda(cudaMalloc(&d_M, bytes), "cudaMalloc d_M");
    checkCuda(cudaMalloc(&d_N, bytes), "cudaMalloc d_N");
    checkCuda(cudaMalloc(&d_P, bytes), "cudaMalloc d_P");

    checkCuda(cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice), "H2D M");
    checkCuda(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice), "H2D N");

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((width / TILE_WIDTH) / COARSE_FACTOR, (width / TILE_WIDTH));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    matrixMulKernelCoarsened<<<grid, block>>>(d_M, d_N, d_P, width);
    checkCuda(cudaGetLastError(), "coarsened warmup launch");
    checkCuda(cudaDeviceSynchronize(), "coarsened warmup sync");

    // Timed runs
    float totalMs = 0.0f;
    for (int i = 0; i < numRuns; ++i) {
        cudaEventRecord(start);
        matrixMulKernelCoarsened<<<grid, block>>>(d_M, d_N, d_P, width);
        cudaEventRecord(stop);
        checkCuda(cudaGetLastError(), "coarsened timed launch");
        checkCuda(cudaEventSynchronize(stop), "coarsened event sync");
        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "coarsened elapsed");
        totalMs += ms;
    }
    if (outTime) *outTime = totalMs / numRuns;

    checkCuda(cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost), "D2H P");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

// cuBLAS validation and timing (warmup + avg over numRuns)
bool validateWithCublas(cublasHandle_t handle,
                        const float* h_M, const float* h_N, const float* h_P,
                        int width, float* outMaxAbsDiff, float* outTime, int numRuns = 1000) {
    const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(width) * sizeof(float);
    float *d_M = nullptr, *d_N = nullptr, *d_P = nullptr;
    checkCuda(cudaMalloc(&d_M, bytes), "cudaMalloc d_M (cublas)");
    checkCuda(cudaMalloc(&d_N, bytes), "cudaMalloc d_N (cublas)");
    checkCuda(cudaMalloc(&d_P, bytes), "cudaMalloc d_P (cublas)");

    // Transpose to column-major for cuBLAS
    std::vector<float> h_M_col(bytes/sizeof(float));
    std::vector<float> h_N_col(bytes/sizeof(float));
    for (int r = 0; r < width; ++r) {
        for (int c = 0; c < width; ++c) {
            h_M_col[c * width + r] = h_M[r * width + c];
            h_N_col[c * width + r] = h_N[r * width + c];
        }
    }
    checkCuda(cudaMemcpy(d_M, h_M_col.data(), bytes, cudaMemcpyHostToDevice), "H2D M (cublas)");
    checkCuda(cudaMemcpy(d_N, h_N_col.data(), bytes, cudaMemcpyHostToDevice), "H2D N (cublas)");

    const float alpha = 1.0f, beta = 0.0f;

    // Warm-up (not timed)
    checkCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            width, width, width,
                            &alpha, d_M, width, d_N, width,
                            &beta, d_P, width), "cublas warmup");

    // Timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float totalMs = 0.0f;
    for (int i = 0; i < numRuns; ++i) {
        cudaEventRecord(start);
        checkCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                width, width, width,
                                &alpha, d_M, width, d_N, width,
                                &beta, d_P, width), "cublas timed");
        cudaEventRecord(stop);
        checkCuda(cudaEventSynchronize(stop), "cublas event sync");
        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "cublas elapsed");
        totalMs += ms;
    }
    if (outTime) *outTime = totalMs / numRuns;

    // Copy back result and convert to row-major for comparison
    std::vector<float> h_P_col(bytes/sizeof(float));
    checkCuda(cudaMemcpy(h_P_col.data(), d_P, bytes, cudaMemcpyDeviceToHost), "D2H P (cublas)");
    std::vector<float> h_P_ref(bytes/sizeof(float));
    for (int r = 0; r < width; ++r) {
        for (int c = 0; c < width; ++c) {
            h_P_ref[r * width + c] = h_P_col[c * width + r];
        }
    }

    float maxAbsDiff = 0.0f;
    for (int i = 0; i < width * width; ++i) {
        float diff = std::fabs(h_P_ref[i] - h_P[i]);
        if (diff > maxAbsDiff) maxAbsDiff = diff;
    }
    if (outMaxAbsDiff) *outMaxAbsDiff = maxAbsDiff;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    const float tol = 1e-3f;
    return maxAbsDiff < tol;
}

int main(int argc, char** argv) {
    int width = 1024; // default, must be multiple of TILE_WIDTH and COARSE_FACTOR*TILE_WIDTH fits width
    if (argc >= 2) {
        width = std::atoi(argv[1]);
        if (width <= 0) width = 1024;
    }
    if (width % TILE_WIDTH != 0 || (width / TILE_WIDTH) % COARSE_FACTOR != 0) {
        std::cout << "Width must be multiple of TILE_WIDTH and grid x divisible by COARSE_FACTOR. Using adjusted width." << std::endl;
        width = ((width + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
        int tilesX = width / TILE_WIDTH;
        tilesX = (tilesX / COARSE_FACTOR) * COARSE_FACTOR;
        width = tilesX * TILE_WIDTH;
    }

    std::cout << "Thread Coarsening GEMM: " << width << "x" << width << std::endl;
    std::cout << "Tile: " << TILE_WIDTH << ", Coarse: " << COARSE_FACTOR << std::endl;
    std::cout << "Benchmark runs: 1000 (average, with warmup)" << std::endl << std::endl;

    const size_t elems = static_cast<size_t>(width) * static_cast<size_t>(width);
    std::vector<float> M(elems), N(elems), P(elems);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < elems; ++i) {
        M[i] = dist(rng);
        N[i] = dist(rng);
    }

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate");

    float timeCoarse = 0.0f;
    launchCoarsened(M.data(), N.data(), P.data(), width, &timeCoarse, 1000);
    std::cout << "Coarsened kernel avg time: " << timeCoarse << " ms" << std::endl;

    float timeCublas = 0.0f, maxDiff = 0.0f;
    bool ok = validateWithCublas(handle, M.data(), N.data(), P.data(), width, &maxDiff, &timeCublas, 1000);
    std::cout << "cuBLAS SGEMM avg time: " << timeCublas << " ms" << std::endl;
    std::cout << "Validation vs cuBLAS: " << (ok ? "OK" : "FAIL") << ", maxAbsDiff=" << maxDiff << std::endl;

    std::cout << "cuBLAS is " << (timeCoarse / timeCublas) << "x faster than coarsened kernel" << std::endl;

    cublasDestroy(handle);
    return ok ? 0 : 1;
}
