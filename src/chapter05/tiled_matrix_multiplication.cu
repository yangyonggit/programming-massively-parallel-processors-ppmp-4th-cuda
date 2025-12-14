#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "cdiv.h"

#define TILE_WIDTH 16

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

// Basic tiled matrix multiplication kernel (assumes Width is multiple of TILE_WIDTH)
__global__ void matrixMulKernel(float* M, float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over the M and N tiles required to compute P element
    float Pvalue = 0;
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
        
        // Collaborative loading of M and N tiles into shared memory
        Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    P[Row*Width + Col] = Pvalue;
}

// Tiled kernel with boundary checking (supports arbitrary Width)
__global__ void matrixMulKernelBoundary(float* M, float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    for (int ph = 0; ph < cdiv(Width, TILE_WIDTH); ++ph) {
        
        // Collaborative loading of M and N tiles into shared memory
        if ((Row < Width) && (ph*TILE_WIDTH+tx) < Width)
            Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        else 
            Mds[ty][tx] = 0.0f;
        
        if ((ph*TILE_WIDTH+ty) < Width && Col < Width)
            Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
        else 
            Nds[ty][tx] = 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    if (Row < Width && Col < Width)
        P[Row*Width + Col] = Pvalue;
}

void launchTiledBasic(const float* h_M, const float* h_N, float* h_P, int Width, float* outTime, int numRuns = 1000) {
    const size_t bytes = static_cast<size_t>(Width) * static_cast<size_t>(Width) * sizeof(float);

    float *d_M = nullptr, *d_N = nullptr, *d_P = nullptr;
    checkCuda(cudaMalloc(&d_M, bytes), "cudaMalloc d_M");
    checkCuda(cudaMalloc(&d_N, bytes), "cudaMalloc d_N");
    checkCuda(cudaMalloc(&d_P, bytes), "cudaMalloc d_P");

    checkCuda(cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice), "memcpy M H2D");
    checkCuda(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice), "memcpy N H2D");

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(Width/TILE_WIDTH, Width/TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm-up run
    matrixMulKernel<<<grid, block>>>(d_M, d_N, d_P, Width);
    cudaDeviceSynchronize();
    
    // Timed runs
    float totalMs = 0.0f;
    for (int run = 0; run < numRuns; ++run) {
        cudaEventRecord(start);
        matrixMulKernel<<<grid, block>>>(d_M, d_N, d_P, Width);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        totalMs += ms;
    }
    
    checkCuda(cudaGetLastError(), "Kernel launch");
    
    if (outTime) *outTime = totalMs / numRuns;

    checkCuda(cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost), "memcpy P D2H");

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void launchTiledBoundary(const float* h_M, const float* h_N, float* h_P, int Width, float* outTime, int numRuns = 1000) {
    const size_t bytes = static_cast<size_t>(Width) * static_cast<size_t>(Width) * sizeof(float);

    float *d_M = nullptr, *d_N = nullptr, *d_P = nullptr;
    checkCuda(cudaMalloc(&d_M, bytes), "cudaMalloc d_M");
    checkCuda(cudaMalloc(&d_N, bytes), "cudaMalloc d_N");
    checkCuda(cudaMalloc(&d_P, bytes), "cudaMalloc d_P");

    checkCuda(cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice), "memcpy M H2D");
    checkCuda(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice), "memcpy N H2D");

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(cdiv(Width, TILE_WIDTH), cdiv(Width, TILE_WIDTH));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm-up run
    matrixMulKernelBoundary<<<grid, block>>>(d_M, d_N, d_P, Width);
    cudaDeviceSynchronize();
    
    // Timed runs
    float totalMs = 0.0f;
    for (int run = 0; run < numRuns; ++run) {
        cudaEventRecord(start);
        matrixMulKernelBoundary<<<grid, block>>>(d_M, d_N, d_P, Width);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        totalMs += ms;
    }
    
    checkCuda(cudaGetLastError(), "Kernel launch");
    
    if (outTime) *outTime = totalMs / numRuns;

    checkCuda(cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost), "memcpy P D2H");

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

bool validateWithCublas(cublasHandle_t handle,
                        const float* h_M, const float* h_N, const float* h_P,
                        int Width, float* outMaxAbsDiff, float* outTime) {
    const size_t bytes = static_cast<size_t>(Width) * static_cast<size_t>(Width) * sizeof(float);

    float *d_M = nullptr, *d_N = nullptr, *d_P = nullptr;
    checkCuda(cudaMalloc(&d_M, bytes), "cudaMalloc d_M (cublas)");
    checkCuda(cudaMalloc(&d_N, bytes), "cudaMalloc d_N (cublas)");
    checkCuda(cudaMalloc(&d_P, bytes), "cudaMalloc d_P (cublas)");

    // Transpose M and N from row-major to column-major for cuBLAS
    std::vector<float> h_M_colmajor(Width * Width);
    std::vector<float> h_N_colmajor(Width * Width);
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            h_M_colmajor[col * Width + row] = h_M[row * Width + col];
            h_N_colmajor[col * Width + row] = h_N[row * Width + col];
        }
    }

    checkCuda(cudaMemcpy(d_M, h_M_colmajor.data(), bytes, cudaMemcpyHostToDevice), "H2D M (cublas)");
    checkCuda(cudaMemcpy(d_N, h_N_colmajor.data(), bytes, cudaMemcpyHostToDevice), "H2D N (cublas)");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Warm-up SGEMM (not timed)
    checkCublas(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    Width, Width, Width,
                    &alpha,
                    d_M, Width,
                    d_N, Width,
                    &beta,
                    d_P, Width),
        "cublasSgemm warmup"
    );

    // Timed SGEMM - multiple runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int numRuns = 1000;
    float totalMs = 0.0f;
    for (int run = 0; run < numRuns; ++run) {
        cudaEventRecord(start);
        checkCublas(
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        Width, Width, Width,
                        &alpha,
                        d_M, Width,
                        d_N, Width,
                        &beta,
                        d_P, Width),
            "cublasSgemm timed"
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        totalMs += ms;
    }
    
    if (outTime) *outTime = totalMs / numRuns;

    std::vector<float> h_P_ref(Width * Width);
    checkCuda(cudaMemcpy(h_P_ref.data(), d_P, bytes, cudaMemcpyDeviceToHost), "D2H P (cublas)");

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Transpose result back from column-major to row-major for comparison
    std::vector<float> h_P_ref_rowmajor(Width * Width);
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            h_P_ref_rowmajor[row * Width + col] = h_P_ref[col * Width + row];
        }
    }

    float maxAbsDiff = 0.0f;
    for (int i = 0; i < Width * Width; ++i) {
        float diff = std::fabs(h_P_ref_rowmajor[i] - h_P[i]);
        if (diff > maxAbsDiff) maxAbsDiff = diff;
    }
    if (outMaxAbsDiff) *outMaxAbsDiff = maxAbsDiff;

    const float tol = 1e-3f;
    return maxAbsDiff < tol;
}

int main(int argc, char** argv) {
    int Width = 512; // default
    if (argc >= 2) {
        Width = std::atoi(argv[1]);
        if (Width <= 0) Width = 512;
    }

    const int numRuns = 1000;
    std::cout << "Tiled Matrix Multiplication: " << Width << "x" << Width << std::endl;
    std::cout << "Tile size: " << TILE_WIDTH << "x" << TILE_WIDTH << std::endl;
    std::cout << "Benchmark runs: " << numRuns << " (average reported)" << std::endl << std::endl;

    const size_t elems = static_cast<size_t>(Width) * static_cast<size_t>(Width);

    std::vector<float> M(elems), N(elems), P_basic(elems), P_boundary(elems);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (size_t i = 0; i < elems; ++i) {
        M[i] = dist(rng);
        N[i] = dist(rng);
    }

    // Create cuBLAS handle once
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate");

    // Run basic tiled kernel (only works if Width is multiple of TILE_WIDTH)
    float timeBasic = 0.0f;
    if (Width % TILE_WIDTH == 0) {
        launchTiledBasic(M.data(), N.data(), P_basic.data(), Width, &timeBasic, numRuns);
        std::cout << "Basic tiled kernel time: " << timeBasic << " ms (avg of " << numRuns << " runs)" << std::endl;

        float maxDiffBasic = 0.0f;
        bool okBasic = validateWithCublas(handle, M.data(), N.data(), P_basic.data(), Width, &maxDiffBasic, nullptr);
        std::cout << "Basic tiled validation: " << (okBasic ? "OK" : "FAIL") 
                  << ", maxAbsDiff=" << maxDiffBasic << std::endl << std::endl;
    } else {
        std::cout << "Skipping basic tiled kernel (Width not multiple of TILE_WIDTH)" << std::endl << std::endl;
    }

    // Run boundary-checked tiled kernel (works for any Width)
    float timeBoundary = 0.0f;
    launchTiledBoundary(M.data(), N.data(), P_boundary.data(), Width, &timeBoundary, numRuns);
    std::cout << "Boundary-checked tiled kernel time: " << timeBoundary << " ms (avg of " << numRuns << " runs)" << std::endl;

    float maxDiffBoundary = 0.0f;
    bool okBoundary = validateWithCublas(handle, M.data(), N.data(), P_boundary.data(), Width, &maxDiffBoundary, nullptr);
    std::cout << "Boundary-checked validation: " << (okBoundary ? "OK" : "FAIL") 
              << ", maxAbsDiff=" << maxDiffBoundary << std::endl << std::endl;

    // cuBLAS benchmark
    float timeCublas = 0.0f;
    validateWithCublas(handle, M.data(), N.data(), P_boundary.data(), Width, nullptr, &timeCublas);
    std::cout << "cuBLAS SGEMM time: " << timeCublas << " ms (avg of " << numRuns << " runs)" << std::endl << std::endl;

    // Summary
    std::cout << "========== Performance Summary ==========" << std::endl;
    if (Width % TILE_WIDTH == 0) {
        std::cout << "Basic tiled:       " << timeBasic << " ms (cuBLAS is " 
                  << (timeBasic/timeCublas) << "x faster)" << std::endl;
    }
    std::cout << "Boundary-checked:  " << timeBoundary << " ms (cuBLAS is " 
              << (timeBoundary/timeCublas) << "x faster)" << std::endl;
    std::cout << "cuBLAS:            " << timeCublas << " ms (baseline)" << std::endl;

    cublasDestroy(handle);
    return (okBoundary) ? 0 : 1;
}
