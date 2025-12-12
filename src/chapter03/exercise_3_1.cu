#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "cdiv.h"

// __global__ kernel from the attachment (square matrices of size Width)
__global__ void MatrixMulKernelByRow(const float* M, const float* N, float* P, int Width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < Width) {
        for (int col = 0; col < Width; ++col) {
            float Pvalue = 0.0f;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = Pvalue;
        }
    }
}

// __global__ kernel from the attachment (square matrices of size Width)
__global__ void MatrixMulKernelByColumn(const float* M, const float* N, float* P, int Width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < Width) {
        for (int row = 0; row < Width; ++row) {
            float Pvalue = 0.0f;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = Pvalue;
        }
    }
}


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

void launchMatrixMulByRow(const float* h_A, const float* h_B, float* h_C, int Width) {
    const size_t elems = static_cast<size_t>(Width) * static_cast<size_t>(Width);
    const size_t bytes = elems * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
    checkCuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");
    checkCuda(cudaMalloc(&d_C, bytes), "cudaMalloc d_C");

    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "memcpy A H2D");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "memcpy B H2D");

    // Launch row-major kernel: 1D over rows, measure time
    cudaEvent_t startRow, stopRow;
    cudaEventCreate(&startRow);
    cudaEventCreate(&stopRow);
    cudaEventRecord(startRow);
    dim3 blockRow(256,1,1);
    dim3 gridRow(cdiv(Width, blockRow.x), 1,1);
    MatrixMulKernelByRow<<<gridRow, blockRow>>>(d_A, d_B, d_C, Width);
    checkCuda(cudaGetLastError(), "Kernel launch");
    cudaEventRecord(stopRow);
    cudaEventSynchronize(stopRow);
    float msRow = 0.0f;
    cudaEventElapsedTime(&msRow, startRow, stopRow);

    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "memcpy C D2H");
    std::cout << "Row kernel time: " << msRow << " ms" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void launchMatrixMulByColumn(const float* h_A, const float* h_B, float* h_C, int Width) {
    const size_t elems = static_cast<size_t>(Width) * static_cast<size_t>(Width);
    const size_t bytes = elems * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A (col)");
    checkCuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B (col)");
    checkCuda(cudaMalloc(&d_C, bytes), "cudaMalloc d_C (col)");

    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "memcpy A H2D (col)");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "memcpy B H2D (col)");

    // Launch column-major sweep: 1D over columns, measure time
    cudaEvent_t startCol, stopCol;
    cudaEventCreate(&startCol);
    cudaEventCreate(&stopCol);
    cudaEventRecord(startCol);
    dim3 blockCol(256,1,1);
    dim3 gridCol(cdiv(Width, blockCol.x), 1,1);
    MatrixMulKernelByColumn<<<gridCol, blockCol>>>(d_A, d_B, d_C, Width);
    checkCuda(cudaGetLastError(), "Kernel launch (col)");
    cudaEventRecord(stopCol);
    cudaEventSynchronize(stopCol);
    float msCol = 0.0f;
    cudaEventElapsedTime(&msCol, startCol, stopCol);

    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "memcpy C D2H (col)");
    std::cout << "Column kernel time: " << msCol << " ms" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Validate against cuBLAS SGEMM: C = A * B
bool validateWithCublas(cublasHandle_t handle,
                        const float* h_A, const float* h_B, const float* h_C,
                        int Width, float* outMaxAbsDiff) {
    const size_t elems = static_cast<size_t>(Width) * static_cast<size_t>(Width);
    const size_t bytes = elems * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A (cublas)");
    checkCuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B (cublas)");
    checkCuda(cudaMalloc(&d_C, bytes), "cudaMalloc d_C (cublas)");

    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "H2D A (cublas)");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "H2D B (cublas)");

    // Assume handle is created outside and reused; no timing penalty here.

    // cuBLAS uses column-major. Our host data is row-major.
    // To compute row-major C = A * B, we can compute column-major C^T = B_col * A_col
    // which maps to cublasSgemm with opN/opN and swapped operands (B then A).
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Warm-up GEMM (not timed): cuBLAS and CUDA often perform one-time
    // initializations on the first call (context setup, kernel load/JIT,
    // heuristic selection). Running an unmeasured SGEMM here ensures the
    // timed call below reflects steady-state performance instead of
    // including first-use overhead.
    checkCublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            Width, Width, Width,
            &alpha,
            d_B, Width,
            d_A, Width,
            &beta,
            d_C, Width
        ), "cublasSgemm warmup"
    );
    // Timed GEMM
    cudaEvent_t startBlas, stopBlas;
    cudaEventCreate(&startBlas);
    cudaEventCreate(&stopBlas);
    cudaEventRecord(startBlas);
    checkCublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            Width, Width, Width,
            &alpha,
            d_B, Width,
            d_A, Width,
            &beta,
            d_C, Width
        ), "cublasSgemm timed"
    );
    cudaEventRecord(stopBlas);
    cudaEventSynchronize(stopBlas);
    float msBlas = 0.0f;
    cudaEventElapsedTime(&msBlas, startBlas, stopBlas);

    std::vector<float> h_C_ref(elems);
    checkCuda(cudaMemcpy(h_C_ref.data(), d_C, bytes, cudaMemcpyDeviceToHost), "D2H C (cublas)");
    std::cout << "cuBLAS SGEMM time: " << msBlas << " ms" << std::endl;

    // Do not destroy handle here; managed by caller.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    float maxAbsDiff = 0.0f;
    for (size_t i = 0; i < elems; ++i) {
        float diff = std::fabs(h_C_ref[i] - h_C[i]);
        if (diff > maxAbsDiff) maxAbsDiff = diff;
    }
    if (outMaxAbsDiff) *outMaxAbsDiff = maxAbsDiff;

    const float tol = 1e-4f;
    return maxAbsDiff < tol;
}

int main(int argc, char** argv) {
    int Width = 256; // default square size
    if (argc >= 2) {
        Width = std::atoi(argv[1]);
        if (Width <= 0) Width = 256;
    }
    std::cout << "MatrixMul size: " << Width << "x" << Width << std::endl;

    const size_t elems = static_cast<size_t>(Width) * static_cast<size_t>(Width);

    std::vector<float> A(elems), B(elems), C(elems);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < elems; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    // Create cuBLAS handle once
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate(main)");

    // Test row-sweep kernel
    launchMatrixMulByRow(A.data(), B.data(), C.data(), Width);
    float maxDiffRow = 0.0f;
    bool okRow = validateWithCublas(handle, A.data(), B.data(), C.data(), Width, &maxDiffRow);
    std::cout << "Row kernel vs cuBLAS: " << (okRow ? "OK" : "FAIL") << ", maxAbsDiff=" << maxDiffRow << std::endl;

    // Test column-sweep kernel
    std::vector<float> Ccol(elems);
    launchMatrixMulByColumn(A.data(), B.data(), Ccol.data(), Width);
    float maxDiffCol = 0.0f;
    bool okCol = validateWithCublas(handle, A.data(), B.data(), Ccol.data(), Width, &maxDiffCol);
    std::cout << "Column kernel vs cuBLAS: " << (okCol ? "OK" : "FAIL") << ", maxAbsDiff=" << maxDiffCol << std::endl;

    bool ok = okRow && okCol;
    cublasDestroy(handle);

    return ok ? 0 : 1;
}
