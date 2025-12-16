#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

constexpr int TILE_SIZE = 32;

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

// A: row-major, B: column-major, C: row-major
// C = A * B
template<int TILE>
__global__ void matmul_tiled_no_corner_turning(
    const float* __restrict__ A,
    const float* __restrict__ B_colmajor,
    float* __restrict__ C,
    int width)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];  // store as [k][n] logically, but loaded unfavorably

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    // Loop over tiles along K dimension
    for (int ph = 0; ph < (width + TILE - 1) / TILE; ++ph)
    {
        const int a_col = ph * TILE + threadIdx.x; // K index
        const int b_row = ph * TILE + threadIdx.y; // K index (row in B)

        // Load A tile (row-major) => coalesced for threads with varying threadIdx.x
        As[threadIdx.y][threadIdx.x] =
            (row < width && a_col < width) ? A[row * width + a_col] : 0.0f;

        // Load B tile (column-major) in a "naive" way
        // B_colmajor element B(b_row, col) is stored at B_colmajor[col*width + b_row]
        // With threadIdx.x varying in a warp, addresses jump by width => uncoalesced.
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < width && col < width) ? B_colmajor[col * width + b_row] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
        {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < width && col < width)
        C[row * width + col] = acc;
}

// A: row-major, B: column-major, C: row-major
// Corner turning: make global loads for B coalesced by swapping thread roles
template<int TILE>
__global__ void matmul_tiled_corner_turning(
    const float* __restrict__ A,
    const float* __restrict__ B_colmajor,
    float* __restrict__ C,
    int width)
{
    __shared__ float As[TILE][TILE];
    // Note: we store B tile "turned" so that compute phase reads Bs[k][tx] normally
    __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int ph = 0; ph < (width + TILE - 1) / TILE; ++ph)
    {
        const int a_col = ph * TILE + threadIdx.x; // K

        // A tile load (same as before, coalesced)
        As[threadIdx.y][threadIdx.x] =
            (row < width && a_col < width) ? A[row * width + a_col] : 0.0f;

        // ----- Corner turning for B tile load -----
        // We want B loads to be coalesced in column-major layout:
        // column-major contiguous along "row" (b_row), i.e., B_colmajor[col*width + b_row].
        //
        // So let consecutive threads vary b_row, while keeping col fixed per group.
        // That corresponds to swapping roles of (tx, ty) when forming the linear index.
        //
        // We load element B(b_row, b_col) where:
        //   b_col = blockIdx.x*TILE + threadIdx.y   (fixed across tx)
        //   b_row = ph*TILE       + threadIdx.x     (varies across tx => contiguous in memory)
        //
        // And we store it into shared as Bs[tx][ty] so compute reads Bs[k][tx] naturally.
        const int b_col = blockIdx.x * TILE + threadIdx.y;
        const int b_row = ph * TILE + threadIdx.x;

        Bs[threadIdx.x][threadIdx.y] =
            (b_row < width && b_col < width) ? B_colmajor[b_col * width + b_row] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
        {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < width && col < width)
        C[row * width + col] = acc;
}

void launchNoCornerTurning(const float* h_A_rowmajor, const float* h_B_colmajor,
                           float* h_C, int width) {
    const size_t bytes = static_cast<size_t>(width) * width * sizeof(float);
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
    checkCuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");
    checkCuda(cudaMalloc(&d_C, bytes), "cudaMalloc d_C");

    checkCuda(cudaMemcpy(d_A, h_A_rowmajor, bytes, cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(d_B, h_B_colmajor, bytes, cudaMemcpyHostToDevice), "H2D B");

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE, (width + TILE_SIZE - 1) / TILE_SIZE);

    matmul_tiled_no_corner_turning<TILE_SIZE><<<grid, block>>>(d_A, d_B, d_C, width);
    checkCuda(cudaGetLastError(), "no corner turning launch");
    checkCuda(cudaDeviceSynchronize(), "no corner turning sync");

    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "D2H C");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void launchCornerTurning(const float* h_A_rowmajor, const float* h_B_colmajor,
                         float* h_C, int width) {
    const size_t bytes = static_cast<size_t>(width) * width * sizeof(float);
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
    checkCuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");
    checkCuda(cudaMalloc(&d_C, bytes), "cudaMalloc d_C");

    checkCuda(cudaMemcpy(d_A, h_A_rowmajor, bytes, cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(d_B, h_B_colmajor, bytes, cudaMemcpyHostToDevice), "H2D B");

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE, (width + TILE_SIZE - 1) / TILE_SIZE);

    matmul_tiled_corner_turning<TILE_SIZE><<<grid, block>>>(d_A, d_B, d_C, width);
    checkCuda(cudaGetLastError(), "corner turning launch");
    checkCuda(cudaDeviceSynchronize(), "corner turning sync");

    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "D2H C");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

bool validateWithCublas(cublasHandle_t handle,
                        const float* h_A_rowmajor, const float* h_B_colmajor,
                        const float* h_C, int width, float* outMaxDiff) {
    const size_t bytes = static_cast<size_t>(width) * width * sizeof(float);
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A (cublas)");
    checkCuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B (cublas)");
    checkCuda(cudaMalloc(&d_C, bytes), "cudaMalloc d_C (cublas)");

    // A is row-major, need to transpose to column-major for cuBLAS
    std::vector<float> h_A_col(width * width);
    for (int r = 0; r < width; ++r) {
        for (int c = 0; c < width; ++c) {
            h_A_col[c * width + r] = h_A_rowmajor[r * width + c];
        }
    }

    checkCuda(cudaMemcpy(d_A, h_A_col.data(), bytes, cudaMemcpyHostToDevice), "H2D A (cublas)");
    checkCuda(cudaMemcpy(d_B, h_B_colmajor, bytes, cudaMemcpyHostToDevice), "H2D B (cublas)");

    const float alpha = 1.0f, beta = 0.0f;
    // C = alpha * A * B + beta * C
    // Both A and B are now column-major in device memory
    checkCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            width, width, width,
                            &alpha, d_A, width, d_B, width,
                            &beta, d_C, width), "cublasSgemm");

    // Get result in column-major, convert to row-major
    std::vector<float> h_C_col(width * width);
    checkCuda(cudaMemcpy(h_C_col.data(), d_C, bytes, cudaMemcpyDeviceToHost), "D2H C (cublas)");

    std::vector<float> h_C_ref(width * width);
    for (int r = 0; r < width; ++r) {
        for (int c = 0; c < width; ++c) {
            h_C_ref[r * width + c] = h_C_col[c * width + r];
        }
    }

    float maxDiff = 0.0f;
    for (int i = 0; i < width * width; ++i) {
        float diff = std::fabs(h_C_ref[i] - h_C[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    if (outMaxDiff) *outMaxDiff = maxDiff;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    const float tol = 1e-3f;
    return maxDiff < tol;
}

int main(int argc, char** argv) {
    int width = 512;
    if (argc >= 2) {
        width = std::atoi(argv[1]);
        if (width <= 0) width = 512;
    }

    std::cout << "Corner Turning Matrix Multiplication Test: " << width << "x" << width << std::endl;
    std::cout << "Tile size: " << TILE_SIZE << std::endl << std::endl;

    const size_t elems = static_cast<size_t>(width) * width;

    // A in row-major, B in column-major, C in row-major
    std::vector<float> A_rowmajor(elems);
    std::vector<float> B_colmajor(elems);
    std::vector<float> C_no_corner(elems);
    std::vector<float> C_corner(elems);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < elems; ++i) {
        A_rowmajor[i] = dist(rng);
        B_colmajor[i] = dist(rng);
    }

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate");

    // Test no corner turning
    std::cout << "Testing no corner turning kernel..." << std::endl;
    launchNoCornerTurning(A_rowmajor.data(), B_colmajor.data(), C_no_corner.data(), width);
    float maxDiff1 = 0.0f;
    bool ok1 = validateWithCublas(handle, A_rowmajor.data(), B_colmajor.data(),
                                   C_no_corner.data(), width, &maxDiff1);
    std::cout << "  Result: " << (ok1 ? "OK" : "FAIL") << ", maxAbsDiff=" << maxDiff1 << std::endl << std::endl;

    // Test corner turning
    std::cout << "Testing corner turning kernel..." << std::endl;
    launchCornerTurning(A_rowmajor.data(), B_colmajor.data(), C_corner.data(), width);
    float maxDiff2 = 0.0f;
    bool ok2 = validateWithCublas(handle, A_rowmajor.data(), B_colmajor.data(),
                                   C_corner.data(), width, &maxDiff2);
    std::cout << "  Result: " << (ok2 ? "OK" : "FAIL") << ", maxAbsDiff=" << maxDiff2 << std::endl << std::endl;

    cublasDestroy(handle);

    if (ok1 && ok2) {
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests failed!" << std::endl;
        return 1;
    }
}
