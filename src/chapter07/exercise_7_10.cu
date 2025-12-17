// 3D tiled convolution using constant memory for the filter + CPU validation
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Configure tile and filter limits
constexpr int IN_TILE_DIM = 10;           // tile size including halo (fits in shared mem)
constexpr int MAX_RADIUS = 2;             // supports up to 5x5x5 filter
constexpr int OUT_TILE_DIM = IN_TILE_DIM - 2 * MAX_RADIUS;
constexpr int MAX_K = 2 * MAX_RADIUS + 1;
constexpr int MAX_KERNEL_ELEMS = MAX_K * MAX_K * MAX_K;

__constant__ float d_F_const[MAX_KERNEL_ELEMS];

__global__ void convolution_tiled_3D_const_kernel(
    const float* __restrict__ N,
    float* __restrict__ P,
    int r,
    int width, int height, int depth)
{
    // Shared tile with halo
    __shared__ float tile[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    // Global coordinates this thread loads
    int inX = blockIdx.x * OUT_TILE_DIM + threadIdx.x - r;
    int inY = blockIdx.y * OUT_TILE_DIM + threadIdx.y - r;
    int inZ = blockIdx.z * OUT_TILE_DIM + threadIdx.z - r;

    // Load input into shared tile with bounds check
    float val = 0.0f;
    if (inX >= 0 && inX < width && inY >= 0 && inY < height && inZ >= 0 && inZ < depth) {
        size_t idx = (size_t(inZ) * height + inY) * width + inX;
        val = N[idx];
    }
    tile[threadIdx.z][threadIdx.y][threadIdx.x] = val;

    __syncthreads();

    // Only threads whose output lies inside the valid output tile compute
    int outX = blockIdx.x * OUT_TILE_DIM + (threadIdx.x - r);
    int outY = blockIdx.y * OUT_TILE_DIM + (threadIdx.y - r);
    int outZ = blockIdx.z * OUT_TILE_DIM + (threadIdx.z - r);

    if (threadIdx.x >= r && threadIdx.x < IN_TILE_DIM - r &&
        threadIdx.y >= r && threadIdx.y < IN_TILE_DIM - r &&
        threadIdx.z >= r && threadIdx.z < IN_TILE_DIM - r &&
        outX < width && outY < height && outZ < depth) {

        int K = 2 * r + 1;
        float sum = 0.0f;
        for (int fZ = 0; fZ < K; ++fZ) {
            for (int fY = 0; fY < K; ++fY) {
                for (int fX = 0; fX < K; ++fX) {
                    int tX = threadIdx.x + fX - r;
                    int tY = threadIdx.y + fY - r;
                    int tZ = threadIdx.z + fZ - r;
                    int fIdx = (fZ * K + fY) * K + fX;
                    sum += d_F_const[fIdx] * tile[tZ][tY][tX];
                }
            }
        }

        size_t outIdx = (size_t(outZ) * height + outY) * width + outX;
        P[outIdx] = sum;
    }
}

// CPU reference for validation
static void convolution_3d_cpu(
    const float* N,
    const float* F,
    float* P,
    int r,
    int width, int height, int depth)
{
    int K = 2 * r + 1;
    for (int outZ = 0; outZ < depth; ++outZ) {
        for (int outY = 0; outY < height; ++outY) {
            for (int outX = 0; outX < width; ++outX) {
                float sum = 0.0f;
                for (int fZ = 0; fZ < K; ++fZ) {
                    int inZ = outZ - r + fZ;
                    for (int fY = 0; fY < K; ++fY) {
                        int inY = outY - r + fY;
                        for (int fX = 0; fX < K; ++fX) {
                            int inX = outX - r + fX;
                            if (inX >= 0 && inX < width &&
                                inY >= 0 && inY < height &&
                                inZ >= 0 && inZ < depth)
                            {
                                int nIdx = (inZ * height + inY) * width + inX;
                                int fIdx = (fZ * K + fY) * K + fX;
                                sum += F[fIdx] * N[nIdx];
                            }
                        }
                    }
                }
                int pIdx = (outZ * height + outY) * width + outX;
                P[pIdx] = sum;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int width = 64;
    int height = 64;
    int depth = 64;
    int r = 1; // 3x3x3 by default

    if (argc >= 2) width = atoi(argv[1]);
    if (argc >= 3) height = atoi(argv[2]);
    if (argc >= 4) depth = atoi(argv[3]);
    if (argc >= 5) r = atoi(argv[4]);

    if (r > MAX_RADIUS) {
        std::cerr << "Radius exceeds MAX_RADIUS=" << MAX_RADIUS << " (supported up to 5x5x5)" << std::endl;
        return 1;
    }

    int K = 2 * r + 1;
    size_t num_elements = size_t(width) * height * depth;
    size_t kernel_elements = size_t(K) * K * K;
    size_t size_bytes = num_elements * sizeof(float);
    size_t kernel_bytes = kernel_elements * sizeof(float);

    std::cout << "3D Tiled Convolution (const memory): " << width << "x" << height << "x" << depth
              << "  K=" << K << "  r=" << r << std::endl;

    std::vector<float> h_N(num_elements);
    std::vector<float> h_F(kernel_elements);
    std::vector<float> h_out_gpu(num_elements);
    std::vector<float> h_out_cpu(num_elements);

    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < num_elements; ++i) h_N[i] = dist(gen);
    for (size_t i = 0; i < kernel_elements; ++i) h_F[i] = dist(gen);

    float *d_N = nullptr, *d_out = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_N, size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, size_bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_N, h_N.data(), size_bytes, cudaMemcpyHostToDevice));

    // Copy filter to constant memory
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_F_const, h_F.data(), kernel_bytes, 0, cudaMemcpyHostToDevice));

    dim3 block(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    dim3 grid(
        (width + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
        (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
        (depth + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    // GPU computation
    convolution_tiled_3D_const_kernel<<<grid, block>>>(d_N, d_out, r, width, height, depth);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_out_gpu.data(), d_out, size_bytes, cudaMemcpyDeviceToHost));

    // CPU computation (validation)
    convolution_3d_cpu(h_N.data(), h_F.data(), h_out_cpu.data(), r, width, height, depth);

    // Compare results
    float max_diff = 0.0f, mae = 0.0f;
    for (size_t i = 0; i < num_elements; ++i) {
        float diff = std::abs(h_out_gpu[i] - h_out_cpu[i]);
        max_diff = std::max(max_diff, diff);
        mae += diff;
    }
    mae /= float(num_elements);

    std::cout << "GPU vs CPU: max_diff=" << max_diff << ", mae=" << mae << std::endl;
    std::cout << ((max_diff < 1e-5f) ? "Test PASSED" : "Test FAILED") << std::endl;

    cudaFree(d_N);
    cudaFree(d_out);
    return 0;
}
