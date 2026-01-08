// Kogge–Stone scan (inclusive) — block-wise only
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef SECTION_SIZE
#define SECTION_SIZE 256
#endif

__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, unsigned int N){
	__shared__ float XY[SECTION_SIZE];
	const unsigned int i  = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
		XY[threadIdx.x] = X[i];
	} else {
        XY[threadIdx.x] = 0.0f;
    }

	for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
        __syncthreads();
		float temp = 0.0f;
		if (threadIdx.x >= stride) 
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
		__syncthreads();
		if (threadIdx.x >= stride)
		    XY[threadIdx.x] = temp;
	}
	if (i < N) {
		Y[i] = XY[threadIdx.x];
	}
}

// exclusive scan
__global__ void Kogge_Stone_exclusive_scan_kernel_ex_11_6(float *X, float *Y, unsigned int N){
	__shared__ float XY[SECTION_SIZE];
	const unsigned int i  = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (threadIdx.x == 0) XY[0] = 0;
        else XY[threadIdx.x] = X[i - 1];
	} else {
        XY[threadIdx.x] = 0.0f;
    }

	for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
        __syncthreads();
		float temp = 0.0f;
		if (threadIdx.x >= stride) 
        {
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
		__syncthreads();
		if (threadIdx.x >= stride)
		    XY[threadIdx.x] = temp;
	}

    if (i < N) {
		Y[i] = XY[threadIdx.x];
	}

}

__global__ void Brent_Kung_scan_kernel(float *X, float *Y, unsigned int N) {
	__shared__ float XY[SECTION_SIZE];
	unsigned int i = 2*blockIdx.x*blockDim.x + threadIdx.x;
	if(i < N) XY[threadIdx.x] = X[i];
	if(i + blockDim.x < N) XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
	for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
		__syncthreads();
		unsigned int index = (threadIdx.x + 1)*2*stride - 1;
		if(index < SECTION_SIZE) {
			XY[index] += XY[index - stride];
		}
	}
	for (int stride = SECTION_SIZE/4; stride > 0; stride /= 2) {
		__syncthreads();
		unsigned int index = (threadIdx.x + 1)*stride*2 - 1;
		if(index + stride < SECTION_SIZE) {
			XY[index + stride] += XY[index];
		}
	}
	__syncthreads();
	if (i < N) Y[i] = XY[threadIdx.x];
	if (i + blockDim.x < N) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}

__global__ void Kogge_Stone_scan_kernel_ex_11_2(const float *X, float *Y, unsigned int N)
{
    __shared__ float A[SECTION_SIZE];
    __shared__ float B[SECTION_SIZE];

    const unsigned int tid = threadIdx.x;
    const unsigned int i   = blockIdx.x * blockDim.x + tid;

    // Load (block-wise; out-of-range -> 0)
    float v = (i < N) ? X[i] : 0.0f;
    A[tid] = v;
    B[tid] = v;
    __syncthreads();

    float* src = A;
    float* dst = B;

    // Kogge–Stone inclusive scan inside one block
    for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1)
    {
        // Everyone writes dst[tid] (no "holes")
        float out = src[tid];
        if (tid >= stride) out = src[tid] + src[tid - stride];
        dst[tid] = out;

        __syncthreads();          // ensure dst fully written before next round reads it

        // swap roles (ping-pong)
        float* tmp = src;
        src = dst;
        dst = tmp;
    }

    if (i < N) Y[i] = src[tid];   // src holds the latest results
}


// CPU reference: block-wise inclusive scan (each block starts from zero)
static void cpu_blockwise_inclusive_scan(const float* X, float* Y, unsigned int N, unsigned int blockSize){
	for (unsigned int base = 0; base < N; base += blockSize) {
		float acc = 0.0f;
		unsigned int end = (base + blockSize < N) ? (base + blockSize) : N;
		for (unsigned int i = base; i < end; ++i) {
			acc += X[i];
			Y[i] = acc;
		}
	}
}

static void cpu_blockwise_exclusive_scan(const float* X, float* Y, unsigned int N, unsigned int blockSize){
	for (unsigned int base = 0; base < N; base += blockSize) {
		float acc = 0.0f;
		unsigned int end = (base + blockSize < N) ? (base + blockSize) : N;
		for (unsigned int i = base; i < end; ++i) {			
			Y[i] = acc;
            acc += X[i];
		}
	}
}

// Typedef for kernel launcher function
typedef void(*KernelLauncher)(float* dX, float* dY, unsigned int N, unsigned int gridSize, unsigned int blockSize);

// Launcher wrappers for each kernel
static void launch_kogge_stone(float* dX, float* dY, unsigned int N, unsigned int gridSize, unsigned int blockSize) {
	Kogge_Stone_scan_kernel<<<gridSize, blockSize>>>(dX, dY, N);
}

static void launch_brent_kung(float* dX, float* dY, unsigned int N, unsigned int gridSize, unsigned int blockSize) {
	Brent_Kung_scan_kernel<<<gridSize, blockSize>>>(dX, dY, N);
}

static void launch_kogge_stone_ex_11_2(float* dX, float* dY, unsigned int N, unsigned int gridSize, unsigned int blockSize) {
	Kogge_Stone_scan_kernel_ex_11_2<<<gridSize, blockSize>>>(dX, dY, N);
}

static void launch_kogge_stone_exclusive_11_6(float* dX, float* dY, unsigned int N, unsigned int gridSize, unsigned int blockSize) {
	Kogge_Stone_exclusive_scan_kernel_ex_11_6<<<gridSize, blockSize>>>(dX, dY, N);
}

// Generic scan test function
static bool run_generic_scan_test(const char* kernel_name, const char* title, unsigned int N, 
                                  unsigned int blockSize, unsigned int elementsPerBlock, KernelLauncher launcher) {
	const unsigned int gridSize = (N + elementsPerBlock - 1) / elementsPerBlock;
	size_t bytes = N * sizeof(float);
	float* hX = (float*)malloc(bytes);
	float* hY = (float*)malloc(bytes);
	float* hRef = (float*)malloc(bytes);

	for (int i = 0; i < N; ++i) {
		hX[i] = 0.1f * ((i % 7) - 3);
	}

	cpu_blockwise_inclusive_scan(hX, hRef, N, elementsPerBlock);

	float *dX = nullptr, *dY = nullptr;
	cudaMalloc(&dX, bytes);
	cudaMalloc(&dY, bytes);
	cudaMemcpy(dX, hX, bytes, cudaMemcpyHostToDevice);

	launcher(dX, dY, N, gridSize, blockSize);
	cudaDeviceSynchronize();

	cudaMemcpy(hY, dY, bytes, cudaMemcpyDeviceToHost);

	bool ok = true;
	for (unsigned int i = 0; i < N; ++i) {
		if (fabs(hY[i] - hRef[i]) > 1e-4f) { ok = false; break; }
	}

	printf("%s Test (%s, N=%u) -> %s\n", kernel_name, title, N, ok ? "PASS" : "FAIL");
    
	cudaFree(dX); cudaFree(dY);
	free(hX); free(hY); free(hRef);
	return ok;
}


static bool run_exclusive_scan_test(const char* kernel_name, const char* title, unsigned int N, 
                                  unsigned int blockSize, unsigned int elementsPerBlock, KernelLauncher launcher) {
	const unsigned int gridSize = (N + elementsPerBlock - 1) / elementsPerBlock;
	size_t bytes = N * sizeof(float);
	float* hX = (float*)malloc(bytes);
	float* hY = (float*)malloc(bytes);
	float* hRef = (float*)malloc(bytes);

	for (int i = 0; i < N; ++i) {
		hX[i] = 0.1f * ((i % 7) - 3);
	}

	cpu_blockwise_exclusive_scan(hX, hRef, N, elementsPerBlock);

	float *dX = nullptr, *dY = nullptr;
	cudaMalloc(&dX, bytes);
	cudaMalloc(&dY, bytes);
	cudaMemcpy(dX, hX, bytes, cudaMemcpyHostToDevice);

	launcher(dX, dY, N, gridSize, blockSize);
	cudaDeviceSynchronize();

	cudaMemcpy(hY, dY, bytes, cudaMemcpyDeviceToHost);

	bool ok = true;
	for (int i = 0; i < N; ++i) {
		if (fabs(hY[i] - hRef[i]) > 1e-4f) { ok = false; break; }
	}

	printf("%s Test (%s, N=%u) -> %s\n", kernel_name, title, N, ok ? "PASS" : "FAIL");
    
	cudaFree(dX); cudaFree(dY);
	free(hX); free(hY); free(hRef);
	return ok;
}

int main(){
	// Kogge-Stone tests
	printf("========== Kogge-Stone Scan Tests ==========\n");
	run_generic_scan_test("Kogge-Stone", "tiny", 1, SECTION_SIZE, SECTION_SIZE, launch_kogge_stone);
	run_generic_scan_test("Kogge-Stone", "small", 7, SECTION_SIZE, SECTION_SIZE, launch_kogge_stone);
	run_generic_scan_test("Kogge-Stone", "one-less-than-block", SECTION_SIZE - 1, SECTION_SIZE, SECTION_SIZE, launch_kogge_stone);
	run_generic_scan_test("Kogge-Stone", "exactly-one-block", SECTION_SIZE, SECTION_SIZE, SECTION_SIZE, launch_kogge_stone);
	run_generic_scan_test("Kogge-Stone", "one-more-than-block", SECTION_SIZE + 1, SECTION_SIZE, SECTION_SIZE, launch_kogge_stone);
	run_generic_scan_test("Kogge-Stone", "multi-block", 1000, SECTION_SIZE, SECTION_SIZE, launch_kogge_stone);
	run_generic_scan_test("Kogge-Stone", "odd multi-block", 2049, SECTION_SIZE, SECTION_SIZE, launch_kogge_stone);
	
	// Brent-Kung tests
	printf("\n========== Brent-Kung Scan Tests ==========\n");
	unsigned int bk_blockSize = SECTION_SIZE / 2;
	unsigned int bk_elementsPerBlock = 2 * bk_blockSize;
	run_generic_scan_test("Brent-Kung", "tiny", 1, bk_blockSize, bk_elementsPerBlock, launch_brent_kung);
	run_generic_scan_test("Brent-Kung", "small", 7, bk_blockSize, bk_elementsPerBlock, launch_brent_kung);
	run_generic_scan_test("Brent-Kung", "one-less-than-section", SECTION_SIZE - 1, bk_blockSize, bk_elementsPerBlock, launch_brent_kung);
	run_generic_scan_test("Brent-Kung", "exactly-one-section", SECTION_SIZE, bk_blockSize, bk_elementsPerBlock, launch_brent_kung);
	run_generic_scan_test("Brent-Kung", "one-more-than-section", SECTION_SIZE + 1, bk_blockSize, bk_elementsPerBlock, launch_brent_kung);
	run_generic_scan_test("Brent-Kung", "multi-block", 1000, bk_blockSize, bk_elementsPerBlock, launch_brent_kung);
	run_generic_scan_test("Brent-Kung", "odd multi-block", 2049, bk_blockSize, bk_elementsPerBlock, launch_brent_kung);

	// Kogge-Stone ex 11.2 (double-buffer variant) tests
	printf("\n========== Kogge-Stone ex11.2 Scan Tests ==========\n");
	unsigned int ex112_blockSize = SECTION_SIZE;
	unsigned int ex112_elementsPerBlock = SECTION_SIZE; // one element per thread, block-wide
	run_generic_scan_test("Kogge-Stone-ex11.2", "tiny", 1, ex112_blockSize, ex112_elementsPerBlock, launch_kogge_stone_ex_11_2);
	run_generic_scan_test("Kogge-Stone-ex11.2", "small", 7, ex112_blockSize, ex112_elementsPerBlock, launch_kogge_stone_ex_11_2);
	run_generic_scan_test("Kogge-Stone-ex11.2", "one-less-than-block", SECTION_SIZE - 1, ex112_blockSize, ex112_elementsPerBlock, launch_kogge_stone_ex_11_2);
	run_generic_scan_test("Kogge-Stone-ex11.2", "exactly-one-block", SECTION_SIZE, ex112_blockSize, ex112_elementsPerBlock, launch_kogge_stone_ex_11_2);
	run_generic_scan_test("Kogge-Stone-ex11.2", "one-more-than-block", SECTION_SIZE + 1, ex112_blockSize, ex112_elementsPerBlock, launch_kogge_stone_ex_11_2);
	run_generic_scan_test("Kogge-Stone-ex11.2", "multi-block", 1000, ex112_blockSize, ex112_elementsPerBlock, launch_kogge_stone_ex_11_2);
	run_generic_scan_test("Kogge-Stone-ex11.2", "odd multi-block", 2049, ex112_blockSize, ex112_elementsPerBlock, launch_kogge_stone_ex_11_2);




	// Exclusive scan tests (ex 11.6)
	printf("\n========== Kogge-Stone ex11.6 Exclusive Scan Tests ==========/n");
	unsigned int ex116_blockSize = SECTION_SIZE;
	unsigned int ex116_elementsPerBlock = SECTION_SIZE;
	run_exclusive_scan_test("Kogge-Stone-ex11.6", "tiny", 1, ex116_blockSize, ex116_elementsPerBlock, launch_kogge_stone_exclusive_11_6);
	run_exclusive_scan_test("Kogge-Stone-ex11.6", "small", 7, ex116_blockSize, ex116_elementsPerBlock, launch_kogge_stone_exclusive_11_6);
	run_exclusive_scan_test("Kogge-Stone-ex11.6", "one-less-than-block", SECTION_SIZE - 1, ex116_blockSize, ex116_elementsPerBlock, launch_kogge_stone_exclusive_11_6);
	run_exclusive_scan_test("Kogge-Stone-ex11.6", "exactly-one-block", SECTION_SIZE, ex116_blockSize, ex116_elementsPerBlock, launch_kogge_stone_exclusive_11_6);
	run_exclusive_scan_test("Kogge-Stone-ex11.6", "one-more-than-block", SECTION_SIZE + 1, ex116_blockSize, ex116_elementsPerBlock, launch_kogge_stone_exclusive_11_6);
	run_exclusive_scan_test("Kogge-Stone-ex11.6", "multi-block", 1000, ex116_blockSize, ex116_elementsPerBlock, launch_kogge_stone_exclusive_11_6);
	run_exclusive_scan_test("Kogge-Stone-ex11.6", "odd multi-block", 2049, ex116_blockSize, ex116_elementsPerBlock, launch_kogge_stone_exclusive_11_6);
	return 0;
}

