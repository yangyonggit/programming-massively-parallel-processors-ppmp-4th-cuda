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

static bool run_scan_test(const char* title, unsigned int N){
	const unsigned int blockSize = SECTION_SIZE;
	const unsigned int gridSize  = (N + blockSize - 1) / blockSize;

	size_t bytes = N * sizeof(float);
	float* hX = (float*)malloc(bytes);
	float* hY = (float*)malloc(bytes);
	float* hRef = (float*)malloc(bytes);

	for (int i = 0; i < N; ++i) {
		hX[i] = 0.1f * ((i % 7) - 3); // mix positives/negatives
	}

	cpu_blockwise_inclusive_scan(hX, hRef, N, blockSize);

	float *dX = nullptr, *dY = nullptr;
	cudaMalloc(&dX, bytes);
	cudaMalloc(&dY, bytes);
	cudaMemcpy(dX, hX, bytes, cudaMemcpyHostToDevice);

	Kogge_Stone_scan_kernel<<<gridSize, blockSize>>>(dX, dY, N);
	cudaDeviceSynchronize();

	cudaMemcpy(hY, dY, bytes, cudaMemcpyDeviceToHost);

	// Validate
	bool ok = true;
	for (unsigned int i = 0; i < N; ++i) {
		if (fabs(hY[i] - hRef[i]) > 1e-4f) { ok = false; break; }
	}

	printf("Scan Test (%s, N=%u) -> %s\n", title, N, ok ? "PASS" : "FAIL");
    
	cudaFree(dX); cudaFree(dY);
	free(hX); free(hY); free(hRef);
	return ok;
}

static bool run_brent_kung_test(const char* title, unsigned int N){
	const unsigned int blockSize = SECTION_SIZE / 2; // Brent-Kung uses blockDim.x threads for 2*blockDim.x elements
	const unsigned int gridSize  = (N + (2*blockSize) - 1) / (2*blockSize);

	size_t bytes = N * sizeof(float);
	float* hX = (float*)malloc(bytes);
	float* hY = (float*)malloc(bytes);
	float* hRef = (float*)malloc(bytes);

	for (int i = 0; i < N; ++i) {
		hX[i] = 0.1f * ((i % 7) - 3);
	}

	cpu_blockwise_inclusive_scan(hX, hRef, N, 2*blockSize);

	float *dX = nullptr, *dY = nullptr;
	cudaMalloc(&dX, bytes);
	cudaMalloc(&dY, bytes);
	cudaMemcpy(dX, hX, bytes, cudaMemcpyHostToDevice);

	Brent_Kung_scan_kernel<<<gridSize, blockSize>>>(dX, dY, N);
	cudaDeviceSynchronize();

	cudaMemcpy(hY, dY, bytes, cudaMemcpyDeviceToHost);

	bool ok = true;
	for (unsigned int i = 0; i < N; ++i) {
		if (fabs(hY[i] - hRef[i]) > 1e-4f) { ok = false; break; }
	}

	printf("Brent-Kung Test (%s, N=%u) -> %s\n", title, N, ok ? "PASS" : "FAIL");
    
	cudaFree(dX); cudaFree(dY);
	free(hX); free(hY); free(hRef);
	return ok;
}

int main(){
	// Kogge-Stone tests: A few arbitrary sizes including non-power-of-two and multi-block
	printf("========== Kogge-Stone Scan Tests ==========\n");
	run_scan_test("tiny", 1);
	run_scan_test("small", 7);
	run_scan_test("one-less-than-block", SECTION_SIZE - 1);
	run_scan_test("exactly-one-block", SECTION_SIZE);
	run_scan_test("one-more-than-block", SECTION_SIZE + 1);
	run_scan_test("multi-block", 1000);
	run_scan_test("odd multi-block", 2049);
	
	// Brent-Kung tests
	printf("\n========== Brent-Kung Scan Tests ==========\n");
	run_brent_kung_test("tiny", 1);
	run_brent_kung_test("small", 7);
	run_brent_kung_test("one-less-than-section", SECTION_SIZE - 1);
	run_brent_kung_test("exactly-one-section", SECTION_SIZE);
	run_brent_kung_test("one-more-than-section", SECTION_SIZE + 1);
	run_brent_kung_test("multi-block", 1000);
	run_brent_kung_test("odd multi-block", 2049);
	return 0;
}

