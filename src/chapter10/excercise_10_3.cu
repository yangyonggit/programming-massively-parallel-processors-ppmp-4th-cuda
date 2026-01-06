// Convergent reduction variant that accumulates to the last element
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void ConvergentSumReductionKernel(float* input, float* output) {
	unsigned int i = threadIdx.x + blockDim.x;
	for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
		if (i >= (2 * blockDim.x - stride)) {
			input[i] += input[i - stride];
		}
		__syncthreads();
	}
	unsigned int last = blockDim.x * 2 - 1;
	if (threadIdx.x == blockDim.x - 1) {
		*output = input[last];
	}
}

// CPU reference sum
static float cpu_sum(const float* a, int n) {
	double s = 0.0; // accumulate in double for better numeric stability
	for (int i = 0; i < n; ++i) s += a[i];
	return (float)s;
}

#define CUDA_CHECK(call) \
	do { \
		cudaError_t err__ = (call); \
		if (err__ != cudaSuccess) { \
			fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
			exit(EXIT_FAILURE); \
		} \
	} while (0)

int main() {
	const int block = 1024;            // threads
	const int size = block * 2;        // kernel expects 2*block elements
	const size_t bytes = size * sizeof(float);

	float* h = (float*)malloc(bytes);
	float h_out = 0.0f;
	float *d_in = nullptr, *d_out = nullptr;
	CUDA_CHECK(cudaMalloc(&d_in, bytes));
	CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

	float eps = 1e-2f;

	// Test 1: all 1s
	printf("Test 1: all 1s\n");
	for (int i = 0; i < size; ++i) h[i] = 1.0f;
	float expect = cpu_sum(h, size);
	CUDA_CHECK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));
	ConvergentSumReductionKernel<<<1, block>>>(d_in, d_out);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
	printf("CPU: %.2f, GPU: %.2f, %s (diff=%.6f)\n", expect, h_out,
		   fabsf(expect - h_out) < eps ? "PASS" : "FAIL", fabsf(expect - h_out));

	// Test 2: sequential values
	printf("\nTest 2: sequential values 0..\n");
	for (int i = 0; i < size; ++i) h[i] = (float)i;
	expect = cpu_sum(h, size);
	CUDA_CHECK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));
	ConvergentSumReductionKernel<<<1, block>>>(d_in, d_out);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
	printf("CPU: %.2f, GPU: %.2f, %s (diff=%.6f)\n", expect, h_out,
		   fabsf(expect - h_out) < eps ? "PASS" : "FAIL", fabsf(expect - h_out));

	// Test 3: random values
	printf("\nTest 3: random values\n");
	for (int i = 0; i < size; ++i) h[i] = (float)(rand() % 100) / 10.0f;
	expect = cpu_sum(h, size);
	CUDA_CHECK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));
	ConvergentSumReductionKernel<<<1, block>>>(d_in, d_out);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
	printf("CPU: %.2f, GPU: %.2f, %s (diff=%.6f)\n", expect, h_out,
		   fabsf(expect - h_out) < eps ? "PASS" : "FAIL", fabsf(expect - h_out));

	// cleanup
	free(h);
	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
}
