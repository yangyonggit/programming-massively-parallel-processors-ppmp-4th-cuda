// Kogge–Stone scan (inclusive) — block-wise only
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef SECTION_SIZE
#define SECTION_SIZE 256
#endif

#define CUDA_CHECK(call) do {                         \
    cudaError_t err = (call);                         \
    if (err != cudaSuccess) {                         \
        fprintf(stderr, "CUDA error %s:%d: %s\n",     \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::abort();                                 \
    }                                                 \
} while (0)

static inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// ----------------------------
// Kernel 1: block-local segmented inclusive scan
// - Output Y_partial (block-local segmented scan result)
// - Output prefixHead[i] = OR(head[base..i]) within the block (inclusive prefix OR)
// - Output carryOut[block] = Y_partial[last valid element in block] (tail sum)
// - Output blockHeadFirst[block] = head[base] (1 if block starts a new segment)
// - Output blockHeadLast[block] = OR(all heads in block) (1 if block contains any head)
// ----------------------------
template<int BLOCK_SIZE>
__global__ void k1_block_segmented_scan(
    const float* __restrict__ X,
    const uint8_t* __restrict__ head,   // 0/1, 1 means new segment starts here
    float* __restrict__ Y_partial,
    uint8_t* __restrict__ prefixHead,   // per-element: prefix OR of head within block
    float* __restrict__ carryOut,       // per-block tail sum
    uint8_t* __restrict__ blockHeadFirst, // per-block head of first element
    uint8_t* __restrict__ blockHeadLast,  // per-block: OR of all heads in block
    unsigned int N)
{
    __shared__ float s_val[BLOCK_SIZE];
    __shared__ uint8_t s_flag[BLOCK_SIZE];

    const unsigned int tid  = threadIdx.x;
    const unsigned int base = blockIdx.x * BLOCK_SIZE;
    const unsigned int i    = base + tid;

    // Load
    float v = 0.0f;
    uint8_t f = 1; // for out-of-range threads, set flag=1 to prevent accidental adds
    if (i < N) {
        v = X[i];
        f = head[i] ? 1 : 0;
    }
    s_val[tid]  = v;
    s_flag[tid] = f;
    __syncthreads();

    // Kogge-Stone style segmented inclusive scan inside the block
    // Pair operator:
    //   flag: always OR propagate
    //   val:  only add from left if current flag == 0 (i.e., not a head yet)
    #pragma unroll
    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        float  left_v = 0.0f;
        uint8_t left_f = 0;

        if (tid >= (unsigned)offset) {
            left_v = s_val[tid - offset];
            left_f = s_flag[tid - offset];
        }
        __syncthreads();

        if (tid >= (unsigned)offset) {
            if (s_flag[tid] == 0) {
                s_val[tid] += left_v;
            }
            s_flag[tid] = (uint8_t)(s_flag[tid] | left_f);
        }
        __syncthreads();
    }

    // Store per-element outputs
    if (i < N) {
        Y_partial[i]  = s_val[tid];
        prefixHead[i] = s_flag[tid]; // inclusive prefix OR of heads inside block
    }

	// Per-block summary
	if (tid == 0) {
		// blockHeadFirst: does this block start with a head?
		uint8_t first = 0;
		if (base < N) first = head[base] ? 1 : 0;
		blockHeadFirst[blockIdx.x] = first;

		// blockHeadLast & carryOut: use last valid element
		unsigned int end = base + BLOCK_SIZE;
		if (end > N) end = N;
		if (end > base) {
			unsigned int last = end - 1;           // global index
			unsigned int lane = last - base;       // lane in this block
			carryOut[blockIdx.x] = s_val[lane];
			blockHeadLast[blockIdx.x] = s_flag[lane]; // OR of all heads in block
		} else {
			carryOut[blockIdx.x] = 0.0f;
			blockHeadLast[blockIdx.x] = 0;
		}
	}
}

// ----------------------------
// Kernel 2: block-level segmented scan (single-block version)
// Input:
//   carryOut[b]         : tail sum from kernel1
//   blockHeadFirst[b]   : 1 if block b starts with a head
//   blockHeadLast[b]    : 1 if block b contains any head
// Output:
//   carryIn[b]          : what should be added to "continuation part" of block b
//
// We compute an inclusive segmented scan on blocks, then shift to build carryIn:
//   carryIn[0] = 0
//   carryIn[b] = (blockHeadFirst[b] ? 0 : incVal[b-1])
// The segmented scan uses blockHeadLast as flags to prevent adding carries from
// blocks that contain segment boundaries.
// ----------------------------
template<int MAX_BLOCKS>
__global__ void k2_scan_block_carries_singleblock(
    const float* __restrict__ carryOut,
	const uint8_t* __restrict__ blockHeadFirst,
	const uint8_t* __restrict__ blockHeadLast,
    float* __restrict__ carryIn,
    unsigned int numBlocks)
{
    __shared__ float s_val[MAX_BLOCKS];
    __shared__ uint8_t s_flag[MAX_BLOCKS];

    const unsigned int t = threadIdx.x;

    // load
    float v = 0.0f;
    uint8_t f = 1;
    if (t < numBlocks) {
        v = carryOut[t];
		f = blockHeadLast[t] ? 1 : 0;  // Use blockHeadLast for segmented scan flag
    }
    s_val[t] = v;
    s_flag[t] = f;
    __syncthreads();

    // inclusive segmented scan across blocks
    #pragma unroll
    for (int offset = 1; offset < MAX_BLOCKS; offset <<= 1) {
        float  left_v = 0.0f;
        uint8_t left_f = 0;

        if (t >= (unsigned)offset) {
            left_v = s_val[t - offset];
            left_f = s_flag[t - offset];
        }
        __syncthreads();

        if (t < numBlocks && t >= (unsigned)offset) {
            if (s_flag[t] == 0) {
                s_val[t] += left_v;
            }
            s_flag[t] = (uint8_t)(s_flag[t] | left_f);
        }
        __syncthreads();
    }

    // build carryIn by shifting inclusive result
    if (t < numBlocks) {
        if (t == 0) {
            carryIn[t] = 0.0f;
        } else {
			carryIn[t] = blockHeadFirst[t] ? 0.0f : s_val[t - 1];
        }
    }
}

// ----------------------------
// Kernel 3: add carry-in back to elements that belong to "continuation segment"
// Rule for element i in block b:
//   add carryIn[b] only if prefixHead[i] == 0
// because prefixHead[i] is OR(head[block_start .. i]) within the block,
// so prefixHead==0 means: no head encountered so far => still continuing from previous block.
// ----------------------------
__global__ void k3_add_carryin(
    float* __restrict__ Y,
    const uint8_t* __restrict__ prefixHead,
    const float* __restrict__ carryIn,
    unsigned int N,
    unsigned int blockSize)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    unsigned int b = i / blockSize;
    if (prefixHead[i] == 0) {
        Y[i] += carryIn[b];
    }
}

// ----------------------------
// Host wrapper
// Constraints (for this learning version):
//   numBlocks <= 1024  (Kernel2 uses one block of 1024 threads)
// ----------------------------
template<int BLOCK_SIZE>
void segmented_inclusive_scan_gpu(
    const float* d_X,
    const uint8_t* d_head,
    float* d_Y,
    unsigned int N)
{
    const unsigned int numBlocks = cdiv(N, (unsigned)BLOCK_SIZE);

    if (numBlocks > 1024) {
        fprintf(stderr, "This demo implementation requires numBlocks <= 1024. "
                        "Got numBlocks=%u. Use recursive/multi-level for larger inputs.\n",
                        numBlocks);
        std::abort();
    }

    // temp buffers
    float* d_carryOut = nullptr;
    float* d_carryIn  = nullptr;
    uint8_t* d_prefixHead = nullptr;
	uint8_t* d_blockHeadFirst = nullptr;
	uint8_t* d_blockHeadLast = nullptr;

    CUDA_CHECK(cudaMalloc(&d_carryOut,        numBlocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_carryIn,         numBlocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prefixHead,      N * sizeof(uint8_t)));
	CUDA_CHECK(cudaMalloc(&d_blockHeadFirst,  numBlocks * sizeof(uint8_t)));
	CUDA_CHECK(cudaMalloc(&d_blockHeadLast,   numBlocks * sizeof(uint8_t)));

    // Kernel 1
    k1_block_segmented_scan<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE>>>(
		d_X, d_head, d_Y, d_prefixHead, d_carryOut, d_blockHeadFirst, d_blockHeadLast, N);
    CUDA_CHECK(cudaGetLastError());

    // Kernel 2 (single block, 1024 threads)
    k2_scan_block_carries_singleblock<1024><<<1, 1024>>>(
		d_carryOut, d_blockHeadFirst, d_blockHeadLast, d_carryIn, numBlocks);
    CUDA_CHECK(cudaGetLastError());

    // Kernel 3
    k3_add_carryin<<<numBlocks, BLOCK_SIZE>>>(
        d_Y, d_prefixHead, d_carryIn, N, BLOCK_SIZE);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_carryOut));
    CUDA_CHECK(cudaFree(d_carryIn));
    CUDA_CHECK(cudaFree(d_prefixHead));
	CUDA_CHECK(cudaFree(d_blockHeadFirst));
	CUDA_CHECK(cudaFree(d_blockHeadLast));
}


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

// CPU reference: segmented inclusive scan
static void cpu_segmented_inclusive_scan(const float* X, const uint8_t* head, float* Y, unsigned int N) {
	float acc = 0.0f;
	for (unsigned int i = 0; i < N; ++i) {
		if (head[i]) {
			acc = X[i];  // start new segment
		} else {
			acc += X[i]; // continue segment
		}
		Y[i] = acc;
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

// Helper function to generate head patterns
typedef void(*HeadPatternGenerator)(uint8_t* head, int N);

static void gen_single_segment(uint8_t* head, int N) {
	head[0] = 1;
	for (int i = 1; i < N; ++i) head[i] = 0;
}

static void gen_equal_segments_64(uint8_t* head, int N) {
	for (int i = 0; i < N; ++i) head[i] = (i % 64 == 0) ? 1 : 0;
}

static void gen_irregular_segments(uint8_t* head, int N) {
	for (int i = 0; i < N; ++i) head[i] = 0;
	if (N > 0) head[0] = 1;
	if (N > 7) head[7] = 1;
	if (N > 23) head[23] = 1;
	if (N > 150) head[150] = 1;
	if (N > 400) head[400] = 1;
	if (N > 999) head[999] = 1;
}

static void gen_all_heads(uint8_t* head, int N) {
	for (int i = 0; i < N; ++i) head[i] = 1;
}

static void gen_cross_block_boundaries(uint8_t* head, int N) {
	for (int i = 0; i < N; ++i) head[i] = 0;
	if (N > 0) head[0] = 1;
	if (N > 250) head[250] = 1;
	if (N > 350) head[350] = 1;
}

// Segmented scan test function
static bool run_segmented_scan_test(const char* title, int N, HeadPatternGenerator gen) {
	size_t bytes = N * sizeof(float);
	size_t head_bytes = N * sizeof(uint8_t);
	float* hX = (float*)malloc(bytes);
	float* hY = (float*)malloc(bytes);
	float* hRef = (float*)malloc(bytes);
	uint8_t* h_head = (uint8_t*)malloc(head_bytes);

	// Generate head pattern
	gen(h_head, N);

	// Generate test data
	for (int i = 0; i < N; ++i) {
		hX[i] = (float)(i % 10 + 1); // 1..10 pattern
	}

	cpu_segmented_inclusive_scan(hX, h_head, hRef, N);

	float *dX = nullptr, *dY = nullptr;
	uint8_t *d_head = nullptr;
	CUDA_CHECK(cudaMalloc(&dX, bytes));
	CUDA_CHECK(cudaMalloc(&dY, bytes));
	CUDA_CHECK(cudaMalloc(&d_head, head_bytes));
	CUDA_CHECK(cudaMemcpy(dX, hX, bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_head, h_head, head_bytes, cudaMemcpyHostToDevice));

	segmented_inclusive_scan_gpu<SECTION_SIZE>(dX, d_head, dY, N);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(hY, dY, bytes, cudaMemcpyDeviceToHost));

	bool ok = true;
	for (int i = 0; i < N; ++i) {
		if (fabs(hY[i] - hRef[i]) > 1e-4f) {
			ok = false;
			printf("  Mismatch at i=%d: GPU=%.2f, CPU=%.2f, head[%d]=%d\n", 
			       i, hY[i], hRef[i], i, h_head[i]);
			// Show surrounding context
			int start = (i > 5) ? i - 5 : 0;
			int end = (i + 5 < N) ? i + 5 : N;
			for (int j = start; j < end; ++j) {
				printf("    [%d] GPU=%.2f CPU=%.2f head=%d\n", 
				       j, hY[j], hRef[j], h_head[j]);
			}
			break;
		}
	}

	printf("Segmented Scan Test (%s, N=%d) -> %s\n", title, N, ok ? "PASS" : "FAIL");
	
	CUDA_CHECK(cudaFree(dX));
	CUDA_CHECK(cudaFree(dY));
	CUDA_CHECK(cudaFree(d_head));
	free(hX); free(hY); free(hRef); free(h_head);
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

	// Segmented scan tests
	printf("\n========== Segmented Inclusive Scan Tests ==========\n");
	run_segmented_scan_test("single-segment", 500, gen_single_segment);
	run_segmented_scan_test("equal-segments-64", 512, gen_equal_segments_64);
	run_segmented_scan_test("irregular-segments", 1000, gen_irregular_segments);
	run_segmented_scan_test("all-heads", 100, gen_all_heads);
	run_segmented_scan_test("cross-block-boundaries", 600, gen_cross_block_boundaries);

	return 0;
}

