#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                         \
    cudaError_t err = (call);                         \
    if (err != cudaSuccess) {                         \
        fprintf(stderr, "CUDA error %s:%d: %s\n",     \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                           \
    }                                                 \
} while (0)

// Type definition for merge function pointer
typedef void (*merge_func_t)(int* A, int m, int* B, int n, int* C);

// Co-rank function from PMPP Figure 12.5
__device__ __host__ int co_rank(int k, int* A, int m, int* B, int n) {
    int i = k < m ? k : m;  // i = min(k,m)
    int j = k - i;
    int i_low = 0 > (k-n) ? 0 : k-n;  // i_low = max(0,k-n)
    int j_low = 0 > (k-m) ? 0 : k-m;  // j_low = max(0,k-m)
    int delta;
    bool active = true;
    
    while(active) {
        if (i > 0 && j < n && A[i-1] > B[j]) {
            delta = ((i - i_low +1) >> 1);  // ceil((i-i_low)/2)
            j_low = j;
            j = j + delta;
            i = i - delta;
        } else if (j > 0 && i < m && B[j-1] >= A[i]) {
            delta = ((j - j_low +1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        } else {
            active = false;
        }
    }
    return i;
}


// CPU merge algorithm without co-rank
// Standard binary merge for two sorted arrays
void cpu_merge(int* A, int m, int* B, int n, int* C) {
    int i = 0, j = 0, k = 0;
    
    // Merge the two sorted arrays
    while (i < m && j < n) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    
    // Copy remaining elements from A
    while (i < m) {
        C[k++] = A[i++];
    }
    
    // Copy remaining elements from B
    while (j < n) {
        C[k++] = B[j++];
    }
}

// Sequential merge helper function
__device__ void merge_sequential(int* A, int m, int* B, int n, int* C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    while (i < m) {
        C[k++] = A[i++];
    }
    while (j < n) {
        C[k++] = B[j++];
    }
}

// CUDA merge basic kernel from PMPP Figure 12.9
__global__ void merge_basic_kernel(int* A, int m, int* B, int n, int* C) {
    int tidy = blockIdx.x * blockDim.x + threadIdx.x;
    int elementsPerThread = (m + n + blockDim.x - 1) / (blockDim.x);
    int k_curr = tidy * elementsPerThread;  // start output index
    int k_next = min((tidy+1)*elementsPerThread, m+n);  // end output index
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    merge_sequential(&A[i_curr], i_next-i_curr, &B[j_curr], j_next-j_curr, &C[k_curr]);
}

// GPU wrapper function for merge using co-rank
void gpu_merge(int* h_A, int m, int* h_B, int n, int* h_C) {
    int* d_A = nullptr;
    int* d_B = nullptr;
    int* d_C = nullptr;
    
    int total = m + n;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_A, m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_B, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_C, total * sizeof(int)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, m * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    merge_basic_kernel<<<gridSize, blockSize>>>(d_A, m, d_B, n, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, total * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}


// Verify if the merge result is correct
int verify_merge(int* A, int m, int* B, int n, int* C) {
    // Check if C is sorted
    for (int i = 0; i < m + n - 1; i++) {
        if (C[i] > C[i + 1]) {
            return 0;  // Not sorted
        }
    }
    
    // Count occurrences in A
    int* count_A = (int*)calloc(1001, sizeof(int));
    for (int i = 0; i < m; i++) {
        count_A[A[i]]++;
    }
    
    // Count occurrences in B
    int* count_B = (int*)calloc(1001, sizeof(int));
    for (int i = 0; i < n; i++) {
        count_B[B[i]]++;
    }
    
    // Count occurrences in C
    int* count_C = (int*)calloc(1001, sizeof(int));
    for (int i = 0; i < m + n; i++) {
        count_C[C[i]]++;
    }
    
    // Verify counts match
    int valid = 1;
    for (int i = 0; i <= 1000; i++) {
        if (count_A[i] + count_B[i] != count_C[i]) {
            valid = 0;
            break;
        }
    }
    
    free(count_A);
    free(count_B);
    free(count_C);
    
    return valid;
}

// Print array
void print_array(int* arr, int n, const char* name) {
    printf("%s: ", name);
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// Generic test case structure
struct TestCase {
    int* A;
    int m;
    int* B;
    int n;
    const char* name;
    int is_dynamic;  // Flag to indicate if arrays are dynamically allocated
};

// Run a single test with given merge function
void run_single_test(struct TestCase* test, merge_func_t merge_func) {
    int* C = (int*)malloc((test->m + test->n) * sizeof(int));
    
    if (test->m <= 10) print_array(test->A, test->m, "Array A");
    if (test->n <= 10) print_array(test->B, test->n, "Array B");
    
    clock_t start = clock();
    merge_func(test->A, test->m, test->B, test->n, C);
    clock_t end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    
    if (test->m + test->n <= 10) {
        print_array(C, test->m + test->n, "Merged");
    } else {
        printf("Merged: %d elements (output truncated)\n", test->m + test->n);
    }
    
    if (verify_merge(test->A, test->m, test->B, test->n, C)) {
        printf("[PASS] %s .", test->name);
        if (elapsed > 0) printf(" (%.6f seconds)", elapsed);
        printf("\n\n");
    } else {
        printf("[FAIL] %s .\n\n", test->name);
    }
    
    free(C);
}

// Run all tests with given merge function
void run_all_tests(merge_func_t merge_func, const char* kernel_name) {
    printf("\n");
    printf("=============================================================\n");
    printf("Testing Merge Algorithm: %s\n", kernel_name);
    printf("=============================================================\n\n");
    
    // Create test cases
    struct TestCase tests[5];
    int test_count = 0;
    
    // Test case 1: Small arrays
    {
        static int A1[] = {1, 3, 5, 7, 9};
        static int B1[] = {2, 4, 6, 8, 10};
        tests[test_count].A = A1;
        tests[test_count].m = 5;
        tests[test_count].B = B1;
        tests[test_count].n = 5;
        tests[test_count].name = "Test Case 1: Small sorted arrays";
        tests[test_count].is_dynamic = 0;
        test_count++;
    }
    
    // Test case 2: Different sizes
    {
        static int A2[] = {1, 5, 9, 13, 17, 21, 25};
        static int B2[] = {2, 4, 6};
        tests[test_count].A = A2;
        tests[test_count].m = 7;
        tests[test_count].B = B2;
        tests[test_count].n = 3;
        tests[test_count].name = "Test Case 2: Different sized arrays";
        tests[test_count].is_dynamic = 0;
        test_count++;
    }
    
    // Test case 3: Overlapping values
    {
        static int A3[] = {1, 3, 3, 5, 7};
        static int B3[] = {2, 3, 4, 6, 8};
        tests[test_count].A = A3;
        tests[test_count].m = 5;
        tests[test_count].B = B3;
        tests[test_count].n = 5;
        tests[test_count].name = "Test Case 3: Overlapping values";
        tests[test_count].is_dynamic = 0;
        test_count++;
    }
    
    // Test case 4: One array is a subset
    {
        static int A4[] = {10, 20, 30, 40, 50};
        static int B4[] = {5, 15, 25};
        tests[test_count].A = A4;
        tests[test_count].m = 5;
        tests[test_count].B = B4;
        tests[test_count].n = 3;
        tests[test_count].name = "Test Case 4: One array is a subset";
        tests[test_count].is_dynamic = 0;
        test_count++;
    }
    
    // Test case 5: Large arrays
    {
        int m5 = 1000, n5 = 1000;
        int* A5 = (int*)malloc(m5 * sizeof(int));
        int* B5 = (int*)malloc(n5 * sizeof(int));
        
        srand(42);  // Fixed seed for reproducibility
        for (int i = 0; i < m5; i++) {
            A5[i] = (rand() % 500) * 2;
        }
        std::sort(A5, A5 + m5);
        
        for (int i = 0; i < n5; i++) {
            B5[i] = (rand() % 500) * 2 + 1;
        }
        std::sort(B5, B5 + n5);
        
        tests[test_count].A = A5;
        tests[test_count].m = m5;
        tests[test_count].B = B5;
        tests[test_count].n = n5;
        tests[test_count].name = "Test Case 5: Large arrays (1000 + 1000 elements)";
        tests[test_count].is_dynamic = 1;
        test_count++;
    }
    
    // Run all tests
    for (int i = 0; i < test_count; i++) {
        printf("Test Case %d: %s\n", i + 1, tests[i].name);
        run_single_test(&tests[i], merge_func);
    }
    
    // Clean up dynamically allocated test arrays
    for (int i = 0; i < test_count; i++) {
        if (tests[i].is_dynamic) {
            free(tests[i].A);
            free(tests[i].B);
        }
    }
}

int main() {
    printf("=== Merge Algorithm Test Suite ===\n");
    printf("Testing different merge implementations\n");
    
    // Test CPU merge algorithm
    run_all_tests(cpu_merge, "CPU Merge (Standard)");
    
    // Test GPU merge algorithm with co-rank
    run_all_tests(gpu_merge, "GPU Merge (Co-rank + Basic Kernel)");
    
    printf("\n=============================================================\n");
    printf("All Tests Completed\n");
    printf("=============================================================\n");
    
    return 0;
}
