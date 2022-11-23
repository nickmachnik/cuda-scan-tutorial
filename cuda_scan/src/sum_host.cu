#include <cmath>
#include <gpuerrors.h>
#include <sum_kernels.h>

typedef void (*sum_kernel)(const float *, const int, float *);

void timed_sum_kernel_call(
    const sum_kernel,
    const float *gpu_summands,
    const int n,
    float *gpu_result,
    const dim3 blocks_per_grid,
    const dim3 threads_per_block,
    const int shared_memory_bytes

)
{
    // vars needed for kernel execution timing
    float time;
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // record starting time
    HANDLE_ERROR(cudaEventRecord(start, 0));

    // call kernel
    sum_kernel<<<blocks_per_grid, threads_per_block, shared_memory_bytes>>>(gpu_summands, n, gpu_result);
    CudaCheckError();

    // record finish time
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
    printf("Kernel call time:  %3.1f ms \n", time);
}

float sum_par_scan_cu(const float *summands, const int n)
{
    // declare pointers to gpu data
    float *gpu_summands;
    float *gpu_result;

    // our final result
    float result;

    // allocate
    HANDLE_ERROR(cudaMalloc(&gpu_summands, sizeof(float) * n));
    HANDLE_ERROR(cudaMemcpy(gpu_summands, summands, sizeof(float) * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc(&gpu_result, sizeof(float)));

    // define grid size (not needed here, so set to 1)
    dim3 blocks_per_grid(1);

    // use as many threads as possible, 1024 is max per bock
    dim3 threads_per_block(1024);
    int shared_memory_bytes(1024 * sizeof(float));

    // allow passing shared memory size through kernel call
    HANDLE_ERROR(cudaFuncSetAttribute(sum_par_scan, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes));

    // call kernel
    sum_kernel sk = sum_par_scan;
    timed_sum_kernel_call(sk, gpu_summands, n, gpu_result, blocks_per_grid, threads_per_block, shared_memory_bytes);

    // copy result from gpu to host
    HANDLE_ERROR(cudaMemcpy(&result, gpu_result, sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

int highestPowerof2(int n)
{
    int res = 0;
    for (int i = n; i >= 1; i--)
    {
        // If i is a power of 2
        if ((i & (i - 1)) == 0)
        {
            res = i;
            break;
        }
    }
    return res;
}

float sum_par_cu(const float *summands, const int n)
{
    // declare pointers to gpu data
    float *gpu_summands;
    float *gpu_result;
    float result;
    // allocate
    HANDLE_ERROR(cudaMalloc(&gpu_summands, sizeof(float) * n));
    HANDLE_ERROR(cudaMemcpy(gpu_summands, summands, sizeof(float) * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc(&gpu_result, sizeof(float)));
    dim3 blocks_per_grid(1);
    int num_threads = highestPowerof2(sqrt(n));
    if (num_threads > 1024)
    {
        num_threads = 1024;
    }
    num_threads = 1024;
    dim3 threads_per_block(num_threads);
    int shared_memory_bytes(num_threads * sizeof(float));
    HANDLE_ERROR(cudaFuncSetAttribute(sum_par_scan, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes));
    // call kernel
    sum_par<<<blocks_per_grid, threads_per_block, shared_memory_bytes>>>(gpu_summands, n, gpu_result);
    CudaCheckError();
    HANDLE_ERROR(cudaMemcpy(&result, gpu_result, sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}
