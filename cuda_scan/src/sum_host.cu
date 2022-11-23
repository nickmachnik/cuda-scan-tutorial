#include <cmath>
#include <gpuerrors.h>
#include <sum_kernels.h>

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

    // vars needed for kernel execution timing
    float time;
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // record starting time
    HANDLE_ERROR(cudaEventRecord(start, 0));

    // call kernel
    sum_par_scan<<<blocks_per_grid, threads_per_block, shared_memory_bytes>>>(gpu_summands, n, gpu_result);
    CudaCheckError();

    // record finish time
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
    printf("Kernel call time:  %3.5f ms \n", time);

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

    // define grid size (not needed here, so set to 1)
    dim3 blocks_per_grid(1);

    // find optimal number of threads (should be power of 2 and close to sqrt(n))
    int num_threads = highestPowerof2(sqrt(n));
    if (num_threads > 1024)
    {
        num_threads = 1024;
    }

    // set number of threads
    dim3 threads_per_block(num_threads);

    // compute needed amount of shared memory
    int shared_memory_bytes(num_threads * sizeof(float));

    // allow passing shared memory size to kernel
    HANDLE_ERROR(cudaFuncSetAttribute(sum_par_scan, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes));

    // vars needed for kernel execution timing
    float time;
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // record starting time
    HANDLE_ERROR(cudaEventRecord(start, 0));

    // call kernel
    sum_par<<<blocks_per_grid, threads_per_block, shared_memory_bytes>>>(gpu_summands, n, gpu_result);
    CudaCheckError();

    // record finish time
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
    printf("Kernel call time:  %3.5f ms \n", time);

    HANDLE_ERROR(cudaMemcpy(&result, gpu_result, sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}
