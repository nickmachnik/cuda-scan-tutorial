#include <cmath>
#include <gpuerrors.h>
#include <sum_kernels.h>

float sum_par_scan_cu(const float *summands, const int n)
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
    // use as many threads as possible, 1024 is max per bock
    dim3 threads_per_block(1024);
    int shared_memory_bytes(1024 * sizeof(float));
    HANDLE_ERROR(cudaFuncSetAttribute(sum_par_scan, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes));
    // call kernel
    sum_par_scan<<<blocks_per_grid, threads_per_block, shared_memory_bytes>>>(gpu_summands, n, gpu_result);
    CudaCheckError();
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
