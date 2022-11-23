#include <cmath>
#include <gpuerrors.h>
#include <sum_kernels.h>

float sum_par_scan_cu(const float *summands, const int n)
{
    // declare pointers to gpu data
    float *gpu_summands;
    float *gpu_result;
    float *result;
    // allocate
    HANDLE_ERROR(cudaMalloc(&gpu_summands, sizeof(float) * n));
    HANDLE_ERROR(cudaMalloc(&gpu_result, sizeof(float)));
    dim3 blocks_per_grid(1);
    // use as many threads as possible, 1024 is max per bock
    dim3 threads_per_block(1024);
    // call kernel
    sum_par_scan<<<blocks_per_grid, threads_per_block>>>(gpu_summands, n, gpu_result);
    CudaCheckError();
    HANDLE_ERROR(cudaMemcpy(result, gpu_result, sizeof(float), cudaMemcpyDeviceToHost));
    return *result;
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
    float *result;
    // allocate
    HANDLE_ERROR(cudaMalloc(&gpu_summands, sizeof(float) * n));
    HANDLE_ERROR(cudaMalloc(&gpu_result, sizeof(float)));
    dim3 blocks_per_grid(1);
    int num_threads = highestPowerof2(sqrt(n));
    if (num_threads > 1024)
    {
        num_threads = 1024;
    }
    dim3 threads_per_block(num_threads);
    // call kernel
    sum_par<<<blocks_per_grid, threads_per_block>>>(gpu_summands, n, gpu_result);
    CudaCheckError();
    HANDLE_ERROR(cudaMemcpy(result, gpu_result, sizeof(float), cudaMemcpyDeviceToHost));
    return *result;
}