// compute sum of input vector using the par + scan algorithm
__global__ void par_scan(
    const float *summands,
    const int num_summands,
    float *result)
{
    int tix = threadId.x;
    int num_threads = blockDim.x;
    // divide summands into sections
    const int num_sections = (num_summands + num_threads - 1) / num_threads;

    // --- compute par bit ---
    __shared_ float thread_sums[num_threads];
    float thread_sum = 0.0;
    int vix = 0;

    // note how this is different from a sensible serial loop
    for (size_t section_ix = 0; section_ix < num_sections; section_ix++)
    {
        vix = num_threads * section_ix + tix;
        if (vix < num_summands)
        {
            thread_sum += summands[vix];
        }
    }

    thread_sums[tix] = thread_sum;

    // --- compute scan bit ---
    float tmp = 0.0;
    __syncthreads();
    for (int step = 1; step < num_threads; step = step * 2)
    {
        if (tix < step)
        {
            tmp = thread_sums[tix];
        }
        else
        {
            tmp = thread_sums[tix] + thread_sums[tix - step];
        }
        __syncthreads();
        thread_sums[tix] = tmp;
        __syncthreads();
    }

    if (tx == 0)
    {
        float s = thread_sums[num_threads - 1];
        *results = s;
    }
}