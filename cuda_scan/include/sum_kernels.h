#pragma once

__global__ void sum_par_scan(
    const float *summands,
    const int num_summands,
    float *result);

__global__ void sum_par(
    const float *summands,
    const int num_summands,
    float *result);