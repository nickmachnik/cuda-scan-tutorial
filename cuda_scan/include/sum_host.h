#pragma once

float sum_par_scan_cu(const float *summands, const int n, const int num_threads);

float sum_par_cu(const float *summands, const int n, const int num_threads);