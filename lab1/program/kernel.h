#pragma once

#include <mc_runtime.h>
void call_parallel_sum_every_2nd_element(int *sum, int const *arr, int n);
__global__ void parallel_sum_every_2nd_element(int *sum, int const *arr, int n);