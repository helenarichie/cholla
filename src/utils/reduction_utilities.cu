/*!
 * \file reduction_utilities.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the implementation of the GPU resident reduction utilities
 *
 */

// STL Includes
#include <float.h>

// External Includes

// Local Includes
#include "../utils/DeviceVector.h"
#include "../utils/reduction_utilities.h"

#ifdef CUDA
namespace reduction_utilities
{
// =====================================================================
__global__ void kernelReduceMax(Real* in, Real* out, size_t N)
{
  // Initialize maxVal to the smallest possible number
  Real maxVal = -DBL_MAX;

  // Grid stride loop to perform as much of the reduction as possible
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    // A transformation could go here

    // Grid stride reduction
    maxVal = max(maxVal, in[i]);
  }

  // Find the maximum val in the grid and write it to `out`. Note that
  // there is no execution/memory barrier after this and so the
  // reduced scalar is not available for use in this kernel. The grid
  // wide barrier can be accomplished by ending this kernel here and
  // then launching a new one or by using cooperative groups. If this
  // becomes a need it can be added later
  gridReduceMax(maxVal, out);
}
// =====================================================================

// =====================================================================
__global__ void Kernel_Reduce_Add(Real* in, Real* out, size_t N)
{
  __shared__ Real sum_stride[TPB]; // array shared between each block
 
  for (int i = 0; i < TPB; i++) {
    sum_stride[i] = 0;
  }

// Grid stride loop to read global array into shared block-wide array
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    sum_stride[threadIdx.x] += in[i];
  }
  __syncthreads();

  Grid_Reduction_Add(sum_stride[threadIdx.x], out);
  if (threadIdx.x == 0) {
    printf("kernel: %f\n", *out);
  }
}
// =====================================================================
}  // namespace reduction_utilities
#endif  // CUDA