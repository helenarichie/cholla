/*!
 * \file dust_tracking.cu
 * \author Helena Richie (helenarichie@gmail.com)
 * \brief 
 */

#ifdef CLOUD_TRACKING

  // STL includes
  #include <stdio.h>

  #include <cstdio>
  #include <fstream>
  #include <vector>

  // Local includes
  #include "../dust/dust_cuda.h"
  #include "../global/global.h"
  #include "../global/global_cuda.h"
  #include "../grid/grid3D.h"
  #include "../grid/grid_enum.h"
  #include "../utils/cuda_utilities.h"
  #include "../utils/gpu.hpp"
  #include "../utils/hydro_utilities.h"

void Dust_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma)
{
  int n_cells = nx * ny * nnz;
  int ngrid   = (n_cells + TPB - 1) / TPB;
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(Dust_Kernel, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost, n_fields, dt, gamma);
  CudaCheckError();
}

__global__ void Dust_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma)
{
  // get grid indices
  int n_cells = nx * ny * nz;
  int is, ie, js, je, ks, ke;
  cuda_utilities::Get_Real_Indices(n_ghost, nx, ny, nz, is, ie, js, je, ks, ke);
  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int id      = threadIdx.x + blockId * blockDim.x;
  int id_z    = id / (nx * ny);
  int id_y    = (id - id_z * nx * ny) / nx;
  int id_x    = id - id_z * nx * ny - id_y * nx;

  // define physics variables
  Real d_gas, d_dust;  // fluid mass densities
  Real n;              // gas number density
  Real mu = 0.6;       // mean molecular weight
  Real T, E, P;        // temperature, energy, pressure
  Real vx, vy, vz;     // velocities

  if (id_x >= is && id_x < ie && id_y >= js && id_y < je && id_z >= ks && id_z < ke) {
    d_gas  = dev_conserved[id + n_cells * grid_enum::density];
  }
}

// Shin et al. (2008)
__device__ __host__ Real Calc_Cloud_Velocity(Real mass_cloud, Real density_cl, Real velocity_x_cl)
{
  Real velocity_x_avg

  return velocity_x_avg;
}

#endif  // CLOUD_TRACKING
