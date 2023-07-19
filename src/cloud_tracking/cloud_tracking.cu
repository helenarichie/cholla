/*!
 * \file cloud_tracking.cu
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

  #include "../cloud_tracking/cloud_tracking.h"
  #include "../global/global.h"
  #include "../global/global_cuda.h"
  #include "../grid/grid3D.h"
  #include "../grid/grid_enum.h"
  #include "../utils/DeviceVector.h"
  #include "../utils/cuda_utilities.h"
  #include "../utils/gpu.hpp"
  #include "../utils/hydro_utilities.h"
  #include "../utils/reduction_utilities.h"

// void Cloud_Frame_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma,
// Real density_cloud_init, Real *integrand, Real *density_cloud_tot)
void Cloud_Frame_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma,
                        Real density_cloud_init, Real *integrand, Real *density_cloud_tot)
{
  // Allocate the device memory
  cuda_utilities::DeviceVector<Real> static integrand_cloud(1);
  cuda_utilities::DeviceVector<Real> static density_cloud(1);

  cuda_utilities::AutomaticLaunchParams static const launchParams(Cloud_Tracking_Kernel);
  hipLaunchKernelGGL(Cloud_Tracking_Kernel, launchParams.numBlocks, launchParams.threadsPerBlock, 0, 0, dev_conserved,
                     nx, ny, nz, n_ghost, n_fields, dt, gamma, density_cloud_init, integrand_cloud.data(),
                     density_cloud.data());
  CudaCheckError();

  *integrand         = integrand_cloud[0];
  *density_cloud_tot = density_cloud[0];
}

__global__ void Cloud_Tracking_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt,
                                      Real gamma, Real density_cloud_init, Real *integrand_cloud, Real *density_cloud)
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

  Real density, velocity_x;

  // for (size_t id = threadIdx.x + blockIdx.x * blockDim.x; id < n_cells; id += blockDim.x * gridDim.x) {
  //  threads corresponding to real cells do the calculation
  if (id_x >= is && id_x < ie && id_y >= js && id_y < je && id_z >= ks && id_z < ke) {
    // get thread's gas density
    density    = dev_conserved[id + n_cells * grid_enum::density];
    velocity_x = dev_conserved[id + n_cells * grid_enum::momentum_x] / density;

    if (density > (1 / 3 * (density_cloud_init / DENSITY_UNIT))) {
      // Do grid-wide reduction to compute mass-averaged cloud velocity (Shin et al. 2008, eq. 9)
      reduction_utilities::Grid_Reduction_Add(density * velocity_x, integrand_cloud);
      reduction_utilities::Grid_Reduction_Add(density, density_cloud);
    }
  }
}

void Update_Grid_Velocities(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma,
                            Real velocity_cloud, Real density_cloud_tot)
{
  cuda_utilities::AutomaticLaunchParams static const launchParams(Velocity_Update);
  hipLaunchKernelGGL(Velocity_Update, launchParams.numBlocks, launchParams.threadsPerBlock, 0, 0, dev_conserved, nx, ny,
                     nz, n_ghost, n_fields, dt, gamma, velocity_cloud, density_cloud_tot);
  CudaCheckError();
}

__global__ void Velocity_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt,
                                Real gamma, Real velocity_cloud, Real density_cloud_tot)
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

  // for (size_t id = threadIdx.x + blockIdx.x * blockDim.x; id < n_cells; id += blockDim.x * gridDim.x) {
  //  threads corresponding to real cells do the calculation
  if (id_x >= is && id_x < ie && id_y >= js && id_y < je && id_z >= ks && id_z < ke) {
    dev_conserved[id + n_cells * grid_enum::momentum_x] -= velocity_cloud * density_cloud_tot;
  }
  // }
}

#endif  // CLOUD_TRACKING
