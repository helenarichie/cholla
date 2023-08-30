#ifdef CLOUD_TRACKING

  // STL includes
  #include <stdio.h>

  #include <cstdio>
  #include <fstream>
  #include <vector>

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

void Cloud_Velocity_Reduction(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz, int n_ghost,
                              int n_fields, Real density_cloud_init, Real *mass_cloud,
                              Real *integrand_cloud)
{
  cuda_utilities::AutomaticLaunchParams static const launchParams(Cloud_Reduction_Kernel);

  cuda_utilities::DeviceVector<Real> static dev_mass_cloud(1, true);
  cuda_utilities::DeviceVector<Real> static dev_integrand_cloud(1, true);

  // Initialize host vectors to copy results to
  std::vector<Real> host_mass_cloud(1);
  std::vector<Real> host_integrand_cloud(1);

  // .data() gets device vector pointers
  hipLaunchKernelGGL(Cloud_Reduction_Kernel, launchParams.numBlocks, launchParams.threadsPerBlock, 0, 0, dev_conserved,
                     nx, ny, nz, dx, dy, dz, n_ghost, n_fields, density_cloud_init, dev_mass_cloud.data(),
                     dev_integrand_cloud.data());
  cudaDeviceSynchronize();
  CudaCheckError();

  // Copy result of reductions from device to host
  dev_mass_cloud.cpyDeviceToHost(host_mass_cloud);
  dev_integrand_cloud.cpyDeviceToHost(host_integrand_cloud);

  *mass_cloud      = host_mass_cloud[0];
  *integrand_cloud = host_integrand_cloud[0];
}

void Update_Grid_Frame(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real velocity_x_cloud_avg, Real mass_cloud_tot)
{
  cuda_utilities::AutomaticLaunchParams static const launchParams(Frame_Shift_Kernel);

  // .data() gets device vector pointers
  hipLaunchKernelGGL(Frame_Shift_Kernel, launchParams.numBlocks, launchParams.threadsPerBlock, 0, 0, dev_conserved,
                     nx, ny, nz, n_ghost, n_fields, velocity_x_cloud_avg, mass_cloud_tot);
  cudaDeviceSynchronize();
  CudaCheckError();
}

__global__ void Cloud_Reduction_Kernel(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz,
                                       int n_ghost, int n_fields, Real density_cloud_init,
                                       Real *mass_cloud, Real *integrand_cloud)
{
  int xid, yid, zid, n_cells;
  n_cells = nx * ny * nz;

  Real integrand_stride = 0.0;
  Real mass_stride      = 0.0;

  Real density, velocity_x, mass;

  // Grid stride loop
  for (size_t id = threadIdx.x + blockIdx.x * blockDim.x; id < n_cells; id += blockDim.x * gridDim.x) {
    cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);

    if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
        zid < nz - n_ghost) {
      density    = dev_conserved[id + n_cells * grid_enum::density];
      velocity_x = dev_conserved[id + n_cells * grid_enum::momentum_x] / density;
      mass       = density * dx * dy * dz;
      if ((density * DENSITY_UNIT) > (density_cloud_init / 3)) {
        mass_stride += mass;
        // (Shin et al. (2008) eq. 9)
        integrand_stride += velocity_x * density * dx * dy * dz;
      }
    }
  }
  __syncthreads();

  reduction_utilities::Grid_Reduce_Add(mass_stride, mass_cloud);
  reduction_utilities::Grid_Reduce_Add(integrand_stride, integrand_cloud);
}

__global__ void Frame_Shift_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real velocity_x_cloud_avg, Real mass_cloud_tot)
{
  int xid, yid, zid, n_cells;
  n_cells = nx * ny * nz;

  size_t id = threadIdx.x + blockIdx.x * blockDim.x;
  cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);

  Real density, momentum_x, velocity_x, energy;

  Real energy_kinetic_cloud = 0.5 * mass_cloud_tot * pow(velocity_x_cloud_avg, 2);

  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
  zid < nz - n_ghost) {
    density                                             = dev_conserved[id + n_cells * grid_enum::density];
    momentum_x                                          = dev_conserved[id + n_cells * grid_enum::momentum_x];
    velocity_x                                          = momentum_x / density;
    energy                                              = dev_conserved[id + n_cells * grid_enum::Energy];

    // Apply frame of reference shift
    // printf("pxi: %e\n", momentum_x);
    // printf("pxf: %e\n", (velocity_x - velocity_x_cloud_avg) * density);
    if (id == 80757) {
      printf("vx: %e\n", velocity_x * (KPC / TIME_UNIT));
      printf("vxcl: %e\n", velocity_x_cloud_avg * (KPC / TIME_UNIT));
      printf("pxi: %e\n", momentum_x);
      printf("pxf: %e\n", (velocity_x - velocity_x_cloud_avg) * density);
    }
    // printf("vx: %e\n", velocity_x);
    // printf("vxcl: %e\n", velocity_x_cloud_avg);
    // printf("density: %e\n", density);
    dev_conserved[id + n_cells * grid_enum::momentum_x] = (velocity_x - velocity_x_cloud_avg) * density;
    dev_conserved[id + n_cells * grid_enum::Energy]     = energy - energy_kinetic_cloud;
  }
  __syncthreads();
}

#endif  // CLOUD_TRACKING