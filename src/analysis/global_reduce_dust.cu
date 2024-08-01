#ifdef DUST
  #ifdef GLOBAL_REDUCE_DUST
    // STL includes
    #include <math.h>
    #include <stdio.h>

    #include <cstdio>
    #include <fstream>
    #include <vector>

    #include "../analysis/global_reduce_dust.h"
    #include "../grid/grid_enum.h"
    #include "../utils/DeviceVector.h"
    #include "../utils/cuda_utilities.h"
    #include "../utils/hydro_utilities.h"
    #include "../utils/reduction_utilities.h"

void Global_Reduce_Dust(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz, int n_ghost,
                        int n_fields, int dust_enum, Real *mass_cloud, Real *mass_dust, Real density_cloud_init)
{
  cuda_utilities::AutomaticLaunchParams static const launchParams(Global_Reduce_Dust_Kernel);

  cuda_utilities::DeviceVector<Real> dev_mass_cloud(1, true);
  cuda_utilities::DeviceVector<Real> dev_mass_dust(1, true);

  hipLaunchKernelGGL(Global_Reduce_Dust_Kernel, launchParams.get_numBlocks(), launchParams.get_threadsPerBlock(), 0, 0,
                     dev_conserved, nx, ny, nz, dx, dy, dz, n_ghost, n_fields, dust_enum, dev_mass_cloud.data(),
                     dev_mass_dust.data(), density_cloud_init);
  cudaDeviceSynchronize();

  *mass_cloud = dev_mass_cloud[0];
  *mass_dust  = dev_mass_dust[0];
}

__global__ void Global_Reduce_Dust_Kernel(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz,
                                          int n_ghost, int n_fields, int dust_enum, Real *mass_cloud, Real *mass_dust,
                                          Real density_cloud_init)
{
  int xid, yid, zid, n_cells;
  n_cells = nx * ny * nz;

  Real density_gas;
  Real mass_cloud_stride = 0.0;
  Real density_dust;
  Real mass_dust_stride = 0.0;

  for (size_t id = threadIdx.x + blockIdx.x * blockDim.x; id < n_cells; id += blockDim.x * gridDim.x) {
    cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);
    // grid cells
    if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
        zid < nz - n_ghost) {
      density_gas  = dev_conserved[id + n_cells * grid_enum::density];
      density_dust = dev_conserved[id + n_cells * dust_enum];

      mass_dust_stride += density_dust * dx * dy * dz;

      if ((density_gas * DENSITY_UNIT) >= (density_cloud_init / 3)) {
        mass_cloud_stride += density_gas * dx * dy * dz;
      }
    }
  }

  __syncthreads();

  reduction_utilities::Grid_Reduce_Add(mass_cloud_stride, mass_cloud);
  reduction_utilities::Grid_Reduce_Add(mass_dust_stride, mass_dust);
}

  #endif  // GLOBAL_REDUCE_DUST
#endif    // DUST
