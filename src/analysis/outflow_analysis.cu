#ifdef OUTFLOW_ANALYSIS
  // STL includes
  #include <stdio.h>

  #include <cstdio>
  #include <fstream>
  #include <vector>

  #include "../grid/grid_enum.h"
  #include "../utils/DeviceVector.h"
  #include "../utils/cuda_utilities.h"
  #include "../utils/reduction_utilities.h"
  #include "../analysis/outflow_analysis.h"

void Outflow_Analysis(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz, int n_ghost, 
                      int n_fields, Real density_cloud_init, Real *mass_cloud, Real *mass_dust, Real *rate_cloud, 
                      Real *rate_dust, Real *mass_cloud_bndry, Real *mass_dust_bndry)
{
  cuda_utilities::AutomaticLaunchParams static const launchParams(Outflow_Analysis_Kernel);

  cuda_utilities::DeviceVector<Real> dev_mass_cloud(1, true);
  cuda_utilities::DeviceVector<Real> dev_rate_cloud(1, true);
  cuda_utilities::DeviceVector<Real> dev_mass_cloud_bndry(1, true);
  #ifdef DUST
  cuda_utilities::DeviceVector<Real> dev_mass_dust(1, true);
  cuda_utilities::DeviceVector<Real> dev_rate_dust(1, true);
  cuda_utilities::DeviceVector<Real> dev_mass_dust_bndry(1, true);
  #endif

  // Initialize host vectors to copy results to
  std::vector<Real> host_mass_cloud{0};
  std::vector<Real> host_rate_cloud{0};
  std::vector<Real> host_mass_cloud_bndry{0};
  #ifdef DUST
  std::vector<Real> host_mass_dust{0};
  std::vector<Real> host_rate_dust{0};
  std::vector<Real> host_mass_dust_bndry{0};
  #endif

  hipLaunchKernelGGL(Outflow_Analysis_Kernel, launchParams.numBlocks, launchParams.threadsPerBlock, 0, 0, dev_conserved, 
                     nx, ny, nz, dx, dy, dz, n_ghost, n_fields, density_cloud_init, dev_mass_cloud.data(), dev_mass_dust.data(), 
                     dev_rate_cloud.data(), dev_rate_dust.data(), dev_mass_cloud_bndry.data(), dev_mass_dust_bndry.data());
  cudaDeviceSynchronize();

  dev_mass_cloud.cpyDeviceToHost(host_mass_cloud);
  dev_rate_cloud.cpyDeviceToHost(host_rate_cloud);
  dev_mass_cloud_bndry.cpyDeviceToHost(host_mass_cloud_bndry);
  #ifdef DUST
  dev_mass_dust.cpyDeviceToHost(host_mass_dust);
  dev_rate_dust.cpyDeviceToHost(host_rate_dust);
  dev_mass_dust_bndry.cpyDeviceToHost(host_mass_dust_bndry);
  #endif

  *mass_cloud = host_mass_cloud[0];
  *rate_cloud = host_rate_cloud[0];
  *mass_cloud_bndry = host_mass_cloud_bndry[0];
  #ifdef DUST
  *mass_dust = host_mass_dust[0];
  *rate_dust = host_rate_dust[0];
  *mass_dust_bndry = host_mass_dust_bndry[0];
  #endif
}

__global__ void Outflow_Analysis_Kernel(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz, int n_ghost, 
                                        int n_fields, Real density_cloud_init, Real *mass_cloud, Real *mass_dust, 
                                        Real *rate_cloud, Real *rate_dust, Real *mass_cloud_bndry, Real *mass_dust_bndry)
{
  int xid, yid, zid, n_cells;
  n_cells = nx * ny * nz;

  Real density_gas, velocity_x, velocity_y, velocity_z;
  Real mass_cloud_stride = 0.0;
  Real rate_cloud_stride = 0.0;
  Real mass_cloud_bndry_stride = 0.0;
  #ifdef DUST
  Real density_dust;
  Real mass_dust_stride  = 0.0;
  Real rate_dust_stride = 0.0;
  Real mass_dust_bndry_stride = 0.0;
  #endif  // DUST

  for (size_t id = threadIdx.x + blockIdx.x * blockDim.x; id < n_cells; id += blockDim.x * gridDim.x) {
    cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);

    // grid cells
    if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
        zid < nz - n_ghost) {

      density_gas = dev_conserved[id + n_cells * grid_enum::density];
      #ifdef DUST
      density_dust = dev_conserved[id + n_cells * grid_enum::dust_density];
      #endif  // DUST

      if ((density_gas * DENSITY_UNIT) >= (density_cloud_init / 3)) {
        mass_cloud_stride += density_gas * dx * dy * dz;
      }
      #ifdef DUST
      mass_dust_stride += density_dust * dx * dy * dz;
      #endif  // DUST

      // boundary cells
      if (xid == n_ghost) {
        velocity_x = dev_conserved[id + n_cells * grid_enum::momentum_x] / density_gas;
        if (velocity_x < 0) {
          if ((density_gas * DENSITY_UNIT) >= (density_cloud_init / 3)) {
            mass_cloud_bndry_stride += density_gas * dx * dy * dz;
            rate_cloud_stride += density_gas * std::abs(velocity_x) * dx * dx;
          }
          #ifdef DUST
          mass_dust_bndry_stride += density_dust * dx * dy * dz;
          rate_dust_stride += density_dust * std::abs(velocity_x) * dx * dx;
          #endif  // DUST
        }
      }
      if (xid == (nx - n_ghost - 1)) {
        velocity_x = dev_conserved[id + n_cells * grid_enum::momentum_x] / density_gas;
        if (velocity_x > 0) {
          if ((density_gas * DENSITY_UNIT) >= (density_cloud_init / 3)) {
            mass_cloud_bndry_stride += density_gas * dx * dy * dz;
            rate_cloud_stride += density_gas * std::abs(velocity_x) * dx * dx;
          }
          #ifdef DUST
          mass_dust_bndry_stride += density_dust * dx * dy * dz;
          rate_dust_stride += density_dust * std::abs(velocity_x) * dx * dx;
          #endif  // DUST
        }
      }
      if (yid == n_ghost) {
        velocity_y = dev_conserved[id + n_cells * grid_enum::momentum_y] / density_gas;
        if (velocity_y < 0) {
          if ((density_gas * DENSITY_UNIT) >= (density_cloud_init / 3)) {
            mass_cloud_bndry_stride += density_gas * dx * dy * dz;
            rate_cloud_stride += density_gas * std::abs(velocity_y) * dx * dx;
          }
          #ifdef DUST
          mass_dust_bndry_stride += density_dust * dx * dy * dz;
          rate_dust_stride += density_dust * std::abs(velocity_y) * dx * dx;
          #endif  // DUST
        }
      }
      if (yid == (ny - n_ghost - 1)) {
        velocity_y = dev_conserved[id + n_cells * grid_enum::momentum_y] / density_gas;
        if (velocity_y > 0) {
          if ((density_gas * DENSITY_UNIT) >= (density_cloud_init / 3)) {
            mass_cloud_bndry_stride += density_gas * dx * dy * dz;
            rate_cloud_stride += density_gas * std::abs(velocity_y) * dx * dx;
          }
          #ifdef DUST
          mass_dust_bndry_stride += density_dust * dx * dy * dz;
          rate_dust_stride += density_dust * std::abs(velocity_y) * dx * dx;
          #endif  // DUST
        }
      }
      if (zid == n_ghost) {
        velocity_z = dev_conserved[id + n_cells * grid_enum::momentum_z] / density_gas;
        if (velocity_z < 0) {
          if ((density_gas * DENSITY_UNIT) >= (density_cloud_init / 3)) {
            mass_cloud_bndry_stride += density_gas * dx * dy * dz;
            rate_cloud_stride += density_gas * std::abs(velocity_z) * dx * dx;
          }
          #ifdef DUST
          mass_dust_bndry_stride += density_dust * dx * dy * dz;
          rate_dust_stride += density_dust * std::abs(velocity_z) * dx * dx;
          #endif  // DUST
        }
      }
      if (zid == (nz - n_ghost - 1)) {
        velocity_z = dev_conserved[id + n_cells * grid_enum::momentum_z] / density_gas;
        if (velocity_z > 0) {
          if ((density_gas * DENSITY_UNIT) >= (density_cloud_init / 3)) {
            mass_cloud_bndry_stride += density_gas * dx * dy * dz;
            rate_cloud_stride += density_gas * std::abs(velocity_z) * dx * dx;
          }
          #ifdef DUST
          mass_dust_bndry_stride += density_dust * dx * dy * dz;
          rate_dust_stride += density_dust * std::abs(velocity_z) * dx * dx;
          #endif  // DUST
        }
      }
    }
  }
  __syncthreads();

  reduction_utilities::Grid_Reduce_Add(mass_cloud_stride, mass_cloud);
  reduction_utilities::Grid_Reduce_Add(rate_cloud_stride, rate_cloud);
  reduction_utilities::Grid_Reduce_Add(mass_cloud_bndry_stride, mass_cloud_bndry);
  #ifdef DUST
  reduction_utilities::Grid_Reduce_Add(mass_dust_stride, mass_dust);
  reduction_utilities::Grid_Reduce_Add(rate_dust_stride, rate_dust);
  reduction_utilities::Grid_Reduce_Add(mass_dust_bndry_stride, mass_dust_bndry);
  #endif  // DUST
}

#endif  // OUTFLOW_ANALYSIS