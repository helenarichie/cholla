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
                        int n_fields, int dust_enum, Real gamma, Real *mass_cloud, Real *mass_dust_hot, 
                        Real *mass_dust_mixed, Real *mass_dust_cool, Real density_cloud_init)
{
  cuda_utilities::AutomaticLaunchParams static const launchParams(Global_Reduce_Dust_Kernel);

  cuda_utilities::DeviceVector<Real> dev_mass_cloud(1, true);
  cuda_utilities::DeviceVector<Real> dev_mass_dust_hot(1, true);
  cuda_utilities::DeviceVector<Real> dev_mass_dust_mixed(1, true);
  cuda_utilities::DeviceVector<Real> dev_mass_dust_cool(1, true);

  hipLaunchKernelGGL(Global_Reduce_Dust_Kernel, launchParams.get_numBlocks(), launchParams.get_threadsPerBlock(), 0, 0,
                     dev_conserved, nx, ny, nz, dx, dy, dz, n_ghost, n_fields, dust_enum, gamma, dev_mass_cloud.data(),
                     dev_mass_dust_hot.data(), dev_mass_dust_mixed.data(), dev_mass_dust_cool.data(), density_cloud_init);
  cudaDeviceSynchronize();

  *mass_cloud = dev_mass_cloud[0];
  *mass_dust_hot  = dev_mass_dust_hot[0];
  *mass_dust_mixed  = dev_mass_dust_mixed[0];
  *mass_dust_cool  = dev_mass_dust_cool[0];
}

__global__ void Global_Reduce_Dust_Kernel(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz,
                                          int n_ghost, int n_fields, int dust_enum, Real gamma, Real *mass_cloud, Real *mass_dust_hot, Stashed changes
                                          Real *mass_dust_mixed, Real *mass_dust_cool, Real density_cloud_init)
{
  int xid, yid, zid, n_cells;
  n_cells = nx * ny * nz;

  Real density_gas;
  Real mass_cloud_stride = 0.0;
  Real density_dust;
  Real mass_dust_hot_stride = 0.0;
  Real mass_dust_mixed_stride = 0.0;
  Real mass_dust_cool_stride = 0.0;

  for (size_t id = threadIdx.x + blockIdx.x * blockDim.x; id < n_cells; id += blockDim.x * gridDim.x) {
    cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);
    // grid cells
    if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
        zid < nz - n_ghost) {
      density_gas  = dev_conserved[id + n_cells * grid_enum::density];
      density_dust = dev_conserved[id + n_cells * dust_enum];
      
      // convert mass density to number density
      Real const number_density = density_gas * DENSITY_UNIT / (0.6 * MP);

      // Compute the temperature
  #ifdef DE
      Real const gas_energy  = dev_conserved[id + n_cells * grid_enum::GasEnergy];
      Real const temperature = hydro_utilities::Calc_Temp_DE(gas_energy, gamma, number_density);
    #else  // DE is not enabled
      Real const energy     = dev_conserved[id + n_cells * grid_enum::Energy];
      Real const momentum_x = dev_conserved[id + n_cells * grid_enum::momentum_x];
      Real const momentum_y = dev_conserved[id + n_cells * grid_enum::momentum_y];
      Real const momentum_z = dev_conserved[id + n_cells * grid_enum::momentum_z];

      #ifdef MHD
      auto const [magnetic_x, magnetic_y, magnetic_z] =
          mhd::utils::cellCenteredMagneticFields(C.host, id, xid, yid, zid, H.n_cells, H.nx, H.ny);
      Real const temperature =
          hydro_utilities::Calc_Temp_Conserved(energy, density_gas, momentum_x, momentum_y, momentum_z, gamma,
                                              number_density, magnetic_x, magnetic_y, magnetic_z);
      #else   // MHD is not defined
      Real const temperature = hydro_utilities::Calc_Temp_Conserved(energy, density_gas, momentum_x, momentum_y,
                                                                    momentum_z, gamma, number_density);
    #endif  // MHD
  #endif    // DE

      if (temperature >= 5e5) {
        mass_dust_hot_stride += density_dust * dx * dy * dz;
      } else if ((temperature < 5e5) && (temperature >= 2e4)) {
        mass_dust_mixed_stride += density_dust * dx * dy * dz;
      } else if (temperature < 2e4) {
        mass_dust_cool_stride += density_dust * dx * dy * dz;
      }

      if ((density_gas * DENSITY_UNIT) >= (density_cloud_init / 3)) {
        mass_cloud_stride += density_gas * dx * dy * dz;
      }
    }
  }

  __syncthreads();

  reduction_utilities::Grid_Reduce_Add(mass_cloud_stride, mass_cloud);
  reduction_utilities::Grid_Reduce_Add(mass_dust_hot_stride, mass_dust_hot);
  reduction_utilities::Grid_Reduce_Add(mass_dust_mixed_stride, mass_dust_mixed);
  reduction_utilities::Grid_Reduce_Add(mass_dust_cool_stride, mass_dust_cool);
}

  #endif  // GLOBAL_REDUCE_DUST
#endif    // DUST
