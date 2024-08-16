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

// std::vector<Real> &gas_hot, std::vector<Real> &gas_mixed, std::vector<Real> &gas_cool, std::vector<Real> &dust_hot,
// std::vector<Real> &dust_mixed, std::vector<Real> &dust_cool

void Global_Reduce_Dust(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz, Real zbound, int z_off,
                        int n_ghost, int n_fields, int dust_enum, Real gamma, std::vector<Real> &gas_hot,
                        std::vector<Real> &gas_mixed, std::vector<Real> &gas_cool, std::vector<Real> &dust_hot,
                        std::vector<Real> &dust_mixed, std::vector<Real> &dust_cool, Real density_cloud_init)
{
  cuda_utilities::AutomaticLaunchParams static const launchParams(Global_Reduce_Dust_Kernel);

  cuda_utilities::DeviceVector<Real> dev_gas_hot(N_BINS, true);
  cuda_utilities::DeviceVector<Real> dev_gas_mixed(N_BINS, true);
  cuda_utilities::DeviceVector<Real> dev_gas_cool(N_BINS, true);
  cuda_utilities::DeviceVector<Real> dev_dust_hot(N_BINS, true);
  cuda_utilities::DeviceVector<Real> dev_dust_mixed(N_BINS, true);
  cuda_utilities::DeviceVector<Real> dev_dust_cool(N_BINS, true);

  hipLaunchKernelGGL(Global_Reduce_Dust_Kernel, launchParams.get_numBlocks(), launchParams.get_threadsPerBlock(), 0, 0,
                     dev_conserved, nx, ny, nz, dx, dy, dz, zbound, z_off, n_ghost, n_fields, dust_enum, gamma,
                     dev_gas_hot.data(), dev_gas_mixed.data(), dev_gas_cool.data(), dev_dust_hot.data(),
                     dev_dust_mixed.data(), dev_dust_cool.data(), density_cloud_init);
  GPU_Error_Check();
  cudaDeviceSynchronize();

  // write result of GPU grid-wide reduction back to host
  for (int i = 0; i < N_BINS; i++) {
    gas_hot.at(i)    = dev_gas_hot.at(i);
    gas_mixed.at(i)  = dev_gas_mixed.at(i);
    gas_cool.at(i)   = dev_gas_cool.at(i);
    dust_hot.at(i)   = dev_dust_hot.at(i);
    dust_mixed.at(i) = dev_dust_mixed.at(i);
    dust_cool.at(i)  = dev_dust_cool.at(i);
  }
}

__global__ void Global_Reduce_Dust_Kernel(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz,
                                          Real zbound, int z_off, int n_ghost, int n_fields, int dust_enum, Real gamma,
                                          Real *gas_hot, Real *gas_mixed, Real *gas_cool, Real *dust_hot,
                                          Real *dust_mixed, Real *dust_cool, Real density_cloud_init)
{
  int xid, yid, zid, n_cells;
  n_cells = nx * ny * nz;

  Real density_gas, density_dust;
  Real gas_hot_stride[N_BINS]    = {0.0};
  Real gas_mixed_stride[N_BINS]  = {0.0};
  Real gas_cool_stride[N_BINS]   = {0.0};
  Real dust_hot_stride[N_BINS]   = {0.0};
  Real dust_mixed_stride[N_BINS] = {0.0};
  Real dust_cool_stride[N_BINS]  = {0.0};

  for (size_t id = threadIdx.x + blockIdx.x * blockDim.x; id < n_cells; id += blockDim.x * gridDim.x) {
    cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);

    Real z_pos = (z_off + zid - n_ghost + 0.5) * dz + zbound;
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

      if (abs(z_pos) <= 1) {
        // if sputtered in hot phase
        if (temperature >= 5e5) {
          dust_hot_stride[0] += abs(density_dust * dx * dy * dz);
          gas_hot_stride[0] += abs(density_gas * dx * dy * dz);
          // if sputtered in mixed phase
        } else if ((temperature < 5e5) && (temperature >= 2e4)) {
          dust_mixed_stride[0] += abs(density_dust * dx * dy * dz);
          gas_mixed_stride[0] += abs(density_gas * dx * dy * dz);
          // if sputtered in cool phase
        } else if (temperature < 2e4) {
          dust_cool_stride[0] += abs(density_dust * dx * dy * dz);
          gas_cool_stride[0] += abs(density_gas * dx * dy * dz);
        }
        // if z is 1-2 kpc above/below the disk
      } else if ((abs(z_pos) > 1) && (abs(z_pos) <= 2)) {
        if (temperature >= 5e5) {
          dust_hot_stride[1] += abs(density_dust * dx * dy * dz);
          gas_hot_stride[1] += abs(density_gas * dx * dy * dz);
        } else if ((temperature < 5e5) && (temperature >= 2e4)) {
          dust_mixed_stride[1] += abs(density_dust * dx * dy * dz);
          gas_mixed_stride[1] += abs(density_gas * dx * dy * dz);
        } else if (temperature < 2e4) {
          dust_cool_stride[1] += abs(density_dust * dx * dy * dz);
          gas_cool_stride[1] += abs(density_gas * dx * dy * dz);
        }
        // if z is 2-3 kpc above/below the disk
      } else if ((abs(z_pos) > 2) && (abs(z_pos) <= 3)) {
        if (temperature >= 5e5) {
          dust_hot_stride[2] += abs(density_dust * dx * dy * dz);
          gas_hot_stride[2] += abs(density_gas * dx * dy * dz);
        } else if ((temperature < 5e5) && (temperature >= 2e4)) {
          dust_mixed_stride[2] += abs(density_dust * dx * dy * dz);
          gas_mixed_stride[2] += abs(density_gas * dx * dy * dz);
        } else if (temperature < 2e4) {
          dust_cool_stride[2] += abs(density_dust * dx * dy * dz);
          gas_cool_stride[2] += abs(density_gas * dx * dy * dz);
        }
        // if z is 3-4 kpc above/below the disk
      } else if ((abs(z_pos) > 3) && (abs(z_pos) <= 4)) {
        if (temperature >= 5e5) {
          dust_hot_stride[3] += abs(density_dust * dx * dy * dz);
          gas_hot_stride[3] += abs(density_gas * dx * dy * dz);
        } else if ((temperature < 5e5) && (temperature >= 2e4)) {
          dust_mixed_stride[3] += abs(density_dust * dx * dy * dz);
          gas_mixed_stride[3] += abs(density_gas * dx * dy * dz);
        } else if (temperature < 2e4) {
          dust_cool_stride[3] += abs(density_dust * dx * dy * dz);
          gas_cool_stride[3] += abs(density_gas * dx * dy * dz);
        }
        // if z is 4-5 kpc above/below the disk
      } else if ((abs(z_pos) > 4) && (abs(z_pos) <= 5)) {
        if (temperature >= 5e5) {
          dust_hot_stride[4] += abs(density_dust * dx * dy * dz);
          gas_hot_stride[4] += abs(density_gas * dx * dy * dz);
        } else if ((temperature < 5e5) && (temperature >= 2e4)) {
          dust_mixed_stride[4] += abs(density_dust * dx * dy * dz);
          gas_mixed_stride[4] += abs(density_gas * dx * dy * dz);
        } else if (temperature < 2e4) {
          dust_cool_stride[4] += abs(density_dust * dx * dy * dz);
          gas_cool_stride[4] += abs(density_gas * dx * dy * dz);
        }
        // if z is 5-6 kpc above/below the disk
      } else if ((abs(z_pos) > 5) && (abs(z_pos) <= 6)) {
        if (temperature >= 5e5) {
          dust_hot_stride[5] += abs(density_dust * dx * dy * dz);
          gas_hot_stride[5] += abs(density_gas * dx * dy * dz);
        } else if ((temperature < 5e5) && (temperature >= 2e4)) {
          dust_mixed_stride[5] += abs(density_dust * dx * dy * dz);
          gas_mixed_stride[5] += abs(density_gas * dx * dy * dz);
        } else if (temperature < 2e4) {
          dust_cool_stride[5] += abs(density_dust * dx * dy * dz);
          gas_cool_stride[5] += abs(density_gas * dx * dy * dz);
        }
        // if 6-7 kpc above/below the disk
      } else if ((abs(z_pos) > 6) && (abs(z_pos) <= 7)) {
        if (temperature >= 5e5) {
          dust_hot_stride[6] += abs(density_dust * dx * dy * dz);
          gas_hot_stride[6] += abs(density_gas * dx * dy * dz);
        } else if ((temperature < 5e5) && (temperature >= 2e4)) {
          dust_mixed_stride[6] += abs(density_dust * dx * dy * dz);
          gas_mixed_stride[6] += abs(density_gas * dx * dy * dz);
        } else if (temperature < 2e4) {
          dust_cool_stride[6] += abs(density_dust * dx * dy * dz);
          gas_cool_stride[6] += abs(density_gas * dx * dy * dz);
        }
        // if 7-8 kpc above/below the disk
      } else if ((abs(z_pos) > 7) && (abs(z_pos) <= 8)) {
        if (temperature >= 5e5) {
          dust_hot_stride[7] += abs(density_dust * dx * dy * dz);
          gas_hot_stride[7] += abs(density_gas * dx * dy * dz);
        } else if ((temperature < 5e5) && (temperature >= 2e4)) {
          dust_mixed_stride[7] += abs(density_dust * dx * dy * dz);
          gas_mixed_stride[7] += abs(density_gas * dx * dy * dz);
        } else if (temperature < 2e4) {
          dust_cool_stride[7] += abs(density_dust * dx * dy * dz);
          gas_cool_stride[7] += abs(density_gas * dx * dy * dz);
        }
        // if z is 8-9 kpc above/below the disk
      } else if ((abs(z_pos) > 8) && (abs(z_pos) <= 9)) {
        if (temperature >= 5e5) {
          dust_hot_stride[8] += abs(density_dust * dx * dy * dz);
          gas_hot_stride[8] += abs(density_gas * dx * dy * dz);
        } else if ((temperature < 5e5) && (temperature >= 2e4)) {
          dust_mixed_stride[8] += abs(density_dust * dx * dy * dz);
          gas_mixed_stride[8] += abs(density_gas * dx * dy * dz);
        } else if (temperature < 2e4) {
          dust_cool_stride[8] += abs(density_dust * dx * dy * dz);
          gas_cool_stride[8] += abs(density_gas * dx * dy * dz);
        }
        // if z is 9-10 kpc above/below the disk
      } else if ((abs(z_pos) > 9) && (abs(z_pos) <= 10)) {
        if (temperature >= 5e5) {
          dust_hot_stride[9] += abs(density_dust * dx * dy * dz);
          gas_hot_stride[9] += abs(density_gas * dx * dy * dz);
        } else if ((temperature < 5e5) && (temperature >= 2e4)) {
          dust_mixed_stride[9] += abs(density_dust * dx * dy * dz);
          gas_mixed_stride[9] += abs(density_gas * dx * dy * dz);
        } else if (temperature < 2e4) {
          dust_cool_stride[9] += abs(density_dust * dx * dy * dz);
          gas_cool_stride[9] += abs(density_gas * dx * dy * dz);
        }
      }
    }
  }

  __syncthreads();

  // perform GPU grid-wide reduction and store result in mass_hot/mass_mixed/mass_cool
  for (int i = 0; i < N_BINS; i++) {
    reduction_utilities::Grid_Reduce_Add(gas_hot_stride[i], &gas_hot[i]);
    reduction_utilities::Grid_Reduce_Add(gas_mixed_stride[i], &gas_mixed[i]);
    reduction_utilities::Grid_Reduce_Add(gas_cool_stride[i], &gas_cool[i]);
    reduction_utilities::Grid_Reduce_Add(dust_hot_stride[i], &dust_hot[i]);
    reduction_utilities::Grid_Reduce_Add(dust_mixed_stride[i], &dust_mixed[i]);
    reduction_utilities::Grid_Reduce_Add(dust_cool_stride[i], &dust_cool[i]);
  }
}

  #endif  // GLOBAL_REDUCE_DUST
#endif    // DUST
