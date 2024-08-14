/*!
 * \file dust_cuda.cu
 * \author Helena Richie (helenarichie@gmail.com)
 * \brief Contains code that updates the dust density scalar field. The dust_kernel function determines the rate of
 * change of dust density, which is controlled by the sputtering timescale. The sputtering timescale is from the
 * McKinnon et al. (2017) model of dust sputtering, which depends on the cell's gas density and temperature.
 */

#ifdef DUST

  // STL includes
  #include <math.h>
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
  #include "../utils/DeviceVector.h"
  #include "../utils/cuda_utilities.h"
  #include "../utils/gpu.hpp"
  #include "../utils/hydro_utilities.h"
  #include "../utils/reduction_utilities.h"

void Dust_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dx, Real dy, Real dz,
                 Real zbound, int z_off, Real dt, Real gamma, int dust_enum, Real grain_radius,
                 std::vector<Real> &mass_hot, std::vector<Real> &mass_mixed, std::vector<Real> &mass_cool)
{
  int n_cells = nx * ny * nz;
  int ngrid   = (n_cells + TPB - 1) / TPB;
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);

  cuda_utilities::DeviceVector<Real> dev_mass_hot(N_BINS, true);
  cuda_utilities::DeviceVector<Real> dev_mass_mixed(N_BINS, true);
  cuda_utilities::DeviceVector<Real> dev_mass_cool(N_BINS, true);

  hipLaunchKernelGGL(Dust_Kernel, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost, n_fields, dx, dy, dz,
                     zbound, z_off, dt, gamma, dust_enum, grain_radius, dev_mass_hot.data(), dev_mass_mixed.data(),
                     dev_mass_cool.data());
  GPU_Error_Check();
  cudaDeviceSynchronize();

  // write result of GPU grid-wide reduction back to host
  for (int i = 0; i < N_BINS; i++) {
    mass_hot.at(i)   = dev_mass_hot.at(i);
    mass_mixed.at(i) = dev_mass_mixed.at(i);
    mass_cool.at(i)  = dev_mass_cool.at(i);
  }
}

__global__ void Dust_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dx, Real dy,
                            Real dz, Real zbound, int z_off, Real dt, Real gamma, int dust_enum, Real grain_radius,
                            Real *mass_hot, Real *mass_mixed, Real *mass_cool)
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

  Real z_pos = (z_off + id_z - n_ghost + 0.5) * dz + zbound;

  // define physics variables
  Real density_gas, density_dust;      // fluid mass densities
  Real number_density;                 // gas number density
  Real mu                      = 0.6;  // mean molecular weight
  Real sputtered_hot[N_BINS]   = {0};
  Real sputtered_mixed[N_BINS] = {0};  //
  Real sputtered_cool[N_BINS]  = {0};  // hot, mixed, and cool-phase sputtered dust masses

  // define integration variables
  Real dd_dt;          // instantaneous rate of change in dust density
  Real dd     = 0;     // change in dust density at current timestep
  Real dd_max = 0.01;  // allowable percentage of dust density increase
  Real dt_sub;         // refined timestep

  if (id_x >= is && id_x < ie && id_y >= js && id_y < je && id_z >= ks && id_z < ke) {
    // get conserved quanitites
    density_gas  = dev_conserved[id + n_cells * grid_enum::density];
    density_dust = dev_conserved[id + n_cells * dust_enum];

    // convert mass density to number density
    number_density = density_gas * DENSITY_UNIT / (mu * MP);

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

    Real tau_sp = Calc_Sputtering_Timescale(number_density, temperature, grain_radius) /
                  TIME_UNIT;  // sputtering timescale, kyr (sim units)

    dd_dt = Calc_dd_dt(density_dust, tau_sp);  // rate of change in dust density at current timestep
    dd    = dd_dt * dt;                        // change in dust density at current timestepz_off

    // ensure that dust density is not changing too rapidly
    while (dd / density_dust > dd_max) {
      dt_sub = dd_max * density_dust / dd_dt;
      density_dust += dt_sub * dd_dt;
      dt -= dt_sub;
      dd_dt = Calc_dd_dt(density_dust, tau_sp);
      dd    = dt * dd_dt;
    }

    // update dust density
    density_dust += dd;

    // complete phase-wise reduction of sputtered dust mass, binned in vertical chunks 10 x 10 x 1 kpc^3 chunks

    // printf("z_pos: %f \n", z_pos);
    // else if (((z_pos >= 8) && (z_pos < 9)) || (((z_pos >= 11) && (z_pos < 12))))

    // if z is in the disk region (the central 2 kpc of the volume)
    if (abs(z_pos) <= 1) {
      // if sputtered in hot phase
      if (temperature >= 5e5) {
        sputtered_hot[0] += abs(dd * dx * dy * dz);
        // if sputtered in mixed phase
      } else if ((temperature < 5e5) && (temperature >= 2e4)) {
        sputtered_mixed[0] += abs(dd * dx * dy * dz);
        // if sputtered in cool phase
      } else if (temperature < 2e4) {
        sputtered_cool[0] += abs(dd * dx * dy * dz);
      }
      // if z is 1-2 kpc above/below the disk
    } else if ((abs(z_pos) > 1) && (abs(z_pos) <= 2)) {
      if (temperature >= 5e5) {
        sputtered_hot[1] += abs(dd * dx * dy * dz);
      } else if ((temperature < 5e5) && (temperature >= 2e4)) {
        sputtered_mixed[1] += abs(dd * dx * dy * dz);
      } else if (temperature < 2e4) {
        sputtered_cool[1] += abs(dd * dx * dy * dz);
      }
      // if z is 2-3 kpc above/below the disk
    } else if ((abs(z_pos) > 2) && (abs(z_pos) <= 3)) {
      if (temperature >= 5e5) {
        sputtered_hot[2] += abs(dd * dx * dy * dz);
      } else if ((temperature < 5e5) && (temperature >= 2e4)) {
        sputtered_mixed[2] += abs(dd * dx * dy * dz);
      } else if (temperature < 2e4) {
        sputtered_cool[2] += abs(dd * dx * dy * dz);
      }
      // if z is 3-4 kpc above/below the disk
    } else if ((abs(z_pos) > 3) && (abs(z_pos) <= 4)) {
      if (temperature >= 5e5) {
        sputtered_hot[3] += abs(dd * dx * dy * dz);
      } else if ((temperature < 5e5) && (temperature >= 2e4)) {
        sputtered_mixed[3] += abs(dd * dx * dy * dz);
      } else if (temperature < 2e4) {
        sputtered_cool[3] += abs(dd * dx * dy * dz);
      }
      // if z is 4-5 kpc above/below the disk
    } else if ((abs(z_pos) > 4) && (abs(z_pos) <= 5)) {
      if (temperature >= 5e5) {
        sputtered_hot[4] += abs(dd * dx * dy * dz);
      } else if ((temperature < 5e5) && (temperature >= 2e4)) {
        sputtered_mixed[4] += abs(dd * dx * dy * dz);
      } else if (temperature < 2e4) {
        sputtered_cool[4] += abs(dd * dx * dy * dz);
      }
      // if z is 5-6 kpc above/below the disk
    } else if ((abs(z_pos) > 5) && (abs(z_pos) <= 6)) {
      if (temperature >= 5e5) {
        sputtered_hot[5] += abs(dd * dx * dy * dz);
      } else if ((temperature < 5e5) && (temperature >= 2e4)) {
        sputtered_mixed[5] += abs(dd * dx * dy * dz);
      } else if (temperature < 2e4) {
        sputtered_cool[5] += abs(dd * dx * dy * dz);
      }
      // if 6-7 kpc above/below the disk
    } else if ((abs(z_pos) > 6) && (abs(z_pos) <= 7)) {
      if (temperature >= 5e5) {
        sputtered_hot[6] += abs(dd * dx * dy * dz);
      } else if ((temperature < 5e5) && (temperature >= 2e4)) {
        sputtered_mixed[6] += abs(dd * dx * dy * dz);
      } else if (temperature < 2e4) {
        sputtered_cool[6] += abs(dd * dx * dy * dz);
      }
      // if 7-8 kpc above/below the disk
    } else if ((abs(z_pos) > 7) && (abs(z_pos) <= 8)) {
      if (temperature >= 5e5) {
        sputtered_hot[7] += abs(dd * dx * dy * dz);
      } else if ((temperature < 5e5) && (temperature >= 2e4)) {
        sputtered_mixed[7] += abs(dd * dx * dy * dz);
      } else if (temperature < 2e4) {
        sputtered_cool[7] += abs(dd * dx * dy * dz);
      }
      // if z is 8-9 kpc above/below the disk
    } else if ((abs(z_pos) > 8) && (abs(z_pos) <= 9)) {
      if (temperature >= 5e5) {
        sputtered_hot[8] += abs(dd * dx * dy * dz);
      } else if ((temperature < 5e5) && (temperature >= 2e4)) {
        sputtered_mixed[8] += abs(dd * dx * dy * dz);
      } else if (temperature < 2e4) {
        sputtered_cool[8] += abs(dd * dx * dy * dz);
      }
      // if z is 9-10 kpc above/below the disk
    } else if ((abs(z_pos) > 9) && (abs(z_pos) <= 10)) {
      if (temperature >= 5e5) {
        sputtered_hot[9] += abs(dd * dx * dy * dz);
      } else if ((temperature < 5e5) && (temperature >= 2e4)) {
        sputtered_mixed[9] += abs(dd * dx * dy * dz);
      } else if (temperature < 2e4) {
        sputtered_cool[9] += abs(dd * dx * dy * dz);
      }
    }

    dev_conserved[id + n_cells * dust_enum] = density_dust;
  }
  __syncthreads();

  // perform GPU grid-wide reduction and store result in mass_hot/mass_mixed/mass_cool
  for (int i = 0; i < N_BINS; i++) {
    reduction_utilities::Grid_Reduce_Add(sputtered_hot[i], &mass_hot[i]);
    reduction_utilities::Grid_Reduce_Add(sputtered_mixed[i], &mass_mixed[i]);
    reduction_utilities::Grid_Reduce_Add(sputtered_cool[i], &mass_cool[i]);
  }
}

// McKinnon et al. (2017) sputtering timescale
__device__ __host__ Real Calc_Sputtering_Timescale(Real number_density, Real temperature, Real grain_radius)
{
  Real a             = grain_radius;  // dust grain size in units of 0.1 micrometers
  Real temperature_0 = 2e6;           // temp above which the sputtering rate is ~constant in K
  Real omega         = 2.5;           // controls the low-temperature scaling of the sputtering rate
  Real A             = 5.3618e15;     // 0.17 Gyr in s

  number_density /= (6e-4);  // gas number density in units of 10^-27 g/cm^3

  // sputtering timescale, s
  Real tau_sp = A * (a / number_density) * (pow(temperature_0 / temperature, omega) + 1);

  return tau_sp;
}

// McKinnon et al. (2017) sputtering model
__device__ __host__ Real Calc_dd_dt(Real density_dust, Real tau_sp) { return -density_dust / (tau_sp / 3); }

#endif  // DUST
