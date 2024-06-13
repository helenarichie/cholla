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
                 Real dt, Real gamma, Real grain_radius, Real *mass_hot, Real *mass_mixed)
{
  int n_cells = nx * ny * nz;
  int ngrid   = (n_cells + TPB - 1) / TPB;
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);

  cuda_utilities::DeviceVector<Real> dev_mass_mixed(1, true);
  cuda_utilities::DeviceVector<Real> dev_mass_hot(1, true);

  hipLaunchKernelGGL(Dust_Kernel, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost, n_fields, dx, dy, dz,
                     dt, gamma, grain_radius, dev_mass_mixed.data(), dev_mass_hot.data());
  GPU_Error_Check();
  cudaDeviceSynchronize();

  *mass_mixed = dev_mass_mixed[0];
  *mass_hot   = dev_mass_hot[0];
}

__global__ void Dust_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dx, Real dy,
                            Real dz, Real dt, Real gamma, Real grain_radius, Real *mass_hot, Real *mass_mixed)
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
  Real density_gas, density_dust;  // fluid mass densities
  Real number_density;             // gas number density
  Real mu              = 0.6;      // mean molecular weight
  Real sputtered_hot   = 0;
  Real sputtered_mixed = 0;  // mixed and hot-phase sputtered dust masses

  // define integration variables
  Real dd_dt;          // instantaneous rate of change in dust density
  Real dd     = 0;     // change in dust density at current timestep
  Real dd_max = 0.01;  // allowable percentage of dust density increase
  Real dt_sub;         // refined timestep

  if (id_x >= is && id_x < ie && id_y >= js && id_y < je && id_z >= ks && id_z < ke) {
    // get conserved quanitites
    density_gas  = dev_conserved[id + n_cells * grid_enum::density];
    density_dust = dev_conserved[id + n_cells * grid_enum::dust_density];

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
    dd    = dd_dt * dt;                        // change in dust density at current timestep

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

    if (temperature >= 1e6) {
      sputtered_hot += abs(dd * dx * dy * dz);
    } else {
      sputtered_mixed += abs(dd * dx * dy * dz);
    }

    dev_conserved[id + n_cells * grid_enum::dust_density] = density_dust;
  }
  __syncthreads();

  reduction_utilities::Grid_Reduce_Add(sputtered_hot, mass_hot);
  reduction_utilities::Grid_Reduce_Add(sputtered_mixed, mass_mixed);
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
