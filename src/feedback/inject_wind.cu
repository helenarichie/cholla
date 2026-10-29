// STL includes
#include <stdio.h>

#include <cstdio>
#include <fstream>
#include <vector>

// Local includes
#include "../feedback/inject_wind.h"
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../grid/grid3D.h"
#include "../grid/grid_enum.h"
#include "../utils/cuda_utilities.h"
#include "../utils/gpu.hpp"
#include "../utils/hydro_utilities.h"

void Inject_Wind(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma, 
                 int dx, int dy, int dz, int x_off, int y_off, int z_off, Real xbound, Real ybound, Real zbound)
{
  int n_cells = nx * ny * nz;
  int ngrid   = (n_cells + TPB - 1) / TPB;
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(Wind_Injection_Kernel, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost, n_fields, dt, gamma, dx, dy, dz, x_off, y_off, z_off, xbound, ybound, zbound);
  GPU_Error_Check();
}

__global__ void Wind_Injection_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma, 
                                      int dx, int dy, int dz, int x_off, int y_off, int z_off, Real xbound, Real ybound, Real zbound)
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

  if (id_x >= is && id_x < ie && id_y >= js && id_y < je && id_z >= ks && id_z < ke) {

    Real x_pos, y_pos, z_pos;
    // use the subgrid offset and global boundaries to calculate absolute
    // positions on the grid
    x_pos = (x_off + id_x - n_ghost + 0.5) * dx + xbound;
    y_pos = (y_off + id_y - n_ghost + 0.5) * dy + ybound;
    z_pos = (z_off + id_z - n_ghost + 0.5) * dz + zbound;

    // for disk components, calculate polar r
    Real r = sqrt(x_pos * x_pos + y_pos * y_pos + z_pos * z_pos);

    Real mdot = 2.98e5; // M_sun / kyr / kpc^3 (10 M_sun/yr in a 0.2 kpc radius spehre)
    Real edot = 7.43e-2; // SFR * alpha * Edot_sn = 5 * 1 * 3e41 erg/s (over a 0.2 kpc radius sphere) (eq. 6 of Thompson & Heckman 2024)
    Real pdot = mdot * 1.446e-3;  // mdot/volume * v_infty, v_infty - 1414 km/s

    Real kinetic_energy = 1/2 * mdot * pow(1.446e-3, 2);

    // if position is within the cluster scale radius
    if (r < 0.2) {
        dev_conserved[id + n_cells * grid_enum::density] += dt * mdot;
        dev_conserved[id + n_cells * grid_enum::Energy] += dt * edot;
        dev_conserved[id + n_cells * grid_enum::momentum_x] += dt * pdot * x_pos / r;
        dev_conserved[id + n_cells * grid_enum::momentum_y] += dt * pdot * y_pos / r;
        dev_conserved[id + n_cells * grid_enum::momentum_z] += dt * pdot * z_pos / r;
        // update kinetic energy from wind velocity
        dev_conserved[id + n_cells * grid_enum::Energy] += dt * kinetic_energy;
    }
  }
}