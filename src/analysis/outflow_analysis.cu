#ifdef DUST
  #ifdef OUTFLOW_ANALYSIS
    // STL includes
    #include <math.h>
    #include <stdio.h>

    #include <cstdio>
    #include <fstream>
    #include <vector>

    #include "../analysis/outflow_analysis.h"
    #include "../grid/grid_enum.h"
    #include "../utils/DeviceVector.h"
    #include "../utils/cuda_utilities.h"
    #include "../utils/hydro_utilities.h"
    #include "../utils/reduction_utilities.h"

void Outflow_Analysis(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz, int n_ghost, int n_fields,
                      Real *mass_cloud, Real *mass_dust, Real density_cloud_init)
{
  cuda_utilities::AutomaticLaunchParams static const launchParams(Outflow_Analysis_Kernel);

  cuda_utilities::DeviceVector<Real> dev_mass_cloud(1);
  cuda_utilities::DeviceVector<Real> dev_mass_dust(1);

  hipLaunchKernelGGL(Outflow_Analysis_Kernel, launchParams.numBlocks, launchParams.threadsPerBlock, 0, 0, dev_conserved,
                     nx, ny, nz, dx, dy, dz, n_ghost, n_fields, dev_mass_cloud.data(), dev_mass_dust.data(),
                     density_cloud_init);
  cudaDeviceSynchronize();

  *mass_cloud = dev_mass_cloud[0];
  *mass_dust  = dev_mass_dust[0];
}

__global__ void Outflow_Analysis_Kernel(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz,
                                        int n_ghost, int n_fields, Real *mass_cloud, Real *mass_dust,
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
      density_dust = dev_conserved[id + n_cells * grid_enum::dust_density];

      mass_dust_stride += density_dust * dx * dy * dz;

      if ((density_gas * DENSITY_UNIT) >= (density_cloud_init / 30)) {
        mass_cloud_stride += density_gas * dx * dy * dz;
      }
    }
  }

  __syncthreads();

  reduction_utilities::Grid_Reduce_Add(mass_cloud_stride, mass_cloud);
  reduction_utilities::Grid_Reduce_Add(mass_dust_stride, mass_dust);

  /*
  // int id, xid, yid, zid, gid;

  // calculate ghost cell ID and i,j,k in GPU grid
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  int isize, jsize;

  // +x boundary first
  isize = n_ghost;
  jsize = ny;

  // not true i,j,k but relative i,j,k in the GPU grid
  zid = id / (isize * jsize);
  yid = (id - zid * isize * jsize) / isize;
  xid = id - zid * isize * jsize - yid * isize;

  // map thread id to ghost cell id
  xid += nx - n_ghost;  // +x boundary
  int gid = xid + yid * nx + zid * nx * ny;

  // boundary cells
  // -x boundary
  if (xid == n_ghost && xid < nx && yid < ny && zid < nz) {
    velocity_x = dev_conserved[gid + n_cells * grid_enum::momentum_x] / density_gas;
    density_gas = dev_conserved[gid + n_cells * grid_enum::density];
    #ifdef DUST
    density_dust = dev_conserved[gid + n_cells * grid_enum::dust_density];
    #endif
    if (velocity_x < 0) {
      if ((density_gas * DENSITY_UNIT) >= (density_cloud_init / 3)) {
        printf("-x %e/n", density_gas * DENSITY_UNIT);
        mass_cloud_bndry_stride += density_gas * dx * dy * dz;
        rate_cloud_stride += density_gas * std::abs(velocity_x) * dx * dx;
      }
      #ifdef DUST
      mass_dust_bndry_stride += density_dust * dx * dy * dz;
      rate_dust_stride += density_dust * std::abs(velocity_x) * dx * dx;
      #endif  // DUST
    }
  }
  // +x boundary
  if (xid == nx - n_ghost && xid < nx && yid < ny && zid < nz) {
    velocity_x = dev_conserved[gid + n_cells * grid_enum::momentum_x] / density_gas;
    density_gas = dev_conserved[gid + n_cells * grid_enum::density];
    #ifdef DUST
    density_dust = dev_conserved[gid + n_cells * grid_enum::dust_density];
    #endif
    if (velocity_x > 0) {
      if ((density_gas * DENSITY_UNIT) >= (density_cloud_init / 3)) {
        printf("+x %e/n", density_gas * DENSITY_UNIT);
        mass_cloud_bndry_stride += density_gas * dx * dy * dz;
        rate_cloud_stride += density_gas * std::abs(velocity_x) * dx * dx;
      }
      #ifdef DUST
      mass_dust_bndry_stride += density_dust * dx * dy * dz;
      rate_dust_stride += density_dust * std::abs(velocity_x) * dx * dx;
      #endif  // DUST
    }
  }

  // +y boundary next
  isize = nx;
  jsize = n_ghost;
  // ksize = nz;

  // not true i,j,k but relative i,j,k
  zid = id / (isize * jsize);
  yid = (id - zid * isize * jsize) / isize;
  xid = id - zid * isize * jsize - yid * isize;

  // map thread id to ghost cell id
  yid += ny - n_ghost;
  gid = xid + yid * nx + zid * nx * ny;

  // -y boundary
  if (xid < nx && yid == n_ghost && yid < ny && zid < nz) {
    velocity_y = dev_conserved[gid + n_cells * grid_enum::momentum_y] / density_gas;
    density_gas = dev_conserved[gid + n_cells * grid_enum::density];
    #ifdef DUST
    density_dust = dev_conserved[gid + n_cells * grid_enum::dust_density];
    #endif
    if (velocity_y < 0) {
      if ((density_gas * DENSITY_UNIT) >= (density_cloud_init / 3)) {
        // printf("-y cloud %e/n", density_gas * DENSITY_UNIT);
        printf("yid: %d\n", yid);
        mass_cloud_bndry_stride += density_gas * dx * dy * dz;
        rate_cloud_stride += density_gas * std::abs(velocity_y) * dx * dx;
      }
      #ifdef DUST
      // printf("-y dust %e/n", density_dust * DENSITY_UNIT);
      mass_dust_bndry_stride += density_dust * dx * dy * dz;
      rate_dust_stride += density_dust * std::abs(velocity_y) * dx * dx;
      #endif  // DUST
    }
  }
  // +y boundary
  if (xid < nx && yid == ny - n_ghost && yid < ny && zid < nz) {
    velocity_y = dev_conserved[gid + n_cells * grid_enum::momentum_y] / density_gas;
    density_gas = dev_conserved[gid + n_cells * grid_enum::density];
    #ifdef DUST
    density_dust = dev_conserved[gid + n_cells * grid_enum::dust_density];
    #endif
    if (velocity_y > 0) {
      if ((density_gas * DENSITY_UNIT) >= (density_cloud_init / 3)) {
        // printf("+y %e/n", density_gas * DENSITY_UNIT);
        printf("yid: %d\n", yid);
        printf("yid: %d\n", ny_real);
        mass_cloud_bndry_stride += density_gas * dx * dy * dz;
        rate_cloud_stride += density_gas * std::abs(velocity_y) * dx * dx;
      }
      #ifdef DUST
      mass_dust_bndry_stride += density_dust * dx * dy * dz;
      rate_dust_stride += density_dust * std::abs(velocity_y) * dx * dx;
      #endif  // DUST
    }
  }

  isize = nx;
  jsize = ny;
  // ksize = n_ghost;

  // not true i,j,k but relative i,j,k
  zid = id / (isize * jsize);
  yid = (id - zid * isize * jsize) / isize;
  xid = id - zid * isize * jsize - yid * isize;

  // map thread id to ghost cell id
  zid += nz - n_ghost;
  gid = xid + yid * nx + zid * nx * ny;

  // -z boundary
  if (xid < nx && yid < ny && zid == n_ghost && zid < nz) {
    velocity_z = dev_conserved[gid + n_cells * grid_enum::momentum_z] / density_gas;
    density_gas = dev_conserved[gid + n_cells * grid_enum::density];
    #ifdef DUST
    density_dust = dev_conserved[gid + n_cells * grid_enum::dust_density];
    #endif
    if (velocity_z < 0) {
      if ((density_gas * DENSITY_UNIT) >= (density_cloud_init / 3)) {
        printf("-z %e/n", density_gas * DENSITY_UNIT);
        mass_cloud_bndry_stride += density_gas * dx * dy * dz;
        rate_cloud_stride += density_gas * std::abs(velocity_z) * dx * dx;
      }
      #ifdef DUST
      mass_dust_bndry_stride += density_dust * dx * dy * dz;
      rate_dust_stride += density_dust * std::abs(velocity_z) * dx * dx;
      #endif  // DUST
    }
  }
  // +z bondary
  if (xid < nx && yid < ny && zid == nz - n_ghost && zid < nz) {
    velocity_z = dev_conserved[gid + n_cells * grid_enum::momentum_z] / density_gas;
    density_gas = dev_conserved[gid + n_cells * grid_enum::density];
    #ifdef DUST
    density_dust = dev_conserved[gid + n_cells * grid_enum::dust_density];
    #endif
    if (velocity_z > 0) {
      if ((density_gas * DENSITY_UNIT) >= (density_cloud_init / 3)) {
        printf("+z %e/n", density_gas * DENSITY_UNIT);
        mass_cloud_bndry_stride += density_gas * dx * dy * dz;
        rate_cloud_stride += density_gas * std::abs(velocity_z) * dx * dx;
      }
      #ifdef DUST
      mass_dust_bndry_stride += density_dust * dx * dy * dz;
      rate_dust_stride += density_dust * std::abs(velocity_z) * dx * dx;
      #endif  // DUST
    }
  }
  */
}

  #endif  // OUTFLOW_ANALYSIS
#endif    // DUST
