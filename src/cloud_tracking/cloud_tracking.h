/*!
 * \file cloud_tracking.h
 * \author Helena Richie (helenarichie@pitt.edu)
 * \brief Contains declaration for the kernel that does frame of reference tracking to track clouds.
 *
 */
#ifdef CLOUD_TRACKING
  #ifndef CLOUD_TRACKING_CUDA_H
    #define CLOUD_TRACKING_CUDA_H

    #include <math.h>

    #include "../global/global.h"
    #include "../utils/gpu.hpp"

void Cloud_Frame_Update(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz, int n_ghost,
                        int n_fields, Real dt, Real gamma, Real density_cloud_init, Real *integrand,
                        Real *density_cloud_tot, Real *mass_cloud_tot);

__global__ void Cloud_Tracking_Kernel(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz,
                                      int n_ghost, int n_fields, Real dt, Real gamma, Real density_cloud_init,
                                      Real *integrand_cloud, Real *density_cloud, Real *mass_cloud);

void Update_Grid_Velocities(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma,
                            Real velocity_cloud, Real density_cloud_tot, Real mass_cloud_tot);

__global__ void Velocity_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt,
                                Real gamma, Real velocity_cloud, Real density_cloud_tot, Real mass_cloud_tot);

  #endif  // CLOUD_TRACKING_CUDA_H
#endif    // CLOUD_TRACKING