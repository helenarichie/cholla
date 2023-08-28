#ifdef CLOUD_TRACKING
  #ifndef CLOUD_TRACKING_CUDA_H
    #define CLOUD_TRACKING_CUDA_H

    #include "../global/global.h"
    #include "../utils/gpu.hpp"

void Cloud_Velocity_Reduction(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz, int n_ghost,
                              int n_fields, Real dt, Real gamma, Real density_cloud_init, Real *mass_cloud,
                              Real *integrand_cloud);

__global__ void Cloud_Reduction_Kernel(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz,
                                       int n_ghost, int n_fields, Real dt, Real gamma, Real density_cloud_init,
                                       Real *mass_cloud, Real *integrand_cloud);

  #endif  // CLOUD_TRACKING_H
#endif    // CLOUD_TRACKING