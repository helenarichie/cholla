#ifdef GLOBAL_REDUCE_DUST
  #ifndef GLOBAL_REDUCE_DUST_CUDA_H
    #define GLOBAL_REDUCE_DUST_CUDA_H

    #include "../global/global.h"
    #include "../utils/gpu.hpp"

void Global_Reduce_Dust(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz, int n_ghost,
                        int n_fields, int dust_enum, Real *mass_cloud, Real *mass_dust, Real density_cloud_init);

__global__ void Global_Reduce_Dust_Kernel(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz,
                                          int n_ghost, int n_fields, int dust_enum, Real *mass_cloud, Real *mass_dust,
                                          Real density_cloud_init);
  #endif  //  GLOBAL_REDUCE_DUST_CUDA_H
#endif    //  GLOBAL_REDUCE_DUST
