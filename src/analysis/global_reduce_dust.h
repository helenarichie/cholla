#ifdef GLOBAL_REDUCE_DUST
  #ifndef GLOBAL_REDUCE_DUST_CUDA_H
    #define GLOBAL_REDUCE_DUST_CUDA_H

    #include "../global/global.h"
    #include "../utils/gpu.hpp"

void Global_Reduce_Dust(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz, Real zbound, int z_off,
                        int n_ghost, int n_fields, int dust_enum, Real gamma, std::vector<Real> &gas_hot,
                        std::vector<Real> &gas_mixed, std::vector<Real> &gas_cool, std::vector<Real> &dust_hot,
                        std::vector<Real> &dust_mixed, std::vector<Real> &dust_cool, Real density_cloud_init);

__global__ void Global_Reduce_Dust_Kernel(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz,
                                          Real zbound, int z_off, int n_ghost, int n_fields, int dust_enum, Real gamma,
                                          Real *gas_hot, Real *gas_mixed, Real *gas_cool, Real *dust_hot,
                                          Real *dust_mixed, Real *dust_cool, Real density_cloud_init);

  #endif  //  GLOBAL_REDUCE_DUST_CUDA_H
#endif    //  GLOBAL_REDUCE_DUST
