#ifdef OUTFLOW_ANALYSIS
  #ifndef OUTFLOW_ANALYSIS_CUDA_H
    #define OUTFLOW_ANALYSIS_CUDA_H

    #include "../global/global.h"
    #include "../utils/gpu.hpp"

void Outflow_Analysis(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz, int n_ghost, int n_fields,
                      Real *mass_cloud, Real *mass_dust, Real density_cloud_init);

__global__ void Outflow_Analysis_Kernel(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz,
                                        int n_ghost, int n_fields, Real *mass_cloud, Real *mass_dust,
                                        Real density_cloud_init);
  #endif  //  OUTFLOW_ANALYSIS_H
#endif    //  OUTFLOW_ANALYSIS
