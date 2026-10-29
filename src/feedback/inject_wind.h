#ifndef INJECT_WIND_H
#define INJECT_WIND_H

#include "../global/global.h"
#include "../utils/gpu.hpp"

void Inject_Wind(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma, 
                 int dx, int dy, int dz, int x_off, int y_off, int z_off, Real xbound, Real ybound, Real zbound);

__global__ void Wind_Injection_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma, 
                                      int dx, int dy, int dz, int x_off, int y_off, int z_off, Real xbound, Real ybound, Real zbound);

#endif