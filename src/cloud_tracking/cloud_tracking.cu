#ifdef CLOUD_TRACKING

// STL includes
#include <stdio.h>
#include <cstdio>
#include <fstream>
#include <vector>

#include "../cloud_tracking/cloud_tracking.h"
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../grid/grid3D.h"
#include "../grid/grid_enum.h"
#include "../utils/DeviceVector.h"
#include "../utils/cuda_utilities.h"
#include "../utils/gpu.hpp"
#include "../utils/hydro_utilities.h"
#include "../utils/reduction_utilities.h"

void Cloud_Velocity_Reduction(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz, int n_ghost, 
                              int n_fields, Real dt, Real gamma, Real density_cloud_init, Real *integrand, 
                              Real *density_cloud_tot, Real *mass_cloud_tot) 
{
    cuda_utilities::AutomaticLaunchParams static const launchParams(Cloud_Reduction_Kernel);    

    // Allocate the device memory to store the results of the reductions. "true" argument initializes arrays to zero
    cuda_utilities::DeviceVector<Real> static dev_integrand(1, true);
    cuda_utilities::DeviceVector<Real> static dev_density_cloud_tot(1, true);
    cuda_utilities::DeviceVector<Real> static dev_mass_cloud_tot(1, true);

    // Initialize host vectors to copy results to
    std::vector<Real> host_integrand(1);
    std::vector<Real> host_density_cloud_tot(1);
    std::vector<Real> host_mass_cloud_tot(1);

    // .data() gets device vector pointers
    hipLaunchKernelGGL(Cloud_Reduction_Kernel, launchParams.numBlocks, launchParams.threadsPerBlock, 0, 0, dev_conserved,
                       nx, ny, nz, dx, dy, dz, n_ghost, n_fields, dt, gamma, density_cloud_init, dev_integrand.data(),
                       dev_density_cloud_tot.data(), dev_mass_cloud_tot.data());
    cudaDeviceSynchronize();
    CudaCheckError();

    // Copy result of reductions from device to host
    dev_integrand.cpyDeviceToHost(host_integrand);
    dev_density_cloud_tot.cpyDeviceToHost(host_density_cloud_tot);
    dev_mass_cloud_tot.cpyDeviceToHost(host_mass_cloud_tot);

    *integrand         = host_integrand[0];
    *density_cloud_tot = host_density_cloud_tot[0];
    *mass_cloud_tot    = host_mass_cloud_tot[0];
}

__global__ void Cloud_Reduction_Kernel(Real *dev_conserved, int nx, int ny, int nz, Real dx, Real dy, Real dz, 
                                       int n_ghost, int n_fields, Real dt, Real gamma, Real density_cloud_init, 
                                       Real *integrand, Real *density_cloud, Real *mass_cloud) 
{
    int xid, yid, zid, n_cells;
    n_cells = nx * ny * nz;

    Real density_stride = 0.0;
    Real velocity_x_stride = 0.0;
    Real mass_stride = 0.0;
    int counter = 0;

    Real density, velocity_x, mass;

    // Grid stride loop
    for (size_t id = threadIdx.x + blockIdx.x * blockDim.x; id < n_cells; id += blockDim.x * gridDim.x) {
    
        cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);

        if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
        zid < nz - n_ghost) {
            density    = dev_conserved[id + n_cells * grid_enum::density];
            velocity_x = dev_conserved[id + n_cells * grid_enum::momentum_x] / density;
            mass       = density * dx * dy * dz;
            if (density > (1 / 3 * (density_cloud_init / DENSITY_UNIT))) {
                // printf("inside if statement: %d %e\n", id, mass);
                counter += 1;
                density_stride += density;
                velocity_x_stride += velocity_x;
                mass_stride += mass;
            }
        }
    }
    // __syncthreads();

    // printf("velocity_x_stride: %d %e\n", threadIdx.x, velocity_x_stride);
    // printf("mass_stride: %d %e\n", threadIdx.x, mass_stride);
    printf("counter: %i %i %i --> %i %e\n", gridDim.x, blockIdx.x, threadIdx.x, counter, velocity_x_stride);

    reduction_utilities::Grid_Reduce_Add(density_stride * velocity_x_stride, integrand);
    reduction_utilities::Grid_Reduce_Add(density_stride, density_cloud);
    reduction_utilities::Grid_Reduce_Add(mass_stride, mass_cloud);

}

#endif // CLOUD_TRACKING