/*!
 * \file cloud_tracking.h
 * \author Helena Richie (helenarichie@pitt.edu)
 * \brief Contains declaration for the kernel that does frame of reference tracking to track clouds.
 *
 */

#ifdef CLOUD_TRACKING

  #ifndef CLOUD_TRACKING_H
    #define CLOUD_TRACKING_H

    #include <math.h>

    #include "../global/global.h"
    #include "../utils/gpu.hpp"

/*!
 * \brief Launch the cloud tracking kernel.
 *
 * \param[in,out] dev_conserved The device conserved variable array
 * \param[in] nx Number of cells in the x-direction
 * \param[in] ny Number of cells in the y-direction
 * \param[in] nz Number of cells in the z-direction
 * \param[in] n_ghost Number of ghost cells
 * \param[in] n_fields Number of fields in dev_conserved
 * \param[in] dt Simulation timestep
 * \param[in] gamma Specific heat ratio
 */
void Cloud_Frame_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma);

/*!
 * \brief Compute the mass-averaged cloud velocity and subtract it from the grid.
 *
 * \param[in,out] dev_conserved The device conserved variable array
 * \param[in] nx Number of cells in the x-direction
 * \param[in] ny Number of cells in the y-direction
 * \param[in] nz Number of cells in the z-direction
 * \param[in] n_ghost Number of ghost cells
 * \param[in] n_fields Number of fields in dev_conserved
 * \param[in] dt Simulation timestep
 * \param[in] gamma Specific heat ratio
 */
__global__ void Cloud_Tracking_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt,
                                      Real gamma);

/*!
 * \brief Integrates the density and velocity of all cells meeting the cloud criterion to determine the mass-averaged velocity of the cloud.
 *
 * \param[in] mass_cloud Total cloud mass
 * \param[in] density_cl The density of the cell
 * \param[in] velocity_x_cl The x-velocity of the cell
 *
 * \return Real the mass-averaged cloud velocity (Shin et al. (2008) eq. 8)
 */
__device__ __host__ Real Calc_Cloud_Velocity(Real mass_cloud, Real density_cl, Real velocity_x_cl);

  #endif  // CLOUD_TRACKING_H
#endif    // CLOUD_TRACKING