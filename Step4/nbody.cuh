/**
 * @file      nbody.cuh
 *
 * @author    Pavel Kratochvil \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            xkrato61@fit.vutbr.cz
 *
 * @brief     PCG Assignment 1
 *
 * @version   2024
 *
 * @date      10 November  2024 \n
 */

#ifndef NBODY_CUH
#define NBODY_CUH

#include <cuda_runtime.h>

#include "h5Helper.h"

/**
 * Particles data structure
 */
struct Particles {
  float* position_x;
  float* position_y;
  float* position_z;

  float* velocity_x;
  float* velocity_y;
  float* velocity_z;

  float* mass;
};

/**
 * CUDA kernel to calculate new particles velocity and position
 * @param pIn  - particles in
 * @param pOut - particles out
 * @param N    - Number of particles
 * @param dt   - Size of the time step
 */
__global__ void calculateVelocity(Particles pIn, Particles pOut, const unsigned N, float dt);

/**
 * CUDA kernel to calculate particles center of mass
 * @param p    - particles
 * @param com  - pointer to a center of mass
 * @param lock - pointer to a user-implemented lock
 * @param N    - Number of particles
 */
__global__ void centerOfMass(Particles p, float4* com, int* lock, const unsigned N);

/**
 * CPU implementation of the Center of Mass calculation
 * @param memDesc - Memory descriptor of particle data on CPU side
 */
float4 centerOfMassRef(MemDesc& memDesc);

#endif /* NBODY_H */
