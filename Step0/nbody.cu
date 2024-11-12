/**
 * @file      nbody.cu
 *
 * @author    Name Surname \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            xlogin00@fit.vutbr.cz
 *
 * @brief     PCG Assignment 1
 *
 * @version   2024
 *
 * @date      04 October   2023, 09:00 (created) \n
 */

#include <device_launch_parameters.h>

#include <cfloat>
#include <iostream>

#include "h5Helper.h"
#include "nbody.cuh"

/* Constants */
constexpr float G = 6.67384e-11f;
constexpr float COLLISION_DISTANCE = 0.01f;

/**
 * CUDA kernel to calculate gravitation velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void calculateGravitationVelocity(Particles p, Velocities tmpVel, const unsigned N,
                                             float dt) {
  /********************************************************************************************************************/
  /*              TODO: CUDA kernel to calculate gravitation velocity, see reference CPU version */
  /********************************************************************************************************************/
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  float newVelX{};
  float newVelY{};
  float newVelZ{};

  const float posX = p.position_x[idx];
  const float posY = p.position_y[idx];
  const float posZ = p.position_z[idx];
  const float weight = p.mass[idx];

  for (unsigned i = 0; i < N; i++) {
    if (i == idx) continue;
    const float otherPosX = p.position_x[i];
    const float otherPosY = p.position_y[i];
    const float otherPosZ = p.position_z[i];
    const float otherWeight = p.mass[i];

    const float dx = otherPosX - posX;
    const float dy = otherPosY - posY;
    const float dz = otherPosZ - posZ;

    const float r2 = dx * dx + dy * dy + dz * dz;
    const float r = sqrtf(r2);
    const float r3 = r2 * r;

    const float F = G * weight * otherWeight / (r3 + FLT_MIN);
    if (r > COLLISION_DISTANCE) {
      newVelX += dx * F;
      newVelY += dy * F;
      newVelZ += dz * F;
    }
  }
  newVelX *= dt / weight;
  newVelY *= dt / weight;
  newVelZ *= dt / weight;

  tmpVel.velocity_x[idx] = newVelX;
  tmpVel.velocity_y[idx] = newVelY;
  tmpVel.velocity_z[idx] = newVelZ;
}  // end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate collision velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void calculateCollisionVelocity(Particles p, Velocities tmpVel, const unsigned N,
                                           float dt) {
  /********************************************************************************************************************/
  /*              TODO: CUDA kernel to calculate collision velocity, see reference CPU version */
  /********************************************************************************************************************/
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  const float posX = p.position_x[idx];
  const float posY = p.position_y[idx];
  const float posZ = p.position_z[idx];
  const float velX = p.velocity_x[idx];
  const float velY = p.velocity_y[idx];
  const float velZ = p.velocity_z[idx];
  const float weight = p.mass[idx];

  float newVelX = tmpVel.velocity_x[idx];
  float newVelY = tmpVel.velocity_y[idx];
  float newVelZ = tmpVel.velocity_z[idx];

  for (unsigned i = 0; i < N; i++) {
    if (i == idx) continue;
    const float otherPosX = p.position_x[i];
    const float otherPosY = p.position_y[i];
    const float otherPosZ = p.position_z[i];
    const float otherVelX = p.velocity_x[i];
    const float otherVelY = p.velocity_y[i];
    const float otherVelZ = p.velocity_z[i];
    const float otherWeight = p.mass[i];

    const float dx = otherPosX - posX;
    const float dy = otherPosY - posY;
    const float dz = otherPosZ - posZ;

    float r2 = dx * dx + dy * dy + dz * dz;
    float r = sqrtf(r2);

    if (r > 0.f && r < COLLISION_DISTANCE) {
      int invWeightSum = 1 / (weight + otherWeight);
      newVelX += 2.f * otherWeight * (otherVelX - velX) * invWeightSum;
      newVelY += 2.f * otherWeight * (otherVelY - velY) * invWeightSum;
      newVelZ += 2.f * otherWeight * (otherVelZ - velZ) * invWeightSum;
    }
  }
  tmpVel.velocity_x[idx] = newVelX;
  tmpVel.velocity_y[idx] = newVelY;
  tmpVel.velocity_z[idx] = newVelZ;
}  // end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void updateParticles(Particles p, Velocities tmpVel, const unsigned N, float dt) {
  /********************************************************************************************************************/
  /*             TODO: CUDA kernel to update particles velocities and positions, see reference CPU
   * version            */
  /********************************************************************************************************************/
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  float posX = p.position_x[idx];
  float posY = p.position_y[idx];
  float posZ = p.position_z[idx];

  float velX = p.velocity_x[idx];
  float velY = p.velocity_y[idx];
  float velZ = p.velocity_z[idx];

  const float newVelX = tmpVel.velocity_x[idx];
  const float newVelY = tmpVel.velocity_y[idx];
  const float newVelZ = tmpVel.velocity_z[idx];

  velX += newVelX;
  velY += newVelY;
  velZ += newVelZ;

  posX += velX * dt;
  posY += velY * dt;
  posZ += velZ * dt;

  p.position_x[idx] = posX;
  p.position_y[idx] = posY;
  p.position_z[idx] = posZ;

  p.velocity_x[idx] = velX;
  p.velocity_y[idx] = velY;
  p.velocity_z[idx] = velZ;

}  // end of update_particle
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate particles center of mass
 * @param p    - particles
 * @param com  - pointer to a center of mass
 * @param lock - pointer to a user-implemented lock
 * @param N    - Number of particles
 */
__global__ void centerOfMass(Particles p, float4* com, int* lock, const unsigned N) {

}  // end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
__host__ float4 centerOfMassRef(MemDesc& memDesc) {
  float4 com{};

  for (std::size_t i{}; i < memDesc.getDataSize(); i++) {
    const float3 pos = {memDesc.getPosX(i), memDesc.getPosY(i), memDesc.getPosZ(i)};
    const float w = memDesc.getWeight(i);

    // Calculate the vector on the line connecting current body and most recent position of
    // center-of-mass Calculate weight ratio only if at least one particle isn't massless
    const float4 d = {pos.x - com.x, pos.y - com.y, pos.z - com.z,
                      ((memDesc.getWeight(i) + com.w) > 0.0f)
                          ? (memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w))
                          : 0.0f};

    // Update position and weight of the center-of-mass according to the weight ration and vector
    com.x += d.x * d.w;
    com.y += d.y * d.w;
    com.z += d.z * d.w;
    com.w += w;
  }

  return com;
}  // enf of centerOfMassRef
//----------------------------------------------------------------------------------------------------------------------
