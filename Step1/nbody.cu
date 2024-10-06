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
#include "nbody.cuh"

/* Constants */
constexpr float G                  = 6.67384e-11f;
constexpr float COLLISION_DISTANCE = 0.01f;

/**
 * CUDA kernel to calculate new particles velocity and position
 * @param pIn  - particles in
 * @param pOut - particles out
 * @param N    - Number of particles
 * @param dt   - Size of the time step
 */
__global__ void calculateVelocity(Particles pIn, Particles pOut, const unsigned N, float dt)
{
  /********************************************************************************************************************/
  /*          TODO: CUDA kernel to calculate new particles velocity and position, collapse previous kernels           */
  /********************************************************************************************************************/
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  float newVelX{};
  float newVelY{};
  float newVelZ{};

  const float posX   = pIn.position_x[idx];
  const float posY   = pIn.position_y[idx];
  const float posZ   = pIn.position_z[idx];
  
  const float velX   = pIn.velocity_x[idx];
  const float velY   = pIn.velocity_y[idx];
  const float velZ   = pIn.velocity_z[idx];

  const float weight = pIn.mass[idx];
  
  for (unsigned i = 0; i < N; i++)
  {
    if (i == idx)
      continue;

    const float otherPosX = pIn.position_x[i];
    const float otherPosY = pIn.position_y[i];
    const float otherPosZ = pIn.position_z[i];
    
    const float otherVelX = pIn.velocity_x[i];
    const float otherVelY = pIn.velocity_y[i];
    const float otherVelZ = pIn.velocity_z[i];
    
    const float otherWeight = pIn.mass[i];

    const float dx = otherPosX - posX;
    const float dy = otherPosY - posY;
    const float dz = otherPosZ - posZ;

    const float r2 = dx * dx + dy * dy + dz * dz;
    const float r = sqrtf(r2);

    // Calculate gravitation velocity
    const float F = G * weight * otherWeight / (r2 + FLT_MIN);
    newVelX += (r > COLLISION_DISTANCE) ? dx / r * F : 0.f;
    newVelY += (r > COLLISION_DISTANCE) ? dy / r * F : 0.f;
    newVelZ += (r > COLLISION_DISTANCE) ? dz / r * F : 0.f;

    // Calculate collision velocity
    if (r > 0.f && r < COLLISION_DISTANCE)
    {
      newVelX += (((weight * velX - otherWeight * velX + 2.f * otherWeight * otherVelX) / (weight + otherWeight)) - velX);
      newVelY += (((weight * velY - otherWeight * velY + 2.f * otherWeight * otherVelY) / (weight + otherWeight)) - velY);
      newVelZ += (((weight * velZ - otherWeight * velZ + 2.f * otherWeight * otherVelZ) / (weight + otherWeight)) - velZ);
    }
  }

  newVelX *= dt / weight;
  newVelY *= dt / weight;
  newVelZ *= dt / weight;

  pOut.velocity_x[idx] = newVelX;
  pOut.velocity_y[idx] = newVelY;
  pOut.velocity_z[idx] = newVelZ;

  // Update particle positions and velocities
  float posXUpdated = posX + (velX + newVelX) * dt;
  float posYUpdated = posY + (velY + newVelY) * dt;
  float posZUpdated = posZ + (velZ + newVelZ) * dt;

  pOut.position_x[idx] = posXUpdated;
  pOut.position_y[idx] = posYUpdated;
  pOut.position_z[idx] = posZUpdated;

  pOut.velocity_x[idx] += newVelX;
  pOut.velocity_y[idx] += newVelY;
  pOut.velocity_z[idx] += newVelZ;
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate particles center of mass
 * @param p    - particles
 * @param com  - pointer to a center of mass
 * @param lock - pointer to a user-implemented lock
 * @param N    - Number of particles
 */
__global__ void centerOfMass(Particles p, float4* com, int* lock, const unsigned N)
{

}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
__host__ float4 centerOfMassRef(MemDesc& memDesc)
{
  float4 com{};

  for (std::size_t i{}; i < memDesc.getDataSize(); i++)
  {
    const float3 pos = {memDesc.getPosX(i), memDesc.getPosY(i), memDesc.getPosZ(i)};
    const float  w   = memDesc.getWeight(i);

    // Calculate the vector on the line connecting current body and most recent position of center-of-mass
    // Calculate weight ratio only if at least one particle isn't massless
    const float4 d = {pos.x - com.x,
                      pos.y - com.y,
                      pos.z - com.z,
                      ((memDesc.getWeight(i) + com.w) > 0.0f)
                        ? ( memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w))
                        : 0.0f};

    // Update position and weight of the center-of-mass according to the weight ration and vector
    com.x += d.x * d.w;
    com.y += d.y * d.w;
    com.z += d.z * d.w;
    com.w += w;
  }

  return com;
}// enf of centerOfMassRef
//----------------------------------------------------------------------------------------------------------------------
