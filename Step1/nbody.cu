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

  const float posX = pIn.position_x[idx];
  const float posY = pIn.position_y[idx];
  const float posZ = pIn.position_z[idx];
  
  const float velX = pIn.velocity_x[idx];
  const float velY = pIn.velocity_y[idx];
  const float velZ = pIn.velocity_z[idx];

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

    const float dx = posX - otherPosX;
    const float dy = posY - otherPosY;
    const float dz = posZ - otherPosZ;

    const float r2 = dx * dx + dy * dy + dz * dz;
    const float r = sqrtf(r2);

    // Calculate gravitation velocity
    if (r > COLLISION_DISTANCE)
    {
    const float r3 = r2 * r;
    const float F = G * dt * otherWeight / (r3 + FLT_MIN);
      newVelX += dx * F;
      newVelY += dy * F;
      newVelZ += dz * F;
    } else
    // Calculate collision velocity
    if (r > 0.f && r < COLLISION_DISTANCE)
    {
      int invWeightSum = 1 / (weight + otherWeight);
      newVelX += 2.f * otherWeight * (otherVelX - velX) * invWeightSum;
      newVelY += 2.f * otherWeight * (otherVelY - velY) * invWeightSum;
      newVelZ += 2.f * otherWeight * (otherVelZ - velZ) * invWeightSum;
    }
  }

  pOut.velocity_x[idx] = newVelX;
  pOut.velocity_y[idx] = newVelY;
  pOut.velocity_z[idx] = newVelZ;

  pOut.position_x[idx] = posX + (velX + newVelX) * dt;
  pOut.position_y[idx] = posY + (velY + newVelY) * dt;
  pOut.position_z[idx] = posZ + (velZ + newVelZ) * dt;

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
