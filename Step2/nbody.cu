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

#include "nbody.cuh"
#include <cfloat>

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
  /*  TODO: CUDA kernel to calculate new particles velocity and position, use shared memory to minimize memory access */
  /********************************************************************************************************************/
  extern __shared__ float sharedMem[];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
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


  for (int tile = 0; tile < gridDim.x; tile++) {
    int tileIdx = tile * blockDim.x + threadIdx.x;

    sharedMem[threadIdx.x * 7] = pIn.position_x[tileIdx];
    sharedMem[threadIdx.x * 7 + 1] = pIn.position_y[tileIdx];
    sharedMem[threadIdx.x * 7 + 2] = pIn.position_z[tileIdx];
    sharedMem[threadIdx.x * 7 + 3] = pIn.velocity_x[tileIdx];
    sharedMem[threadIdx.x * 7 + 4] = pIn.velocity_y[tileIdx];
    sharedMem[threadIdx.x * 7 + 5] = pIn.velocity_z[tileIdx];
    sharedMem[threadIdx.x * 7 + 6] = pIn.mass[tileIdx];

    __syncthreads();

    for (int i = 0; i < blockDim.x; i++) {
      const float dx = posX - sharedMem[i * 7];
      const float dy = posY - sharedMem[i * 7 + 1];
      const float dz = posZ - sharedMem[i * 7 + 2];

      const float r2 = dx * dx + dy * dy + dz * dz;
      const float r = sqrtf(r2);
      const float r3 = r2 * r;

      // Calculate gravitation velocity
      const float F = G * weight * sharedMem[i * 7 + 6] / (r3 + FLT_MIN);
      if (r > COLLISION_DISTANCE) {
        newVelX += dx * F;
        newVelY += dy * F;
        newVelZ += dz * F;
      } else
      // Calculate collision velocity
      if (r > 0.f && r < COLLISION_DISTANCE)
      {
        int invWeightSum = 1 / (weight + sharedMem[i * 7 + 6]);
        newVelX += 2.f * sharedMem[i * 7 + 6] * (sharedMem[i * 7 + 3] - velX) * invWeightSum;
        newVelY += 2.f * sharedMem[i * 7 + 6] * (sharedMem[i * 7 + 4] - velY) * invWeightSum;
        newVelZ += 2.f * sharedMem[i * 7 + 6] * (sharedMem[i * 7 + 5] - velZ) * invWeightSum;
      }
    }
    __syncthreads();

    
  }
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
