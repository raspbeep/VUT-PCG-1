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

#define POS_X blockDim.x * 0
#define POS_Y blockDim.x * 1
#define POS_Z blockDim.x * 2
#define VEL_X blockDim.x * 3
#define VEL_Y blockDim.x * 4
#define VEL_Z blockDim.x * 5
#define MASS  blockDim.x * 6

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

  float newVelX{};
  float newVelY{};
  float newVelZ{};

  unsigned int nThreads = gridDim.x * blockDim.x;

  // iterate over all chunks of particles
  for (unsigned int gridIdx = 0; gridIdx < ceil(float(N) / nThreads); gridIdx++) {
    newVelX = 0.0f;
    newVelY = 0.0f;
    newVelZ = 0.0f;

    unsigned int globalIdx = (gridIdx * nThreads) + (blockIdx.x * blockDim.x + threadIdx.x);

    const float posX   = globalIdx < N ? pIn.position_x[idx] : 0.f;
    const float posY   = globalIdx < N ? pIn.position_y[idx] : 0.f;
    const float posZ   = globalIdx < N ? pIn.position_z[idx] : 0.f;
    const float velX   = globalIdx < N ? pIn.velocity_x[idx] : 0.f;
    const float velY   = globalIdx < N ? pIn.velocity_y[idx] : 0.f;
    const float velZ   = globalIdx < N ? pIn.velocity_z[idx] : 0.f;
    const float weight = globalIdx < N ? pIn.mass[idx] : 0.f;

    // iterate over all tiles
    for (unsigned int tile = 0; tile < ceil(float(N) / blockDim.x); tile++) {
      unsigned int tileOffset = tile * blockDim.x;
      unsigned int threadOffset = tileOffset + threadIdx.x;

      // all threads load data into shared memory
      sharedMem[POS_X + threadIdx.x] = threadOffset < N ? pIn.position_x[threadOffset] : 0.f;
      sharedMem[POS_Y + threadIdx.x] = threadOffset < N ? pIn.position_y[threadOffset] : 0.f;
      sharedMem[POS_Z + threadIdx.x] = threadOffset < N ? pIn.position_z[threadOffset] : 0.f;
      sharedMem[VEL_X + threadIdx.x] = threadOffset < N ? pIn.velocity_x[threadOffset] : 0.f;
      sharedMem[VEL_Y + threadIdx.x] = threadOffset < N ? pIn.velocity_y[threadOffset] : 0.f;
      sharedMem[VEL_Z + threadIdx.x] = threadOffset < N ? pIn.velocity_z[threadOffset] : 0.f;
      sharedMem[MASS + threadIdx.x] = threadOffset < N ? pIn.mass[threadOffset] : 0.f;

      __syncthreads();
      
      // iterate over all particles in the tile
      for (int j = 0; j < blockDim.x; j++) {
        const float otherPosX   = sharedMem[POS_X + j];
        const float otherPosY   = sharedMem[POS_Y + j];
        const float otherPosZ   = sharedMem[POS_Z + j];
        const float otherVelX   = sharedMem[VEL_X + j];
        const float otherVelY   = sharedMem[VEL_Y + j];
        const float otherVelZ   = sharedMem[VEL_Z + j];
        const float otherWeight = sharedMem[MASS  + j];

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
          int invWeightSum = 1.f / (weight + otherWeight);
          newVelX += 2.f * otherWeight * (otherPosX - velX) * invWeightSum;
          newVelY += 2.f * otherWeight * (otherPosY - velY) * invWeightSum;
          newVelZ += 2.f * otherWeight * (otherPosZ - velZ) * invWeightSum;
        }
      }
      
      __syncthreads(); 
    }
    if (globalIdx < N) {
      pOut.velocity_x[idx] = newVelX;
      pOut.velocity_y[idx] = newVelY;
      pOut.velocity_z[idx] = newVelZ;

      pOut.position_x[idx] = posX + (velX + newVelX) * dt;
      pOut.position_y[idx] = posY + (velY + newVelY) * dt;
      pOut.position_z[idx] = posZ + (velZ + newVelZ) * dt;
    }
  }
}
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
