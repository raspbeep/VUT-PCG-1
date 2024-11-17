/**
 * @file       nbody.cu
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

#include <device_launch_parameters.h>

#include <cfloat>

#include "nbody.cuh"

#define POS_X blockDim.x * 0
#define POS_Y blockDim.x * 1
#define POS_Z blockDim.x * 2
#define VEL_X blockDim.x * 3
#define VEL_Y blockDim.x * 4
#define VEL_Z blockDim.x * 5
#define MASS blockDim.x * 6

#define RED_POS_X blockDim.x * 0
#define RED_POS_Y blockDim.x * 1
#define RED_POS_Z blockDim.x * 2
#define RED_MASS blockDim.x * 3

/* Constants */
constexpr float G = 6.67384e-11f;
constexpr float COLLISION_DISTANCE = 0.01f;

/**
 * CUDA kernel to calculate new particles velocity and position
 * @param pIn  - particles in
 * @param pOut - particles out
 * @param N    - Number of particles
 * @param dt   - Size of the time step
 */
__global__ void calculateVelocity(Particles pIn, Particles pOut, const unsigned N, float dt) {
  extern __shared__ float sharedMem[];

  const unsigned globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

  // Early exit if thread is out of bounds
  // if (globalIdx >= N) return;

  float newVelX_grav = 0.0f, newVelY_grav = 0.0f, newVelZ_grav = 0.0f;
  float newVelX_coll = 0.0f, newVelY_coll = 0.0f, newVelZ_coll = 0.0f;

  const float posX = pIn.position_x[globalIdx];
  const float posY = pIn.position_y[globalIdx];
  const float posZ = pIn.position_z[globalIdx];
  const float velX = pIn.velocity_x[globalIdx];
  const float velY = pIn.velocity_y[globalIdx];
  const float velZ = pIn.velocity_z[globalIdx];
  const float weight = pIn.mass[globalIdx];

  for (unsigned int tile = 0; tile < (N + blockDim.x - 1) / blockDim.x; tile++) {
    unsigned int tileOffset = tile * blockDim.x;
    unsigned int threadOffset = tileOffset + threadIdx.x;

    if (threadOffset < N) {
      sharedMem[POS_X + threadIdx.x] = pIn.position_x[threadOffset];
      sharedMem[POS_Y + threadIdx.x] = pIn.position_y[threadOffset];
      sharedMem[POS_Z + threadIdx.x] = pIn.position_z[threadOffset];
      sharedMem[VEL_X + threadIdx.x] = pIn.velocity_x[threadOffset];
      sharedMem[VEL_Y + threadIdx.x] = pIn.velocity_y[threadOffset];
      sharedMem[VEL_Z + threadIdx.x] = pIn.velocity_z[threadOffset];
      sharedMem[MASS + threadIdx.x] = pIn.mass[threadOffset];
    }
    __syncthreads();

    for (int j = 0; j < blockDim.x; j++) {
      if (tileOffset + j >= N) break;

      const float otherPosX = sharedMem[POS_X + j];
      const float otherPosY = sharedMem[POS_Y + j];
      const float otherPosZ = sharedMem[POS_Z + j];
      const float otherVelX = sharedMem[VEL_X + j];
      const float otherVelY = sharedMem[VEL_Y + j];
      const float otherVelZ = sharedMem[VEL_Z + j];
      const float otherWeight = sharedMem[MASS + j];

      const float dx = otherPosX - posX;
      const float dy = otherPosY - posY;
      const float dz = otherPosZ - posZ;

      const float r2 = dx * dx + dy * dy + dz * dz;
      const float r = sqrtf(r2);

      if (r > COLLISION_DISTANCE) {
        const float r3 = r2 * r;
        const float F = G * otherWeight / (r3 + FLT_MIN);
        newVelX_grav += dx * F;
        newVelY_grav += dy * F;
        newVelZ_grav += dz * F;
      } else if (r > 0.f && r < COLLISION_DISTANCE) {
        const float invWeightSum = 1.0f / (weight + otherWeight);
        newVelX_coll += 2.f * otherWeight * (otherVelX - velX) * invWeightSum;
        newVelY_coll += 2.f * otherWeight * (otherVelY - velY) * invWeightSum;
        newVelZ_coll += 2.f * otherWeight * (otherVelZ - velZ) * invWeightSum;
      }
    }
    __syncthreads();
  }

  const float updatedVelX = velX + (newVelX_grav * dt) + newVelX_coll;
  const float updatedVelY = velY + (newVelY_grav * dt) + newVelY_coll;
  const float updatedVelZ = velZ + (newVelZ_grav * dt) + newVelZ_coll;

  pOut.position_x[globalIdx] = posX + updatedVelX * dt;
  pOut.position_y[globalIdx] = posY + updatedVelY * dt;
  pOut.position_z[globalIdx] = posZ + updatedVelZ * dt;

  pOut.velocity_x[globalIdx] = updatedVelX;
  pOut.velocity_y[globalIdx] = updatedVelY;
  pOut.velocity_z[globalIdx] = updatedVelZ;
}  // end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate particles center of mass
 * @param p    - particles
 * @param com  - pointer to a center of mass
 * @param lock - pointer to a user-implemented lock
 * @param N    - Number of particles
 */
__global__ void centerOfMass(Particles p, float4* com, int* lock, const unsigned N) {
  extern __shared__ float sharedData[];
  int idx = threadIdx.x;

  float sumX = 0.0f;
  float sumY = 0.0f;
  float sumZ = 0.0f;
  float sumW = 0.0f;

  for (int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x; i < N;
       i += blockDim.x * 2 * gridDim.x) {
    // first particle
    float mass_ratio = p.mass[i] / (sumW + p.mass[i]);

    float dx = p.position_x[i] - sumX;
    float dy = p.position_y[i] - sumY;
    float dz = p.position_z[i] - sumZ;

    sumX += dx * mass_ratio;
    sumY += dy * mass_ratio;
    sumZ += dz * mass_ratio;
    sumW += p.mass[i];

    // second particle
    if (i + blockDim.x < N) {
      mass_ratio = p.mass[i + blockDim.x] / (sumW + p.mass[i + blockDim.x]);

      float dx = p.position_x[i + blockDim.x] - sumX;
      float dy = p.position_y[i + blockDim.x] - sumY;
      float dz = p.position_z[i + blockDim.x] - sumZ;

      sumX += dx * mass_ratio;
      sumY += dy * mass_ratio;
      sumZ += dz * mass_ratio;
      sumW += p.mass[i + blockDim.x];
    }
  }

  // Store in shared memory
  sharedData[RED_POS_X + idx] = sumX;
  sharedData[RED_POS_Y + idx] = sumY;
  sharedData[RED_POS_Z + idx] = sumZ;
  sharedData[RED_MASS + idx] = sumW;
  __syncthreads();

  // Parallel reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (idx < stride) {
      const float mass1 = sharedData[RED_MASS + idx];
      const float mass2 = sharedData[RED_MASS + idx + stride];
      const float mass_ratio_red = mass2 / (mass1 + mass2);

      float dx = sharedData[RED_POS_X + idx + stride] - sharedData[RED_POS_X + idx];
      float dy = sharedData[RED_POS_Y + idx + stride] - sharedData[RED_POS_Y + idx];
      float dz = sharedData[RED_POS_Z + idx + stride] - sharedData[RED_POS_Z + idx];

      sharedData[RED_POS_X + idx] += dx * mass_ratio_red;
      sharedData[RED_POS_Y + idx] += dy * mass_ratio_red;
      sharedData[RED_POS_Z + idx] += dz * mass_ratio_red;
      sharedData[RED_MASS + idx] += mass2;
    }
    __syncthreads();
  }

  // Write results to global memory
  if (idx == 0) {
    while (atomicCAS(lock, 0, 1) != 0);
    float mass_ratio = sharedData[RED_MASS] / (com->w + sharedData[RED_MASS]);

    float dx = sharedData[RED_POS_X] - com->x;
    float dy = sharedData[RED_POS_Y] - com->y;
    float dz = sharedData[RED_POS_Z] - com->z;

    com->x += dx * mass_ratio;
    com->y += dy * mass_ratio;
    com->z += dz * mass_ratio;
    com->w += sharedData[RED_MASS];
    atomicExch(lock, 0);
  }
}
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
    // center-of-mass Calculate weight ratio
    const float4 d = {pos.x - com.x, pos.y - com.y, pos.z - com.z,
                      memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w)};

    // Update position and weight of the center-of-mass according to the weight ratio and vector
    com.x += d.x * d.w;
    com.y += d.y * d.w;
    com.z += d.z * d.w;
    com.w += w;
  }

  return com;
}  // enf of centerOfMassRef
//----------------------------------------------------------------------------------------------------------------------
