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

#include <cmath>
#include <limits>

#include "nbody.h"

/* Constants */
constexpr float G                  = 6.67384e-11f;
constexpr float COLLISION_DISTANCE = 0.01f;

/**
 * Kernel to calculate gravitation velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
void calculateGravitationVelocity(Particles p, Velocities tmpVel, const unsigned N, float dt)
{
  float* const pPosX   = p.posX;
  float* const pPosY   = p.posY;
  float* const pPosZ   = p.posZ;
  float* const pVelX   = p.velX;
  float* const pVelY   = p.velY;
  float* const pVelZ   = p.velZ;
  float* const pWeight = p.weight;

  float* const tmpVelX = tmpVel.x;
  float* const tmpVelY = tmpVel.y;
  float* const tmpVelZ = tmpVel.z;

# pragma omp parallel for firstprivate(pPosX, pPosY, pPosZ, pVelX, pVelY, pVelZ, pWeight, tmpVelX, tmpVelY, tmpVelZ, N, dt)
  for (unsigned i = 0u; i < N; ++i)
  {
    float newVelX{};
    float newVelY{};
    float newVelZ{};

    const float posX   = pPosX[i];
    const float posY   = pPosY[i];
    const float posZ   = pPosZ[i];
    const float weight = pWeight[i];

#   pragma omp simd aligned(pPosX, pPosY, pPosZ, pVelX, pVelY, pVelZ, pWeight, tmpVelX, tmpVelY, tmpVelZ: dataAlignment)
    for (unsigned j = 0u; j < N; ++j)
    {
      const float otherPosX   = pPosX[j];
      const float otherPosY   = pPosY[j];
      const float otherPosZ   = pPosZ[j];
      const float otherWeight = pWeight[j];

      const float dx = otherPosX - posX;
      const float dy = otherPosY - posY;
      const float dz = otherPosZ - posZ;

      const float r2 = dx * dx + dy * dy + dz * dz;
      const float r = std::sqrt(r2) + std::numeric_limits<float>::min();

      const float f = G * weight * otherWeight / r2 + std::numeric_limits<float>::min();

      newVelX += (r > COLLISION_DISTANCE) ? dx / r * f : 0.f;
      newVelY += (r > COLLISION_DISTANCE) ? dy / r * f : 0.f;
      newVelZ += (r > COLLISION_DISTANCE) ? dz / r * f : 0.f;
    }

    newVelX *= dt / weight;
    newVelY *= dt / weight;
    newVelZ *= dt / weight;

    tmpVelX[i] = newVelX;
    tmpVelY[i] = newVelY;
    tmpVelZ[i] = newVelZ;
  }
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Kernel to calculate collision velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
void calculateCollisionVelocity(Particles p, Velocities tmpVel, const unsigned N, float dt)
{
  float* const pPosX   = p.posX;
  float* const pPosY   = p.posY;
  float* const pPosZ   = p.posZ;
  float* const pVelX   = p.velX;
  float* const pVelY   = p.velY;
  float* const pVelZ   = p.velZ;
  float* const pWeight = p.weight;

  float* const tmpVelX = tmpVel.x;
  float* const tmpVelY = tmpVel.y;
  float* const tmpVelZ = tmpVel.z;

# pragma omp parallel for firstprivate(pPosX, pPosY, pPosZ, pVelX, pVelY, pVelZ, pWeight, tmpVelX, tmpVelY, tmpVelZ, N, dt)
  for (unsigned i = 0u; i < N; ++i)
  {
    float newVelX{};
    float newVelY{};
    float newVelZ{};

    const float posX   = pPosX[i];
    const float posY   = pPosY[i];
    const float posZ   = pPosZ[i];
    const float velX   = pVelX[i];
    const float velY   = pVelY[i];
    const float velZ   = pVelZ[i];
    const float weight = pWeight[i];

#   pragma omp simd aligned(pPosX, pPosY, pPosZ, pVelX, pVelY, pVelZ, pWeight, tmpVelX, tmpVelY, tmpVelZ: dataAlignment)
    for (unsigned j = 0u; j < N; ++j)
    {
      const float otherPosX   = pPosX[j];
      const float otherPosY   = pPosY[j];
      const float otherPosZ   = pPosZ[j];
      const float otherVelX   = pVelX[j];
      const float otherVelY   = pVelY[j];
      const float otherVelZ   = pVelZ[j];
      const float otherWeight = pWeight[j];

      const float dx = otherPosX - posX;
      const float dy = otherPosY - posY;
      const float dz = otherPosZ - posZ;

      const float r2 = dx * dx + dy * dy + dz * dz;
      const float r = std::sqrt(r2);

      newVelX += (r > 0.f && r < COLLISION_DISTANCE)
                 ? (((weight * velX - otherWeight * velX + 2.f * otherWeight * otherVelX) / (weight + otherWeight)) - velX)
                 : 0.f;
      newVelY += (r > 0.f && r < COLLISION_DISTANCE)
                 ? (((weight * velY - otherWeight * velY + 2.f * otherWeight * otherVelY) / (weight + otherWeight)) - velY)
                 : 0.f;
      newVelZ += (r > 0.f && r < COLLISION_DISTANCE)
                 ? (((weight * velZ - otherWeight * velZ + 2.f * otherWeight * otherVelZ) / (weight + otherWeight)) - velZ)
                 : 0.f;
    }

    tmpVelX[i] += newVelX;
    tmpVelY[i] += newVelY;
    tmpVelZ[i] += newVelZ;
  }
}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Kernel to update particles
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
void updateParticle(Particles p, Velocities tmpVel, const unsigned N, float dt)
{
  float* const pPosX   = p.posX;
  float* const pPosY   = p.posY;
  float* const pPosZ   = p.posZ;
  float* const pVelX   = p.velX;
  float* const pVelY   = p.velY;
  float* const pVelZ   = p.velZ;
  float* const pWeight = p.weight;

  float* const tmpVelX = tmpVel.x;
  float* const tmpVelY = tmpVel.y;
  float* const tmpVelZ = tmpVel.z;

# pragma omp parallel for simd \
         firstprivate(pPosX, pPosY, pPosZ, pVelX, pVelY, pVelZ, pWeight, tmpVelX, tmpVelY, tmpVelZ, N, dt) \
         aligned(pPosX, pPosY, pPosZ, pVelX, pVelY, pVelZ, pWeight, tmpVelX, tmpVelY, tmpVelZ: dataAlignment)
  for (unsigned i = 0u; i < N; ++i)
  {
    float posX = pPosX[i];
    float posY = pPosY[i];
    float posZ = pPosZ[i];

    float velX = pVelX[i];
    float velY = pVelY[i];
    float velZ = pVelZ[i];

    const float newVelX = tmpVelX[i];
    const float newVelY = tmpVelY[i];
    const float newVelZ = tmpVelZ[i];

    velX += newVelX;
    velY += newVelY;
    velZ += newVelZ;

    posX += velX * dt;
    posY += velY * dt;
    posZ += velZ * dt;

    pPosX[i] = posX;
    pPosY[i] = posY;
    pPosZ[i] = posZ;

    pVelX[i] = velX;
    pVelY[i] = velY;
    pVelZ[i] = velZ;
  }
}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------

/**
 * Function to calculate center of mass of 2 particles
 * @param a - First particle (inout)
 * @param b - Second particle (in)
 */
#pragma omp declare simd
static inline void centerOfMassReduction(float4& a, const float4& b)
{
  float4 d = {b.x - a.x,
              b.y - a.y,
              b.z - a.z,
              (a.w + b.w) > 0.f ? (b.w / (a.w + b.w)) : 0.f};

  a.x += d.x * d.w;
  a.y += d.y * d.w;
  a.z += d.z * d.w;
  a.w += b.w;
}

// Declare reduction for center of mass
#pragma omp declare reduction(comRed : float4 : centerOfMassReduction(omp_out, omp_in))

/**
 * Kernel to calculate particles center of mass
 * @param p    - particles
 * @param N    - Number of particles
 */
float4 centerOfMass(Particles p, const unsigned N)
{
  float4 com{};

  float* const pPosX   = p.posX;
  float* const pPosY   = p.posY;
  float* const pPosZ   = p.posZ;
  float* const pWeight = p.weight;

# pragma omp parallel for simd \
         firstprivate(pPosX, pPosY, pPosZ, pWeight, N) \
         aligned(pPosX, pPosY, pPosZ, pWeight: dataAlignment) \
         reduction(comRed: com)
  for (unsigned i = 0u; i < N; ++i)
  {
    const float4 particle = {pPosX[i], pPosY[i], pPosZ[i], pWeight[i]};

    centerOfMassReduction(com, particle);
  }

  return com;
}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
float4 centerOfMassRef(MemDesc& memDesc)
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
