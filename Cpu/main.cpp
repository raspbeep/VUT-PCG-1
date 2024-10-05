/**
 * @file      main.cu
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
#include <cstdio>
#include <chrono>
#include <string>

#include <vector_types.h>

#include "nbody.h"
#include "h5Helper.h"

/**
 * Main rotine
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
  if (argc != 10)
  {
    std::printf("Usage: nbody <N> <dt> <steps> <threads/block> <write intesity> <reduction threads> <reduction threads/block> <input> <output>\n");
    std::exit(1);
  }

  // Number of particles
  const unsigned N         = static_cast<unsigned>(std::stoul(argv[1]));
  // Length of time step
  const float    dt        = std::stof(argv[2]);
  // Number of steps
  const unsigned steps     = static_cast<unsigned>(std::stoul(argv[3]));
  // Write frequency
  const unsigned writeFreq = static_cast<unsigned>(std::stoul(argv[5]));

  // Log benchmark setup
  std::printf("       NBODY CPU simulation\n"
              "N:                       %u\n"
              "dt:                      %f\n"
              "steps:                   %u\n",
              N, dt, steps);

  const std::size_t recordsCount = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;

  Particles  particles{};
  Velocities tmpVelocities{};

  // Allocate memory
  particles.posX   = static_cast<float*>(operator new[](N * sizeof(float), std::align_val_t{dataAlignment}));
  particles.posY   = static_cast<float*>(operator new[](N * sizeof(float), std::align_val_t{dataAlignment}));
  particles.posZ   = static_cast<float*>(operator new[](N * sizeof(float), std::align_val_t{dataAlignment}));
  particles.velX   = static_cast<float*>(operator new[](N * sizeof(float), std::align_val_t{dataAlignment}));
  particles.velY   = static_cast<float*>(operator new[](N * sizeof(float), std::align_val_t{dataAlignment}));
  particles.velZ   = static_cast<float*>(operator new[](N * sizeof(float), std::align_val_t{dataAlignment}));
  particles.weight = static_cast<float*>(operator new[](N * sizeof(float), std::align_val_t{dataAlignment}));

  tmpVelocities.x = static_cast<float*>(operator new[](N * sizeof(float), std::align_val_t{dataAlignment}));
  tmpVelocities.y = static_cast<float*>(operator new[](N * sizeof(float), std::align_val_t{dataAlignment}));
  tmpVelocities.z = static_cast<float*>(operator new[](N * sizeof(float), std::align_val_t{dataAlignment}));

  /********************************************************************************************************************/
  /*                              TODO: Fill memory descriptor layout                                                 */
  /********************************************************************************************************************/
  /*
   * Caution! Create only after CPU side allocation
   * parameters:
   *                            Stride of two            Offset of the first
   *       Data pointer       consecutive elements        element in FLOATS,
   *                          in FLOATS, not bytes            not bytes
  */
  MemDesc md(particles.posX,           1,                         0,
             particles.posY,           1,                         0,
             particles.posZ,           1,                         0,
             particles.velX,           1,                         0,
             particles.velY,           1,                         0,
             particles.velZ,           1,                         0,
             particles.weight,         1,                         0,
             N,
             recordsCount);

  // Initialisation of helper class and loading of input data
  H5Helper h5Helper(argv[8], argv[9], md);

  try
  {
    h5Helper.init();
    h5Helper.readParticleData();
  }
  catch (const std::exception& e)
  {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return EXIT_FAILURE;
  }

  // Lambda for checking if we should write current step to the file
  auto shouldWrite = [writeFreq](unsigned s) -> bool
  {
    return writeFreq > 0u && (s % writeFreq == 0u);
  };

  // Lamda for getting record number
  auto getRecordNum = [writeFreq](unsigned s) -> unsigned
  {
    return s / writeFreq;
  };

  // Start measurement
  const auto start = std::chrono::steady_clock::now();

  for (unsigned s = 0u; s < steps; ++s)
  {
    if (shouldWrite(s))
    {
      const auto recordNum = getRecordNum(s);

      float4 com = centerOfMass(particles, N);

      h5Helper.writeParticleData(recordNum);
      h5Helper.writeCom(com, recordNum);
    }

    calculateGravitationVelocity(particles, tmpVelocities, N, dt);
    calculateCollisionVelocity(particles, tmpVelocities, N, dt);
    updateParticle(particles, tmpVelocities, N, dt);
  }

  float4 finalCom = centerOfMass(particles, N);

  // End measurement
  const auto end = std::chrono::steady_clock::now();

  // Approximate simulation wall time
  const float elapsedTime = std::chrono::duration<float>(end - start).count();
  std::printf("Time: %f s\n", elapsedTime);

  // Compute reference center of mass on CPU
  const float4 refCenterOfMass = centerOfMassRef(md);

  std::printf("Reference center of mass: %f, %f, %f, %f\n",
              refCenterOfMass.x,
              refCenterOfMass.y,
              refCenterOfMass.z,
              refCenterOfMass.w);

  std::printf("Center of mass on CPU: %f, %f, %f, %f\n",
              finalCom.x,
              finalCom.y,
              finalCom.z,
              finalCom.w);

  // Writing final values to the file
  h5Helper.writeComFinal(finalCom);
  h5Helper.writeParticleDataFinal();

  // Free memory
  operator delete[](particles.posX,   std::align_val_t{dataAlignment});
  operator delete[](particles.posY,   std::align_val_t{dataAlignment});
  operator delete[](particles.posZ,   std::align_val_t{dataAlignment});
  operator delete[](particles.velX,   std::align_val_t{dataAlignment});
  operator delete[](particles.velY,   std::align_val_t{dataAlignment});
  operator delete[](particles.velZ,   std::align_val_t{dataAlignment});
  operator delete[](particles.weight, std::align_val_t{dataAlignment});

  operator delete[](tmpVelocities.x, std::align_val_t{dataAlignment});
  operator delete[](tmpVelocities.y, std::align_val_t{dataAlignment});
  operator delete[](tmpVelocities.z, std::align_val_t{dataAlignment});
}// end of main
//----------------------------------------------------------------------------------------------------------------------
