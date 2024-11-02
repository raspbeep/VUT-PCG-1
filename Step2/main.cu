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

#include "nbody.cuh"
#include "h5Helper.h"

/**
 * @brief CUDA error checking macro
 * @param call CUDA API call
 */
#define CUDA_CALL(call) \
  do { \
    const cudaError_t _error = (call); \
    if (_error != cudaSuccess) \
    { \
      std::fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(_error)); \
      std::exit(EXIT_FAILURE); \
    } \
  } while(0)

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
  const unsigned N                   = static_cast<unsigned>(std::stoul(argv[1]));
  // Length of time step
  const float    dt                   = std::stof(argv[2]);
  // Number of steps
  const unsigned steps               = static_cast<unsigned>(std::stoul(argv[3]));
  // Number of thread blocks
  const unsigned simBlockDim         = static_cast<unsigned>(std::stoul(argv[4]));
  // Write frequency
  const unsigned writeFreq           = static_cast<unsigned>(std::stoul(argv[5]));
  // number of reduction threads
  const unsigned redTotalThreadCount = static_cast<unsigned>(std::stoul(argv[6]));
  // Number of reduction threads/blocks
  const unsigned redBlockDim         = static_cast<unsigned>(std::stoul(argv[7]));

  // Size of the simulation CUDA grid - number of blocks
  const unsigned simGridDim = (N + simBlockDim - 1) / simBlockDim;
  // Size of the reduction CUDA grid - number of blocks
  const unsigned redGridDim = (redTotalThreadCount + redBlockDim - 1) / redBlockDim;

  // Log benchmark setup
  std::printf("       NBODY GPU simulation\n"
              "N:                       %u\n"
              "dt:                      %f\n"
              "steps:                   %u\n"
              "threads/block:           %u\n"
              "blocks/grid:             %u\n"
              "reduction threads/block: %u\n"
              "reduction blocks/grid:   %u\n",
              N, dt, steps, simBlockDim, simGridDim, redBlockDim, redGridDim);

  const std::size_t recordsCount = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;
  
  const size_t allocSize = N * sizeof(float);
  
  Particles hParticles{};

  /********************************************************************************************************************/
  /*                              TODO: CPU side memory allocation (pinned)                                           */
  /********************************************************************************************************************/
  CUDA_CALL(cudaHostAlloc(&hParticles.position_x, allocSize, cudaHostAllocDefault));
  CUDA_CALL(cudaHostAlloc(&hParticles.position_y, allocSize, cudaHostAllocDefault));
  CUDA_CALL(cudaHostAlloc(&hParticles.position_z, allocSize, cudaHostAllocDefault));

  CUDA_CALL(cudaHostAlloc(&hParticles.velocity_x, allocSize, cudaHostAllocDefault));
  CUDA_CALL(cudaHostAlloc(&hParticles.velocity_y, allocSize, cudaHostAllocDefault));
  CUDA_CALL(cudaHostAlloc(&hParticles.velocity_z, allocSize, cudaHostAllocDefault));

  CUDA_CALL(cudaHostAlloc(&hParticles.mass, allocSize, cudaHostAllocDefault));

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
  MemDesc md(hParticles.position_x,     1,                        0,
             hParticles.position_y,     1,                        0,
             hParticles.position_z,     1,                        0,
             hParticles.velocity_x,     1,                        0,
             hParticles.velocity_y,     1,                        0,
             hParticles.velocity_z,     1,                        0,
             hParticles.mass,           1,                        0,
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

  Particles dParticles[2]{};

  /********************************************************************************************************************/
  /*                                     TODO: GPU side memory allocation                                             */
  /********************************************************************************************************************/
  CUDA_CALL(cudaMalloc(&dParticles[0].position_x, allocSize));
  CUDA_CALL(cudaMalloc(&dParticles[0].position_y, allocSize));
  CUDA_CALL(cudaMalloc(&dParticles[0].position_z, allocSize));
  
  CUDA_CALL(cudaMalloc(&dParticles[0].velocity_x, allocSize));
  CUDA_CALL(cudaMalloc(&dParticles[0].velocity_y, allocSize));
  CUDA_CALL(cudaMalloc(&dParticles[0].velocity_z, allocSize));
  
  CUDA_CALL(cudaMalloc(&dParticles[0].mass, allocSize));

  CUDA_CALL(cudaMalloc(&dParticles[1].position_x, allocSize));
  CUDA_CALL(cudaMalloc(&dParticles[1].position_y, allocSize));
  CUDA_CALL(cudaMalloc(&dParticles[1].position_z, allocSize));

  CUDA_CALL(cudaMalloc(&dParticles[1].velocity_x, allocSize));
  CUDA_CALL(cudaMalloc(&dParticles[1].velocity_y, allocSize));
  CUDA_CALL(cudaMalloc(&dParticles[1].velocity_z, allocSize));

  CUDA_CALL(cudaMalloc(&dParticles[1].mass, allocSize));

  /********************************************************************************************************************/
  /*                                     TODO: Memory transfer CPU -> GPU                                             */
  /********************************************************************************************************************/
  CUDA_CALL(cudaMemcpy(dParticles[0].position_x, hParticles.position_x, allocSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].position_y, hParticles.position_y, allocSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].position_z, hParticles.position_z, allocSize, cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemcpy(dParticles[0].velocity_x, hParticles.velocity_x, allocSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].velocity_y, hParticles.velocity_y, allocSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].velocity_z, hParticles.velocity_z, allocSize, cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemcpy(dParticles[0].mass, hParticles.mass, allocSize, cudaMemcpyHostToDevice));

  // can be copied in device
  CUDA_CALL(cudaMemcpy(dParticles[1].position_x, dParticles[0].position_x, allocSize, cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[1].position_y, dParticles[0].position_y, allocSize, cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[1].position_z, dParticles[0].position_z, allocSize, cudaMemcpyDeviceToDevice));

  CUDA_CALL(cudaMemcpy(dParticles[1].velocity_x, dParticles[0].velocity_x, allocSize, cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[1].velocity_y, dParticles[0].velocity_y, allocSize, cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[1].velocity_z, dParticles[0].velocity_z, allocSize, cudaMemcpyDeviceToDevice));

  CUDA_CALL(cudaMemcpy(dParticles[1].mass, dParticles[0].mass, allocSize, cudaMemcpyDeviceToDevice));

  /********************************************************************************************************************/
  /*                                  TODO: Set dynamic shared memory computation                                     */
  /********************************************************************************************************************/
  const std::size_t sharedMemSize = simBlockDim * sizeof(float) * 7;

  // Start measurement
  const auto start = std::chrono::steady_clock::now();

  for (unsigned s = 0u; s < steps; ++s)
  {
    const unsigned srcIdx = s % 2;        // source particles index
    const unsigned dstIdx = (s + 1) % 2;  // destination particles index

    /******************************************************************************************************************/
    /*                   TODO: GPU kernel invocation with correctly set dynamic memory size                           */
    /******************************************************************************************************************/
    calculateVelocity<<<simGridDim, simBlockDim, sharedMemSize>>>(dParticles[srcIdx], dParticles[dstIdx], N, dt);
  }

  // Wait for all CUDA kernels to finish
  CUDA_CALL(cudaDeviceSynchronize());

  // End measurement
  const auto end = std::chrono::steady_clock::now();

  // Approximate simulation wall time
  const float elapsedTime = std::chrono::duration<float>(end - start).count();
  std::printf("Time: %f s\n", elapsedTime);

  const unsigned resIdx = steps % 2;    // result particles index

  /********************************************************************************************************************/
  /*                                     TODO: Memory transfer GPU -> CPU                                             */
  /********************************************************************************************************************/
  CUDA_CALL(cudaMemcpy(hParticles.position_x, dParticles[resIdx].position_x, allocSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.position_y, dParticles[resIdx].position_y, allocSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.position_z, dParticles[resIdx].position_z, allocSize, cudaMemcpyDeviceToHost));
  
  CUDA_CALL(cudaMemcpy(hParticles.velocity_x, dParticles[resIdx].velocity_x, allocSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velocity_y, dParticles[resIdx].velocity_y, allocSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velocity_z, dParticles[resIdx].velocity_z, allocSize, cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaMemcpy(hParticles.mass, dParticles[resIdx].mass, allocSize, cudaMemcpyDeviceToHost));

  // Compute reference center of mass on CPU
  const float4 refCenterOfMass = centerOfMassRef(md);

  std::printf("Reference center of mass: %f, %f, %f, %f\n",
              refCenterOfMass.x,
              refCenterOfMass.y,
              refCenterOfMass.z,
              refCenterOfMass.w);

  std::printf("Center of mass on GPU: %f, %f, %f, %f\n", 0.f, 0.f, 0.f, 0.f);

  // Writing final values to the file
  h5Helper.writeComFinal(refCenterOfMass);
  h5Helper.writeParticleDataFinal();

  /********************************************************************************************************************/
  /*                                     TODO: GPU side memory deallocation                                           */
  /********************************************************************************************************************/
  CUDA_CALL(cudaFree(dParticles[0].position_x));
  CUDA_CALL(cudaFree(dParticles[0].position_y));
  CUDA_CALL(cudaFree(dParticles[0].position_z));

  CUDA_CALL(cudaFree(dParticles[0].velocity_x));
  CUDA_CALL(cudaFree(dParticles[0].velocity_y));
  CUDA_CALL(cudaFree(dParticles[0].velocity_z));

  CUDA_CALL(cudaFree(dParticles[0].mass));

  CUDA_CALL(cudaFree(dParticles[1].position_x));
  CUDA_CALL(cudaFree(dParticles[1].position_y));
  CUDA_CALL(cudaFree(dParticles[1].position_z));

  CUDA_CALL(cudaFree(dParticles[1].velocity_x));
  CUDA_CALL(cudaFree(dParticles[1].velocity_y));
  CUDA_CALL(cudaFree(dParticles[1].velocity_z));
  
  CUDA_CALL(cudaFree(dParticles[1].mass)); 
  
  /********************************************************************************************************************/
  /*                                     TODO: CPU side memory deallocation                                           */
  /********************************************************************************************************************/
  CUDA_CALL(cudaFreeHost(hParticles.position_x));
  CUDA_CALL(cudaFreeHost(hParticles.position_y));
  CUDA_CALL(cudaFreeHost(hParticles.position_z));

  CUDA_CALL(cudaFreeHost(hParticles.velocity_x));
  CUDA_CALL(cudaFreeHost(hParticles.velocity_y));
  CUDA_CALL(cudaFreeHost(hParticles.velocity_z));

  CUDA_CALL(cudaFreeHost(hParticles.mass));

}// end of main
//----------------------------------------------------------------------------------------------------------------------
