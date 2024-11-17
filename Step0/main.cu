/**
 * @file       main.cu
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

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>

#include "h5Helper.h"
#include "nbody.cuh"

/**
 * @brief CUDA error checking macro
 * @param call CUDA API call
 */
#define CUDA_CALL(call)                                                     \
  do {                                                                     \
    const cudaError_t _error = (call);                                     \
    if (_error != cudaSuccess) {                                           \
      std::fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(_error));                            \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

/**
 * Main rotine
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char** argv) {
  if (argc != 10) {
    std::printf(
        "Usage: nbody <N> <dt> <steps> <threads/block> <write intesity> <reduction threads> "
        "<reduction threads/block> <input> <output>\n");
    std::exit(1);
  }

  // Number of particles
  const unsigned N = static_cast<unsigned>(std::stoul(argv[1]));
  // Length of time step
  const float dt = std::stof(argv[2]);
  // Number of steps
  const unsigned steps = static_cast<unsigned>(std::stoul(argv[3]));
  // Number of thread blocks
  const unsigned simBlockDim = static_cast<unsigned>(std::stoul(argv[4]));
  // Write frequency
  const unsigned writeFreq = static_cast<unsigned>(std::stoul(argv[5]));
  // number of reduction threads
  const unsigned redTotalThreadCount = static_cast<unsigned>(std::stoul(argv[6]));
  // Number of reduction threads/blocks
  const unsigned redBlockDim = static_cast<unsigned>(std::stoul(argv[7]));

  // Size of the simulation CUDA grid - number of blocks
  const unsigned simGridDim = (N + simBlockDim - 1) / simBlockDim;
  // Size of the reduction CUDA grid - number of blocks
  const unsigned redGridDim = (redTotalThreadCount + redBlockDim - 1) / redBlockDim;

  // Log benchmark setup
  std::printf(
      "       NBODY GPU simulation\n"
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
  /*                              CPU side memory allocation (pinned) */
  /********************************************************************************************************************/
  CUDA_CALL(cudaHostAlloc(&hParticles.position_x, allocSize, cudaHostAllocDefault));
  CUDA_CALL(cudaHostAlloc(&hParticles.position_y, allocSize, cudaHostAllocDefault));
  CUDA_CALL(cudaHostAlloc(&hParticles.position_z, allocSize, cudaHostAllocDefault));

  CUDA_CALL(cudaHostAlloc(&hParticles.velocity_x, allocSize, cudaHostAllocDefault));
  CUDA_CALL(cudaHostAlloc(&hParticles.velocity_y, allocSize, cudaHostAllocDefault));
  CUDA_CALL(cudaHostAlloc(&hParticles.velocity_z, allocSize, cudaHostAllocDefault));

  CUDA_CALL(cudaHostAlloc(&hParticles.mass, allocSize, cudaHostAllocDefault));

  /********************************************************************************************************************/
  /*                              Fill memory descriptor layout */
  /********************************************************************************************************************/
  /*
   * Caution! Create only after CPU side allocation
   * parameters:
   *                            Stride of two            Offset of the first
   *       Data pointer       consecutive elements        element in FLOATS,
   *                          in FLOATS, not bytes            not bytes
   */
  MemDesc md(hParticles.position_x, 1, 0, hParticles.position_y, 1, 0, hParticles.position_z, 1, 0,
             hParticles.velocity_x, 1, 0, hParticles.velocity_y, 1, 0, hParticles.velocity_z, 1, 0,
             hParticles.mass, 1, 0, N, recordsCount);

  // Initialisation of helper class and loading of input data
  H5Helper h5Helper(argv[8], argv[9], md);

  try {
    h5Helper.init();
    h5Helper.readParticleData();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return EXIT_FAILURE;
  }

  Particles dParticles{};
  Velocities dTmpVelocities{};

  /********************************************************************************************************************/
  /*                                     GPU side memory allocation */
  /********************************************************************************************************************/
  CUDA_CALL(cudaMalloc(&dParticles.position_x, allocSize));
  CUDA_CALL(cudaMalloc(&dParticles.position_y, allocSize));
  CUDA_CALL(cudaMalloc(&dParticles.position_z, allocSize));

  CUDA_CALL(cudaMalloc(&dParticles.velocity_x, allocSize));
  CUDA_CALL(cudaMalloc(&dParticles.velocity_y, allocSize));
  CUDA_CALL(cudaMalloc(&dParticles.velocity_z, allocSize));

  CUDA_CALL(cudaMalloc(&dParticles.mass, allocSize));

  CUDA_CALL(cudaMalloc(&dTmpVelocities.velocity_x, allocSize));
  CUDA_CALL(cudaMalloc(&dTmpVelocities.velocity_y, allocSize));
  CUDA_CALL(cudaMalloc(&dTmpVelocities.velocity_z, allocSize));

  /********************************************************************************************************************/
  /*                                     Memory transfer CPU -> GPU */
  /********************************************************************************************************************/
  CUDA_CALL(
      cudaMemcpy(dParticles.position_x, hParticles.position_x, allocSize, cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(dParticles.position_y, hParticles.position_y, allocSize, cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(dParticles.position_z, hParticles.position_z, allocSize, cudaMemcpyHostToDevice));

  CUDA_CALL(
      cudaMemcpy(dParticles.velocity_x, hParticles.velocity_x, allocSize, cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(dParticles.velocity_y, hParticles.velocity_y, allocSize, cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(dParticles.velocity_z, hParticles.velocity_z, allocSize, cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemcpy(dParticles.mass, hParticles.mass, allocSize, cudaMemcpyHostToDevice));

  // Start measurement
  const auto start = std::chrono::steady_clock::now();

  for (unsigned s = 0u; s < steps; ++s) {
    /******************************************************************************************************************/
    /*                                     GPU kernels invocation */
    /******************************************************************************************************************/
    calculateGravitationVelocity<<<simGridDim, simBlockDim>>>(dParticles, dTmpVelocities, N, dt);
    calculateCollisionVelocity<<<simGridDim, simBlockDim>>>(dParticles, dTmpVelocities, N, dt);
    updateParticles<<<simGridDim, simBlockDim>>>(dParticles, dTmpVelocities, N, dt);
  }

  // Wait for all CUDA kernels to finish
  CUDA_CALL(cudaDeviceSynchronize());

  // End measurement
  const auto end = std::chrono::steady_clock::now();

  // Approximate simulation wall time
  const float elapsedTime = std::chrono::duration<float>(end - start).count();
  std::printf("Time: %f s\n", elapsedTime);

  /********************************************************************************************************************/
  /*                                     Memory transfer GPU -> CPU */
  /********************************************************************************************************************/
  CUDA_CALL(
      cudaMemcpy(hParticles.position_x, dParticles.position_x, allocSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(
      cudaMemcpy(hParticles.position_y, dParticles.position_y, allocSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(
      cudaMemcpy(hParticles.position_z, dParticles.position_z, allocSize, cudaMemcpyDeviceToHost));

  CUDA_CALL(
      cudaMemcpy(hParticles.velocity_x, dParticles.velocity_x, allocSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(
      cudaMemcpy(hParticles.velocity_y, dParticles.velocity_y, allocSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(
      cudaMemcpy(hParticles.velocity_z, dParticles.velocity_z, allocSize, cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaMemcpy(hParticles.mass, dParticles.mass, allocSize, cudaMemcpyDeviceToHost));

  // Compute reference center of mass on CPU
  const float4 refCenterOfMass = centerOfMassRef(md);

  std::printf("Reference center of mass: %f, %f, %f, %f\n", refCenterOfMass.x, refCenterOfMass.y,
              refCenterOfMass.z, refCenterOfMass.w);

  std::printf("Center of mass on GPU: %f, %f, %f, %f\n", 0.f, 0.f, 0.f, 0.f);

  // Writing final values to the file
  h5Helper.writeComFinal(refCenterOfMass);
  h5Helper.writeParticleDataFinal();

  /********************************************************************************************************************/
  /*                                     GPU side memory deallocation */
  /********************************************************************************************************************/
  CUDA_CALL(cudaFree(dParticles.position_x));
  CUDA_CALL(cudaFree(dParticles.position_y));
  CUDA_CALL(cudaFree(dParticles.position_z));

  CUDA_CALL(cudaFree(dParticles.velocity_x));
  CUDA_CALL(cudaFree(dParticles.velocity_y));
  CUDA_CALL(cudaFree(dParticles.velocity_z));

  CUDA_CALL(cudaFree(dParticles.mass));

  CUDA_CALL(cudaFree(dTmpVelocities.velocity_x));
  CUDA_CALL(cudaFree(dTmpVelocities.velocity_y));
  CUDA_CALL(cudaFree(dTmpVelocities.velocity_z));

  /********************************************************************************************************************/
  /*                                     CPU side memory deallocation */
  /********************************************************************************************************************/
  CUDA_CALL(cudaFreeHost(hParticles.position_x));
  CUDA_CALL(cudaFreeHost(hParticles.position_y));
  CUDA_CALL(cudaFreeHost(hParticles.position_z));

  CUDA_CALL(cudaFreeHost(hParticles.velocity_x));
  CUDA_CALL(cudaFreeHost(hParticles.velocity_y));
  CUDA_CALL(cudaFreeHost(hParticles.velocity_z));

  CUDA_CALL(cudaFreeHost(hParticles.mass));

}  // end of main
//----------------------------------------------------------------------------------------------------------------------
