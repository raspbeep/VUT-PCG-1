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
  const float    dt                  = std::stof(argv[2]);
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
  float4*   hCenterOfMass{};

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

  CUDA_CALL(cudaHostAlloc(&hCenterOfMass, sizeof(float4), cudaHostAllocMapped));

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
  float4*   dCenterOfMass{};
  int*      dLock{};

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

  CUDA_CALL(cudaMalloc(&dCenterOfMass, sizeof(float4)));
  CUDA_CALL(cudaMalloc(&dLock, sizeof(int)));

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
  /*                                     TODO: Clear GPU center of mass                                               */
  /********************************************************************************************************************/
  CUDA_CALL(cudaMemset(dCenterOfMass, 0, sizeof(float) * 4));
  CUDA_CALL(cudaMemset(dLock, 0, sizeof(int)));


  /********************************************************************************************************************/
  /*                           TODO: Declare and create necessary CUDA streams and events                             */
  /********************************************************************************************************************/
  cudaEvent_t event_velocity_done, event_com_done, event_com_memset_done;

  CUDA_CALL(cudaEventCreate(&event_velocity_done));
  CUDA_CALL(cudaEventCreate(&event_com_done));
  CUDA_CALL(cudaEventCreate(&event_com_memset_done));

  cudaStream_t stream_velocity, stream_write, stream_com;
  CUDA_CALL(cudaStreamCreate(&stream_velocity));
  CUDA_CALL(cudaStreamCreate(&stream_write));
  CUDA_CALL(cudaStreamCreate(&stream_com));

  /********************************************************************************************************************/


  // Get CUDA device warp size
  int device;
  int warpSize;

  CUDA_CALL(cudaGetDevice(&device));
  CUDA_CALL(cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device));

  /********************************************************************************************************************/
  /*                                  TODO: Set dynamic shared memory computation                                     */
  /********************************************************************************************************************/
  const std::size_t sharedMemSize    = simBlockDim * sizeof(float) * 7;
  const std::size_t redSharedMemSize = (simBlockDim /* warpSize*/) * sizeof(float) * 4;   // you can use warpSize variable

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

  /********************************************************************************************************************/
  /*            TODO: Edit the loop to work asynchronously and overlap computation with data transfers.               */
  /*                  Use shouldWrite lambda to determine if data should be outputted to file.                        */
  /*                           if (shouldWrite(s, writeFreq)) { ... }                                                 */
  /*                        Use getRecordNum lambda to get the record number.                                         */
  /********************************************************************************************************************/
  unsigned int recordNum{};
  for (unsigned s = 0u; s < steps; ++s)
  {
    const unsigned srcIdx = s % 2;        // source particles index
    const unsigned dstIdx = (s + 1) % 2;  // destination particles index

    /******************************************************************************************************************/
    /*                TODO: GPU kernels invocation with correctly set dynamic memory size and stream                  */
    /******************************************************************************************************************/
    calculateVelocity<<<simGridDim, simBlockDim, sharedMemSize, stream_velocity>>>(dParticles[srcIdx], dParticles[dstIdx], N, dt);
    
    if (shouldWrite(s)) {
      // memset memory for center of mass
      CUDA_CALL(cudaMemsetAsync(dCenterOfMass, 0, sizeof(float) * 4, stream_write));
      CUDA_CALL(cudaEventRecord(event_com_memset_done, stream_write));

      // write stream copies the srcIdx particles to the CPU
      CUDA_CALL(cudaMemcpyAsync(hParticles.position_x, dParticles[srcIdx].position_x, allocSize, cudaMemcpyDeviceToHost, stream_write));
      CUDA_CALL(cudaMemcpyAsync(hParticles.position_y, dParticles[srcIdx].position_y, allocSize, cudaMemcpyDeviceToHost, stream_write));
      CUDA_CALL(cudaMemcpyAsync(hParticles.position_z, dParticles[srcIdx].position_z, allocSize, cudaMemcpyDeviceToHost, stream_write));
      CUDA_CALL(cudaMemcpyAsync(hParticles.velocity_x, dParticles[srcIdx].velocity_x, allocSize, cudaMemcpyDeviceToHost, stream_write));
      CUDA_CALL(cudaMemcpyAsync(hParticles.velocity_y, dParticles[srcIdx].velocity_y, allocSize, cudaMemcpyDeviceToHost, stream_write));
      CUDA_CALL(cudaMemcpyAsync(hParticles.velocity_z, dParticles[srcIdx].velocity_z, allocSize, cudaMemcpyDeviceToHost, stream_write));

      CUDA_CALL(cudaStreamSynchronize(stream_write));
      // meanwhile CPU gets the record number
      recordNum = getRecordNum(s);
      // CPU writes the particle data to the file
      h5Helper.writeParticleData(recordNum);
      
      // center of mass stream waits for memset to finish
      CUDA_CALL(cudaStreamWaitEvent(stream_com, event_com_memset_done));
      // center of mass can work
      centerOfMass<<<redGridDim, redBlockDim, redSharedMemSize, stream_com>>>(dParticles[srcIdx], dCenterOfMass, dLock, N);
      CUDA_CALL(cudaEventRecord(event_com_done, stream_com));
      
      // write stream waits for center of mass to finish
      CUDA_CALL(cudaStreamWaitEvent(stream_write, event_com_done));
      CUDA_CALL(cudaMemcpyAsync(hCenterOfMass, dCenterOfMass, sizeof(float) * 4, cudaMemcpyDeviceToHost, stream_write));
      // CPU waits until the write stream finishes copying the data
      CUDA_CALL(cudaStreamSynchronize(stream_write));
      h5Helper.writeCom(*hCenterOfMass, recordNum);
      CUDA_CALL(cudaStreamSynchronize(stream_com));
    }
    CUDA_CALL(cudaStreamSynchronize(stream_velocity));
  }

  const unsigned resIdx = steps % 2;    // result particles index

  /********************************************************************************************************************/
  /*                          TODO: Invocation of center of mass kernel, do not forget to add                         */
  /*                              additional synchronization and set appropriate stream                               */
  /********************************************************************************************************************/
  CUDA_CALL(cudaMemset(dCenterOfMass, 0, sizeof(float) * 4));
  centerOfMass<<<redGridDim, redBlockDim, redSharedMemSize>>>(dParticles[resIdx], dCenterOfMass, dLock, N);

  // Wait for all CUDA kernels to finish
  CUDA_CALL(cudaDeviceSynchronize());

  // End measurement
  const auto end = std::chrono::steady_clock::now();

  // Approximate simulation wall time
  const float elapsedTime = std::chrono::duration<float>(end - start).count();
  std::printf("Time: %f s\n", elapsedTime);

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

  CUDA_CALL(cudaMemcpy(hCenterOfMass, dCenterOfMass, sizeof(float) * 4, cudaMemcpyDeviceToHost));


  // Compute reference center of mass on CPU
  const float4 refCenterOfMass = centerOfMassRef(md);

  std::printf("Reference center of mass: %f, %f, %f, %f\n",
              refCenterOfMass.x,
              refCenterOfMass.y,
              refCenterOfMass.z,
              refCenterOfMass.w);

  std::printf("Center of mass on GPU: %f, %f, %f, %f\n",
              hCenterOfMass->x,
              hCenterOfMass->y,
              hCenterOfMass->z,
              hCenterOfMass->w);

  // Writing final values to the file
  h5Helper.writeComFinal(*hCenterOfMass);
  h5Helper.writeParticleDataFinal();

  /********************************************************************************************************************/
  /*                                  TODO: CUDA streams and events destruction                                       */
  /********************************************************************************************************************/
  CUDA_CALL(cudaEventDestroy(event_velocity_done));
  CUDA_CALL(cudaEventDestroy(event_com_done));
  
  CUDA_CALL(cudaStreamDestroy(stream_velocity));
  CUDA_CALL(cudaStreamDestroy(stream_write));
  CUDA_CALL(cudaStreamDestroy(stream_com));

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

  CUDA_CALL(cudaFree(dCenterOfMass));
  CUDA_CALL(cudaFree(dLock));

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

  CUDA_CALL(cudaFreeHost(hCenterOfMass));

}// end of main
//----------------------------------------------------------------------------------------------------------------------
