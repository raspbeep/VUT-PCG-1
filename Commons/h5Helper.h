/**
 * @file      h5Helper.h
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

#ifndef H5_HELPER_H
#define H5_HELPER_H

#include <cstddef>
#include <string>
#include <stdexcept>
#include <iostream>
#include <vector>

#include <hdf5_hl.h>
#include <hdf5.h>
#include <vector_types.h>

/// Forward definition for the H5Helper
class H5Helper;

 /**
  * @class MemDesc
  * Memory descriptor class
  */
class MemDesc
{
  public:
    /**
     * Constructor
     * parameters:
     *                      Stride of two               Offset of the first
     *      Data pointer    consecutive elements        element in floats,
     *                      in floats, not bytes        not bytes
     */
    MemDesc(float* pos_x,  const std::size_t pos_x_stride,  const std::size_t pos_x_offset,
            float* pos_y,  const std::size_t pos_y_stride,  const std::size_t pos_y_offset,
            float* pos_z,  const std::size_t pos_z_stride,  const std::size_t pos_z_offset,
            float* vel_x,  const std::size_t vel_x_stride,  const std::size_t vel_x_offset,
            float* vel_y,  const std::size_t vel_y_stride,  const std::size_t vel_y_offset,
            float* vel_z,  const std::size_t vel_z_stride,  const std::size_t vel_z_offset,
            float* weight, const std::size_t weight_stride, const std::size_t weight_offset,
            const std::size_t N,
            const std::size_t steps);

    /// Getter for the i-th particle's position in X
    float& getPosX(std::size_t i)
    {
      return mDataPtr[Atr::kPosX][i*mStride[Atr::kPosX] + mOffset[Atr::kPosX]];
    }

    /// Getter for the i-th particle's position in Y
    float& getPosY(std::size_t i)
    {
      return mDataPtr[Atr::kPosY][i * mStride[Atr::kPosY] + mOffset[Atr::kPosY]];
    }

    /// Getter for the i-th particle's position in Z
    float& getPosZ(std::size_t i)
    {
      return mDataPtr[Atr::kPosZ][i * mStride[Atr::kPosZ] + mOffset[Atr::kPosZ]];
    }

    /// Getter for the i-th particle's weight
    float& getWeight(std::size_t i)
    {
      return mDataPtr[Atr::kWeight][i * mStride[Atr::kWeight] + mOffset[Atr::kWeight]];
    }

    /// Getter for the data size
    std::size_t getDataSize(){ return mSize; }

    /// Default constructor is not allowed
    MemDesc() = delete;

  protected:
    /// Vector of data pointers
    std::vector<float*> mDataPtr;
    /// Stride of two consecutive elements in memory pointed to by data pointers (in floats, not bytes)
    std::vector<std::size_t> mStride;
    /// Offset of the first element in the memory pointed to by data pointers (in floats, not bytes)
    std::vector<std::size_t> mOffset;

    std::size_t mSize;
    std::size_t mRecordsNum;

  private:
    friend H5Helper;

    /// Enum for dataset names
    enum Atr : std::size_t
    {
      kPosX,
      kPosY,
      kPosZ,
      kVelX,
      kVelY,
      kVelZ,
      kWeight,
    };

    /// Number of datasets
    static constexpr std::size_t kAttrNum = 7;
};// end of MemDesc
//----------------------------------------------------------------------------------------------------------------------

/**
 * @class H5Helper
 *
 * HDF5 file format reader and writer
 *
 * @param inputFile
 * @param outputFile
 * @param md
 */
class H5Helper
{
  public:

    /**
     * Constructor requires file names and memory layout descriptor
     * @param inputFile
     * @param outputFile
     * @param md - memory descriptor
     */
    H5Helper(const std::string& inputFile,
             const std::string& outputFile,
             MemDesc            md);

    /// Default constructor is not allowed
    H5Helper() = delete;

    /// Destructor
    ~H5Helper();

    /// Initialize helper
    void init();

    /// Read input data to the memory according to the memory descriptor
    void readParticleData();
    /// Write simulation data to the output file according to the memory descriptor and record number
    void writeParticleData(const std::size_t record);
    /// Write final simulation data to the output file according to the memory descriptor
    void writeParticleDataFinal();
    /// Write center-of-mass to the output file according record number
    void writeCom(const float4& com, const std::size_t record);
    /// Write final center-of-mass to the output file */
    void writeComFinal(const float4& com);
  private:
    /// Memory descriptor
   	MemDesc mMd;

    /// File names
    std::string mInputFile;
    std::string mOutputFile;

    /// File handles
    hid_t mInputFileId;
    hid_t mOutputFileId;

    /// Input size
    std::size_t mInputSize;

    /// Vector of dataset names
    static const std::vector<std::string> mDatasetNames;
};// end of H5Helper
//----------------------------------------------------------------------------------------------------------------------

#endif /* H5_HELPER_H */
