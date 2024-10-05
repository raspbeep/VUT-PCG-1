/**
 * @file      h5Helper.cpp
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

#include <vector>

#include "h5Helper.h"

/// Dataset names
const std::vector<std::string> H5Helper::mDatasetNames =
{
  "pos_x",
  "pos_y",
  "pos_z",
  "vel_x",
  "vel_y",
  "vel_z",
  "weight",
};// end of mDatasetNames initialization
//----------------------------------------------------------------------------------------------------------------------

MemDesc::MemDesc(float* pos_x,  const std::size_t pos_x_stride,  const std::size_t pos_x_offset,
                 float* pos_y,  const std::size_t pos_y_stride,  const std::size_t pos_y_offset,
                 float* pos_z,  const std::size_t pos_z_stride,  const std::size_t pos_z_offset,
                 float* vel_x,  const std::size_t vel_x_stride,  const std::size_t vel_x_offset,
                 float* vel_y,  const std::size_t vel_y_stride,  const std::size_t vel_y_offset,
                 float* vel_z,  const std::size_t vel_z_stride,  const std::size_t vel_z_offset,
                 float* weight, const std::size_t weight_stride, const std::size_t weight_offset,
                 const std::size_t N,
                 const std::size_t steps)
: mDataPtr(kAttrNum), mStride(kAttrNum), mOffset(kAttrNum), mSize(N), mRecordsNum(steps)
{
  mDataPtr[Atr::kPosX] = pos_x;
  mStride[Atr::kPosX]  = pos_x_stride;
  mOffset[Atr::kPosX]  = pos_x_offset;

  mDataPtr[Atr::kPosY] = pos_y;
  mStride[Atr::kPosY]  = pos_y_stride;
  mOffset[Atr::kPosY]  = pos_y_offset;

  mDataPtr[Atr::kPosZ] = pos_z;
  mStride[Atr::kPosZ]  = pos_z_stride;
  mOffset[Atr::kPosZ]  = pos_z_offset;

  mDataPtr[Atr::kVelX] = vel_x;
  mStride[Atr::kVelX]  = vel_x_stride;
  mOffset[Atr::kVelX]  = vel_x_offset;

  mDataPtr[Atr::kVelY] = vel_y;
  mStride[Atr::kVelY]  = vel_y_stride;
  mOffset[Atr::kVelY]  = vel_y_offset;

  mDataPtr[Atr::kVelZ] = vel_z;
  mStride[Atr::kVelZ]  = vel_z_stride;
  mOffset[Atr::kVelZ]  = vel_z_offset;

  mDataPtr[Atr::kWeight] = weight;
  mStride[Atr::kWeight]  = weight_stride;
  mOffset[Atr::kWeight]  = weight_offset;
}

H5Helper::H5Helper(const std::string& inputFile,
                   const std::string& outputFile,
                   MemDesc md)
: mMd(md),
  mInputFile(inputFile),
  mOutputFile(outputFile),
  mInputFileId(H5I_INVALID_HID),
  mOutputFileId(H5I_INVALID_HID)
{}

/**
 * Destructor
 */
H5Helper::~H5Helper()
{
  if (mInputFileId != H5I_INVALID_HID)
  {
    H5Fclose(mInputFileId);
  }

  if(mOutputFileId != H5I_INVALID_HID)
  {
    H5Fclose(mOutputFileId);
  }

}// end of destructor
//----------------------------------------------------------------------------------------------------------------------

/**
 * Initialize helper
 */
void H5Helper::init()
{
  // Open HDF5 files
  mInputFileId = H5Fopen(mInputFile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  if (mInputFileId == H5I_INVALID_HID)
  {
    throw std::runtime_error("Could not open input file!");
  }

  // Open dataset
  hid_t dataset   = H5Dopen2(mInputFileId, mDatasetNames[0].c_str(), H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(dataset);

  const int ndims = H5Sget_simple_extent_ndims(dataspace);

  std::vector<hsize_t> dims(ndims);
  H5Sget_simple_extent_dims(dataspace, dims.data(), nullptr);

  mInputSize = 1;
  for(int i = 0; i < ndims; ++i)
  {
    mInputSize *= dims[i];
  }

  if(mInputSize < mMd.mSize)
  {
    throw std::runtime_error("Input file contains less elements than required!");
  }
  else
  {
    if(mInputSize > mMd.mSize)
    {
      std::cout<<"Input file contains more elements than required! Ommiting the rest."<<std::endl;
    }
  }
  H5Sclose(dataspace);
  H5Dclose(dataset);


  // Create output file
  mOutputFileId = H5Fcreate(mOutputFile.c_str(), H5F_ACC_TRUNC,  H5P_DEFAULT, H5P_DEFAULT);

  if (mOutputFileId < 0)
  {
    throw std::runtime_error("Could not create output file!");
  }

  if(mMd.mRecordsNum > 0)
  {
    for(int i = 0; i < MemDesc::kAttrNum; ++i)
    {
      hsize_t dims[2]  = {mMd.mSize, mMd.mRecordsNum};
      hid_t dataspace = H5Screate_simple(2, dims, nullptr);
      hid_t dataset   = H5Dcreate2(mOutputFileId, mDatasetNames[i].c_str(), H5T_NATIVE_FLOAT, dataspace,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      H5Sclose(dataspace);
      H5Dclose(dataset);
    }
  }

  for(std::size_t i = 0; i < MemDesc::kAttrNum; ++i)
  {
    std::string sufix = "_final";

    hsize_t dims  = mMd.mSize;
    hid_t dataspace = H5Screate_simple(1, &dims, nullptr);
    hid_t dataset   = H5Dcreate2(mOutputFileId, (mDatasetNames[i] + sufix).c_str(), H5T_NATIVE_FLOAT, dataspace,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5Sclose(dataspace);
    H5Dclose(dataset);
  }

  if(mMd.mRecordsNum > 0)
  {
    hsize_t dim  = mMd.mRecordsNum;
    dataspace = H5Screate_simple(1, &dim, nullptr);

    dataset   = H5Dcreate2(mOutputFileId, "com_x", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dclose(dataset);

    dataset   = H5Dcreate2(mOutputFileId, "com_y", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dclose(dataset);

    dataset   = H5Dcreate2(mOutputFileId, "com_z", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dclose(dataset);

    dataset   = H5Dcreate2(mOutputFileId, "com_w", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dclose(dataset);

    H5Sclose(dataspace);
  }

  hsize_t dim  = 1;
  dataspace = H5Screate_simple(1, &dim, nullptr);

  dataset   = H5Dcreate2(mOutputFileId, "com_x_final", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dclose(dataset);

  dataset   = H5Dcreate2(mOutputFileId, "com_y_final", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dclose(dataset);

  dataset   = H5Dcreate2(mOutputFileId, "com_z_final", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dclose(dataset);

  dataset   = H5Dcreate2(mOutputFileId, "com_w_final", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dclose(dataset);

  H5Sclose(dataspace);
}// end of init
//----------------------------------------------------------------------------------------------------------------------


/**
 * Rad particles
 */
void H5Helper::readParticleData()
{

  for(std::size_t i = 0; i < MemDesc::kAttrNum; i++)
  {
    hid_t dataset  = H5Dopen2(mInputFileId, mDatasetNames[i].c_str(), H5P_DEFAULT);

    hid_t dataspace = H5Dget_space(dataset);
    hsize_t dStart  = 0;
    hsize_t dStride = 1;
    hsize_t dCount  = mMd.mSize;
    hsize_t dBlock  = 1;

    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, &dStart, &dStride, &dCount, &dBlock);

    hsize_t dim = mMd.mStride[i] * mMd.mSize;
    hid_t memspace = H5Screate_simple(1, &dim, nullptr);

    hsize_t start  = mMd.mOffset[i];
    hsize_t stride = mMd.mStride[i];
    hsize_t count  = mMd.mSize;
    hsize_t block  = 1;

    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, &start, &stride, &count, &block);

    H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, mMd.mDataPtr[i]);

    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
  }
}// end of readParticleData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write particles
 * @param record
 */
void H5Helper::writeParticleData(const size_t record)
{

  for(std::size_t i = 0; i < MemDesc::kAttrNum; i++)
  {
    hid_t dataset  = H5Dopen2(mOutputFileId, mDatasetNames[i].c_str(), H5P_DEFAULT);

    hid_t dataspace = H5Dget_space(dataset);

    hsize_t dStart[2]  = {0, record};
    hsize_t dCount[2]  = {mMd.mSize, 1};

    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, dStart, nullptr, dCount, nullptr);

    hsize_t dim = mMd.mStride[i] * mMd.mSize;
    hid_t memspace = H5Screate_simple(1, &dim, nullptr);

    hsize_t start  = mMd.mOffset[i];
    hsize_t stride = mMd.mStride[i];
    hsize_t count  = mMd.mSize;
    hsize_t block  = 1;

    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, &start, &stride, &count, &block);

    H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, mMd.mDataPtr[i]);

    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
  }
}// end of writeParticleData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write final particle data
 */
void H5Helper::writeParticleDataFinal()
{

  for(std::size_t i = 0; i < MemDesc::kAttrNum; i++)
  {
    std::string sufix = "_final";
    hid_t dataset  = H5Dopen2(mOutputFileId, (mDatasetNames[i] + sufix).c_str(), H5P_DEFAULT);

    hid_t dataspace = H5Dget_space(dataset);

    hsize_t dim = mMd.mStride[i] * mMd.mSize;
    hid_t memspace = H5Screate_simple(1, &dim, nullptr);

    hsize_t start  = mMd.mOffset[i];
    hsize_t stride = mMd.mStride[i];
    hsize_t count  = mMd.mSize;
    hsize_t block  = 1;

    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, &start, &stride, &count, &block);

    H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, H5S_ALL, H5P_DEFAULT, mMd.mDataPtr[i]);

    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
  }
}// end of writeParticleDataFinal
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write centre of mass
 * @param comX
 * @param comY
 * @param comZ
 * @param comW
 * @param record
 */
void H5Helper::writeCom(const float4& com, const size_t record)
{

  hid_t dataset  = H5Dopen2(mOutputFileId, "com_x", H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(dataset);
  hsize_t coords  = record;
  hsize_t dim = 1;
  hid_t memspace = H5Screate_simple(1, &dim, nullptr);

  H5Sselect_elements(dataspace, H5S_SELECT_SET, 1, &coords);

  H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, &com.x);

  H5Sclose(dataspace);
  H5Dclose(dataset);

  dataset  = H5Dopen2(mOutputFileId, "com_y", H5P_DEFAULT);
  dataspace = H5Dget_space(dataset);

  H5Sselect_elements(dataspace, H5S_SELECT_SET, 1, &coords);

  H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, &com.y);

  H5Sclose(dataspace);
  H5Dclose(dataset);

  dataset  = H5Dopen2(mOutputFileId, "com_z", H5P_DEFAULT);
  dataspace = H5Dget_space(dataset);

  H5Sselect_elements(dataspace, H5S_SELECT_SET, 1, &coords);

  H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, &com.z);

  H5Sclose(dataspace);
  H5Dclose(dataset);

  dataset  = H5Dopen2(mOutputFileId, "com_w", H5P_DEFAULT);
  dataspace = H5Dget_space(dataset);

  H5Sselect_elements(dataspace, H5S_SELECT_SET, 1, &coords);

  H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, &com.w);

  H5Sclose(dataspace);
  H5Dclose(dataset);
}//end of writeCom
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write final Centre of Mass
 * @param com
 */
void H5Helper::writeComFinal(const float4& com)
{

  hid_t dataset  = H5Dopen2(mOutputFileId, "com_x_final", H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &com.x);
  H5Dclose(dataset);

  dataset  = H5Dopen2(mOutputFileId, "com_y_final", H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &com.y);
  H5Dclose(dataset);

  dataset  = H5Dopen2(mOutputFileId, "com_z_final", H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &com.z);
  H5Dclose(dataset);

  dataset  = H5Dopen2(mOutputFileId, "com_w_final", H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &com.w);
  H5Dclose(dataset);
}// end of writeComFinal
//----------------------------------------------------------------------------------------------------------------------
