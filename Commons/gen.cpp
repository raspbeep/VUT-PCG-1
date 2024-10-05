/**
 * @file      gen.cpp
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

#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <hdf5_hl.h>
#include <hdf5.h>

/**
 * main routine
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char** argv)
{
  // parse commandline parameters
  if (argc != 3)
  {
    std::fprintf(stderr, "Usage: gen <N> <output>\n");
    std::exit(1);
  }

  const unsigned N = std::stoul(argv[1]);

  // allocate memory
  std::vector<float> pos_x(N);
  std::vector<float> pos_y(N);
  std::vector<float> pos_z(N);
  std::vector<float> vel_x(N);
  std::vector<float> vel_y(N);
  std::vector<float> vel_z(N);
  std::vector<float> weight(N);

  // print parameters
  std::printf("N: %u\n", N);

  // Create output file
  hid_t file_id = H5Fcreate(argv[2], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file_id == H5I_INVALID_HID)
  {
    std::fprintf(stderr, "Can't open file %s!\n", argv[2]);
    std::exit(1);
  }

  hsize_t size = N;
  hid_t dataspace = H5Screate_simple(1, &size, nullptr);

  // Random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(std::numeric_limits<float>::min(), 1.f);

  // Generate random values
  for (unsigned i = 0u; i < N; i++)
  {
    pos_x[i]  = dis(gen) * 100.0f;
    pos_z[i]  = dis(gen) * 100.0f;
    pos_y[i]  = dis(gen) * 100.0f;
    vel_x[i]  = dis(gen) * 4.0f - 2.0f;
    vel_y[i]  = dis(gen) * 4.0f - 2.0f;
    vel_z[i]  = dis(gen) * 4.0f - 2.0f;
    weight[i] = dis(gen) * 2500000000.0f;
  }

  // Lambda to write a dataset
  auto writeDataset = [=](const char* datasetName, std::vector<float>& vector)
  {
    hid_t dataset = H5Dcreate2(file_id, datasetName, H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vector.data());
    H5Dclose(dataset);
  };// end of v

  // write necessary datasets
  writeDataset("pos_x", pos_x);
  writeDataset("pos_y", pos_y);
  writeDataset("pos_z", pos_z);

  writeDataset("vel_x", vel_x);
  writeDataset("vel_y", vel_y);
  writeDataset("vel_z", vel_z);

  writeDataset("weight", weight);

  // close dataspace
  H5Sclose(dataspace);

  // close file
  H5Fclose(file_id);
}// end of main
//----------------------------------------------------------------------------------------------------------------------
