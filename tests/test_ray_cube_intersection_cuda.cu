#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <cuda_runtime.h>
#include "utils.h" // Include the header with ray_cube_intersection_joseph()

// CUDA kernel to test ray_cube_intersection_joseph on the device
__global__ void test_ray_cube_intersection_kernel(const float *xstart, const float *xend,
                                                  const float *img_origin, const float *voxsize,
                                                  const int *img_dim, int *directions,
                                                  float *corrections, int *start_planes, int *end_planes,
                                                  int num_tests)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_tests)
    return;

  // Call the function under test
  ray_cube_intersection_joseph(&xstart[idx * 3], &xend[idx * 3], img_origin, voxsize, img_dim,
                               directions[idx], corrections[idx], start_planes[idx], end_planes[idx]);
}

// Function to test with device arrays
void test_ray_cube_intersection_device()
{
  std::cout << "Running test with device arrays...\n";

  // Define test cases
  struct TestCase
  {
    float xstart[3];
    float xend[3];
    int expected_start_plane;
    int expected_end_plane;
    int expected_direction;
  };

  float voxsize[3] = {2.0f, 1.0f, 4.0f};
  int img_dim[3] = {6, 12, 3};
  float img_origin[3] = {
      -voxsize[0] * img_dim[0] / 2 + 0.5f * voxsize[0] + 1.0f,
      -voxsize[1] * img_dim[1] / 2 + 0.5f * voxsize[1] + 1.0f,
      -voxsize[2] * img_dim[2] / 2 + 0.5f * voxsize[2] + 1.0f};

  std::vector<TestCase> test_cases = {
      {{-14.0f, 0.0f, 0.0f}, {14.0f, 0.0f, 0.0f}, 0, 5, 0},
      {{14.0f, 0.0f, 0.0f}, {-14.0f, 0.0f, 0.0f}, 0, 5, 0},
      {{-14.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, 0, 3, 0},
      {{2.0f, 0.0f, 0.0f}, {14.0f, 0.0f, 0.0f}, 4, 5, 0},
      {{0.0f, -14.0f, 0.0f}, {0.0f, 14.0f, 0.0f}, 0, 11, 1},
      {{0.0f, 14.0f, 0.0f}, {0.0f, -14.0f, 0.0f}, 0, 11, 1},
      {{0.0f, 0.0f, -14.0f}, {0.0f, 0.0f, 14.0f}, 0, 2, 2},
      {{0.0f, 0.0f, -14.0f}, {0.0f, 0.0f, 1.0f}, 0, 1, 2},
      {{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 14.0f}, 2, 2, 2},
      {{-14.0f, -14.0f, -14.0f}, {0.0f, 0.0f, 1.0f}, 0, 1, 2},
      {{-15.1f, -14.0f, -14.0f}, {0.0f, 0.0f, 1.0f}, 0, 2, 0},
      {{0.0f, 0.0f, 1.0f}, {-15.1f, -14.0f, -14.0f}, 0, 2, 0},
      //{{-4.5f, 7.0f, 0.0f}, {6.5f, 7.0f, 0.0f}, -1, -1, 0}, // Ray touching the cube on one side, behavior not well defined, do not test
      //{{-4.5f, -5.0f, 0.0f}, {6.5f, -5.0f, 0.0f}, 0, 5, 0}, // Ray touching the cube on other side, behavior not well defined, do not test
  };

  int num_tests = test_cases.size();

  // Allocate and copy test data to device
  float *d_xstart, *d_xend, *d_img_origin, *d_voxsize;
  int *d_img_dim, *d_directions, *d_start_planes, *d_end_planes;
  float *d_corrections;

  cudaMalloc(&d_xstart, num_tests * 3 * sizeof(float));
  cudaMalloc(&d_xend, num_tests * 3 * sizeof(float));
  cudaMalloc(&d_img_origin, 3 * sizeof(float));
  cudaMalloc(&d_voxsize, 3 * sizeof(float));
  cudaMalloc(&d_img_dim, 3 * sizeof(int));
  cudaMalloc(&d_directions, num_tests * sizeof(int));
  cudaMalloc(&d_corrections, num_tests * sizeof(float));
  cudaMalloc(&d_start_planes, num_tests * sizeof(int));
  cudaMalloc(&d_end_planes, num_tests * sizeof(int));

  std::vector<float> h_xstart(num_tests * 3), h_xend(num_tests * 3);
  for (int i = 0; i < num_tests; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      h_xstart[i * 3 + j] = test_cases[i].xstart[j];
      h_xend[i * 3 + j] = test_cases[i].xend[j];
    }
  }

  cudaMemcpy(d_xstart, h_xstart.data(), num_tests * 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_xend, h_xend.data(), num_tests * 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_img_origin, img_origin, 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_voxsize, voxsize, 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_img_dim, img_dim, 3 * sizeof(int), cudaMemcpyHostToDevice);

  // Launch kernel
  int threads_per_block = 256;
  int num_blocks = (num_tests + threads_per_block - 1) / threads_per_block;
  test_ray_cube_intersection_kernel<<<num_blocks, threads_per_block>>>(
      d_xstart, d_xend, d_img_origin, d_voxsize, d_img_dim,
      d_directions, d_corrections, d_start_planes, d_end_planes, num_tests);

  // Copy results back to host
  std::vector<int> h_directions(num_tests), h_start_planes(num_tests), h_end_planes(num_tests);
  std::vector<float> h_corrections(num_tests);
  cudaMemcpy(h_directions.data(), d_directions, num_tests * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_corrections.data(), d_corrections, num_tests * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_start_planes.data(), d_start_planes, num_tests * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_end_planes.data(), d_end_planes, num_tests * sizeof(int), cudaMemcpyDeviceToHost);

  // Validate results
  for (int i = 0; i < num_tests; ++i)
  {
    const auto &tc = test_cases[i];
    std::cout << "Test case " << i + 1 << ":\n";
    std::cout << "  Expected direction: " << tc.expected_direction << ", Got: " << h_directions[i] << "\n";
    std::cout << "  Expected start_plane: " << tc.expected_start_plane << ", Got: " << h_start_planes[i] << "\n";
    std::cout << "  Expected end_plane: " << tc.expected_end_plane << ", Got: " << h_end_planes[i] << "\n";

    assert(h_directions[i] == tc.expected_direction);
    assert(h_start_planes[i] == tc.expected_start_plane);
    assert(h_end_planes[i] == tc.expected_end_plane);
  }

  std::cout << "All CUDA test cases passed with device arrays.\n";

  // Free device memory
  cudaFree(d_xstart);
  cudaFree(d_xend);
  cudaFree(d_img_origin);
  cudaFree(d_voxsize);
  cudaFree(d_img_dim);
  cudaFree(d_directions);
  cudaFree(d_corrections);
  cudaFree(d_start_planes);
  cudaFree(d_end_planes);
}

// Function to test with CUDA managed memory
void test_ray_cube_intersection_managed()
{
  std::cout << "Running test with CUDA managed memory...\n";

  // Define test cases
  struct TestCase
  {
    float xstart[3];
    float xend[3];
    int expected_start_plane;
    int expected_end_plane;
    int expected_direction;
  };

  float voxsize[3] = {2.0f, 1.0f, 4.0f};
  int img_dim[3] = {6, 12, 3};
  float img_origin[3] = {
      -voxsize[0] * img_dim[0] / 2 + 0.5f * voxsize[0] + 1.0f,
      -voxsize[1] * img_dim[1] / 2 + 0.5f * voxsize[1] + 1.0f,
      -voxsize[2] * img_dim[2] / 2 + 0.5f * voxsize[2] + 1.0f};

  std::vector<TestCase> test_cases = {
      {{-14.0f, 0.0f, 0.0f}, {14.0f, 0.0f, 0.0f}, 0, 5, 0},
      {{14.0f, 0.0f, 0.0f}, {-14.0f, 0.0f, 0.0f}, 0, 5, 0},
      {{-14.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, 0, 3, 0},
      {{2.0f, 0.0f, 0.0f}, {14.0f, 0.0f, 0.0f}, 4, 5, 0},
      {{0.0f, -14.0f, 0.0f}, {0.0f, 14.0f, 0.0f}, 0, 11, 1},
      {{0.0f, 14.0f, 0.0f}, {0.0f, -14.0f, 0.0f}, 0, 11, 1},
      {{0.0f, 0.0f, -14.0f}, {0.0f, 0.0f, 14.0f}, 0, 2, 2},
      {{0.0f, 0.0f, -14.0f}, {0.0f, 0.0f, 1.0f}, 0, 1, 2},
      {{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 14.0f}, 2, 2, 2},
      {{-14.0f, -14.0f, -14.0f}, {0.0f, 0.0f, 1.0f}, 0, 1, 2},
      {{-15.1f, -14.0f, -14.0f}, {0.0f, 0.0f, 1.0f}, 0, 2, 0},
      {{0.0f, 0.0f, 1.0f}, {-15.1f, -14.0f, -14.0f}, 0, 2, 0},
      //{{-4.5f, 7.0f, 0.0f}, {6.5f, 7.0f, 0.0f}, -1, -1, 0}, // Ray touching the cube on one side, behavior not well defined, do not test
      //{{-4.5f, -5.0f, 0.0f}, {6.5f, -5.0f, 0.0f}, 0, 5, 0}, // Ray touching the cube on other side, behavior not well defined, do not test
  };

  int num_tests = test_cases.size();

  // Allocate managed memory
  float *xstart, *xend, *img_origin_managed, *voxsize_managed;
  int *img_dim_managed, *directions, *start_planes, *end_planes;
  float *corrections;

  cudaMallocManaged(&xstart, num_tests * 3 * sizeof(float));
  cudaMallocManaged(&xend, num_tests * 3 * sizeof(float));
  cudaMallocManaged(&img_origin_managed, 3 * sizeof(float));
  cudaMallocManaged(&voxsize_managed, 3 * sizeof(float));
  cudaMallocManaged(&img_dim_managed, 3 * sizeof(int));
  cudaMallocManaged(&directions, num_tests * sizeof(int));
  cudaMallocManaged(&corrections, num_tests * sizeof(float));
  cudaMallocManaged(&start_planes, num_tests * sizeof(int));
  cudaMallocManaged(&end_planes, num_tests * sizeof(int));

  for (int i = 0; i < num_tests; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      xstart[i * 3 + j] = test_cases[i].xstart[j];
      xend[i * 3 + j] = test_cases[i].xend[j];
    }
  }

  for (int i = 0; i < 3; ++i)
  {
    img_origin_managed[i] = img_origin[i];
    voxsize_managed[i] = voxsize[i];
    img_dim_managed[i] = img_dim[i];
  }

  // Launch kernel
  int threads_per_block = 256;
  int num_blocks = (num_tests + threads_per_block - 1) / threads_per_block;
  test_ray_cube_intersection_kernel<<<num_blocks, threads_per_block>>>(
      xstart, xend, img_origin_managed, voxsize_managed, img_dim_managed,
      directions, corrections, start_planes, end_planes, num_tests);

  cudaDeviceSynchronize();

  // Validate results
  for (int i = 0; i < num_tests; ++i)
  {
    const auto &tc = test_cases[i];
    std::cout << "Test case " << i + 1 << ":\n";
    std::cout << "  Expected direction: " << tc.expected_direction << ", Got: " << directions[i] << "\n";
    std::cout << "  Expected start_plane: " << tc.expected_start_plane << ", Got: " << start_planes[i] << "\n";
    std::cout << "  Expected end_plane: " << tc.expected_end_plane << ", Got: " << end_planes[i] << "\n";

    assert(directions[i] == tc.expected_direction);
    assert(start_planes[i] == tc.expected_start_plane);
    assert(end_planes[i] == tc.expected_end_plane);
  }

  std::cout << "All CUDA test cases passed with managed memory.\n";

  // Free managed memory
  cudaFree(xstart);
  cudaFree(xend);
  cudaFree(img_origin_managed);
  cudaFree(voxsize_managed);
  cudaFree(img_dim_managed);
  cudaFree(directions);
  cudaFree(corrections);
  cudaFree(start_planes);
  cudaFree(end_planes);
}

int main()
{
  test_ray_cube_intersection_device();
  test_ray_cube_intersection_managed();
  return 0;
}
