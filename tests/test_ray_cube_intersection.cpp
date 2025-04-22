#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include "utils.h" // Include the header with ray_cube_intersection_joseph()

void test_ray_cube_intersection()
{
  // Define test cases
  struct TestCase
  {
    float xstart[3];
    float xend[3];
    int expected_start_plane;
    int expected_end_plane;
    int expected_direction;
  };

  // Define voxel size, image dimensions, and origin (same as in ray_cube.py)
  float voxsize[3] = {2.0f, 1.0f, 4.0f};
  int img_dim[3] = {6, 12, 3};
  float img_origin[3] = {
      -voxsize[0] * img_dim[0] / 2 + 0.5f * voxsize[0] + 1.0f,
      -voxsize[1] * img_dim[1] / 2 + 0.5f * voxsize[1] + 1.0f,
      -voxsize[2] * img_dim[2] / 2 + 0.5f * voxsize[2] + 1.0f};

  // Test cases (translated from ray_cube.py)
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

  // Run test cases
  for (size_t i = 0; i < test_cases.size(); ++i)
  {
    const auto &tc = test_cases[i];

    int direction;
    float correction;
    int start_plane, end_plane;

    float eps = 1e-6f;

    // Call the function under test
    ray_cube_intersection_joseph(tc.xstart, tc.xend, img_origin, voxsize, img_dim,
                                 direction, correction, start_plane, end_plane);

    // Validate results
    std::cout << "Test case " << i + 1 << ":\n";
    std::cout << "  Expected direction: " << tc.expected_direction << ", Got: " << direction << "\n";
    std::cout << "  Expected start_plane: " << tc.expected_start_plane << ", Got: " << start_plane << "\n";
    std::cout << "  Expected end_plane: " << tc.expected_end_plane << ", Got: " << end_plane << "\n";

    assert(direction == tc.expected_direction);
    assert(start_plane == tc.expected_start_plane);
    assert(end_plane == tc.expected_end_plane);

    // Additional validation for correction factor
    float expected_correction = 1.0f;

    // the correction factor is only calculated if the ray intersects the cube
    // otherwise it is set to 1.0
    if (start_plane != -1 && end_plane != -1)
    {
      float dr[3] = {
          tc.xend[0] - tc.xstart[0],
          tc.xend[1] - tc.xstart[1],
          tc.xend[2] - tc.xstart[2]};
      float dr_length = std::sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]);
      float cs = (std::fabs(dr[direction]) / dr_length);
      expected_correction = voxsize[direction] / cs;
    }

    std::cout << "  Expected correction: " << expected_correction << ", Got: " << correction << "\n";
    assert(std::fabs(correction - expected_correction) < eps);

    std::cout << "  Test case " << i + 1 << " passed.\n";
  }

  std::cout << "All test cases passed.\n";
}

int main()
{
  test_ray_cube_intersection();
  return 0;
}
