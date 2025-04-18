#include "utils.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Kernel to test bilinear_interp_fixed0
__global__ void test_bilinear_interp_fixed0_kernel(const float *img, int n0, int n1, int n2, int i0, float i_f1, float i_f2, float *result)
{
    *result = bilinear_interp_fixed0<float>(img, n0, n1, n2, i0, i_f1, i_f2);
}

// Kernel to test bilinear_interp_fixed1
__global__ void test_bilinear_interp_fixed1_kernel(const float *img, int n0, int n1, int n2, float i_f0, int i1, float i_f2, float *result)
{
    *result = bilinear_interp_fixed1<float>(img, n0, n1, n2, i_f0, i1, i_f2);
}

// Kernel to test bilinear_interp_fixed2
__global__ void test_bilinear_interp_fixed2_kernel(const float *img, int n0, int n1, int n2, float i_f0, float i_f1, int i2, float *result)
{
    *result = bilinear_interp_fixed2<float>(img, n0, n1, n2, i_f0, i_f1, i2);
}

// Helper function to run a single test for bilinear_interp_fixed0
void run_test_fixed0(const char *test_name, const float *img, int n0, int n1, int n2, int i0, float i_f1, float i_f2, float expected, float eps)
{
    std::cout << "Running test: " << test_name << "\n";

    // Allocate memory for the result on the device
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));

    // Launch the kernel
    test_bilinear_interp_fixed0_kernel<<<1, 1>>>(img, n0, n1, n2, i0, i_f1, i_f2, d_result);

    // Copy the result back to the host
    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_result);

    // Validate the result
    std::cout << "    fixed0 (i0=" << i0 << ", i_f1=" << i_f1 << ", i_f2=" << i_f2
              << ") => res=" << result << ", expected=" << expected << "\n";
    assert(std::fabs(result - expected) < eps);

    std::cout << "Test passed: " << test_name << "\n";
}

// Helper function to run a single test for bilinear_interp_fixed1
void run_test_fixed1(const char *test_name, const float *img, int n0, int n1, int n2, float i_f0, int i1, float i_f2, float expected, float eps)
{
    std::cout << "Running test: " << test_name << "\n";

    // Allocate memory for the result on the device
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));

    // Launch the kernel
    test_bilinear_interp_fixed1_kernel<<<1, 1>>>(img, n0, n1, n2, i_f0, i1, i_f2, d_result);

    // Copy the result back to the host
    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_result);

    // Validate the result
    std::cout << "    fixed1 (i_f0=" << i_f0 << ", i1=" << i1 << ", i_f2=" << i_f2
              << ") => res=" << result << ", expected=" << expected << "\n";
    assert(std::fabs(result - expected) < eps);

    std::cout << "Test passed: " << test_name << "\n";
}

// Helper function to run a single test for bilinear_interp_fixed2
void run_test_fixed2(const char *test_name, const float *img, int n0, int n1, int n2, float i_f0, float i_f1, int i2, float expected, float eps)
{
    std::cout << "Running test: " << test_name << "\n";

    // Allocate memory for the result on the device
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));

    // Launch the kernel
    test_bilinear_interp_fixed2_kernel<<<1, 1>>>(img, n0, n1, n2, i_f0, i_f1, i2, d_result);

    // Copy the result back to the host
    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_result);

    // Validate the result
    std::cout << "    fixed2 (i_f0=" << i_f0 << ", i_f1=" << i_f1 << ", i2=" << i2
              << ") => res=" << result << ", expected=" << expected << "\n";
    assert(std::fabs(result - expected) < eps);

    std::cout << "Test passed: " << test_name << "\n";
}

int main()
{
    // Dimensions matching np.arange(24).reshape(2,3,4)
    const int n0 = 2, n1 = 3, n2 = 4;
    const int num_elements = n0 * n1 * n2;
    float eps = 1e-6f;

    // Host array
    std::vector<float> host_img(num_elements);
    for (int i = 0; i < num_elements; ++i)
    {
        host_img[i] = static_cast<float>(i);
    }

    // Allocate managed memory
    float *managed_img;
    cudaMallocManaged(&managed_img, num_elements * sizeof(float));
    cudaMemcpy(managed_img, host_img.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Test 1: bilinear_interp_fixed0
    run_test_fixed0("bilinear_interp_fixed0", managed_img, n0, n1, n2, 0, 0.3f, 1.8f, 3.0f, eps);

    // Test 2: bilinear_interp_fixed1
    run_test_fixed1("bilinear_interp_fixed1", managed_img, n0, n1, n2, 0.5f, 1, 2.5f, 12.5f, eps);

    // Test 3: bilinear_interp_fixed2
    run_test_fixed2("bilinear_interp_fixed2", managed_img, n0, n1, n2, 0.5f, 1.5f, 2, 14.0f, eps);

    cudaFree(managed_img);

    std::cout << "All CUDA bilinear interpolation tests passed.\n";
    return 0;
}
