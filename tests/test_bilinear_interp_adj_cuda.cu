// test_bilinear_cuda.cu
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include "utils.h"

// ----------------------------------------------------------------------------
// Minimal kernels: each one computes fwd = A(x) and then scatters A^T(y) into grad.
// ----------------------------------------------------------------------------

__global__ void kernel_test_fixed0(
    const float *img, float *grad,
    int n0, int n1, int n2,
    int i0, float i_f1, float i_f2,
    float y,
    float *out_fwd)
{
    // forward
    float fwd = bilinear_interp_fixed0<float>(img, n0, n1, n2, i0, i_f1, i_f2);
    out_fwd[0] = fwd;
    // adjoint
    bilinear_interp_adj_fixed0<float>(grad, n0, n1, n2, i0, i_f1, i_f2, y);
}

__global__ void kernel_test_fixed1(
    const float *img, float *grad,
    int n0, int n1, int n2,
    float i_f0, int i1, float i_f2,
    float y,
    float *out_fwd)
{
    float fwd = bilinear_interp_fixed1<float>(img, n0, n1, n2, i_f0, i1, i_f2);
    out_fwd[0] = fwd;
    bilinear_interp_adj_fixed1<float>(grad, n0, n1, n2, i_f0, i1, i_f2, y);
}

__global__ void kernel_test_fixed2(
    const float *img, float *grad,
    int n0, int n1, int n2,
    float i_f0, float i_f1, int i2,
    float y,
    float *out_fwd)
{
    float fwd = bilinear_interp_fixed2<float>(img, n0, n1, n2, i_f0, i_f1, i2);
    out_fwd[0] = fwd;
    bilinear_interp_adj_fixed2<float>(grad, n0, n1, n2, i_f0, i_f1, i2, y);
}

// ----------------------------------------------------------------------------
// Host‐side driver
// ----------------------------------------------------------------------------
static float host_dot(const std::vector<float> &a,
                      const std::vector<float> &b)
{
    assert(a.size() == b.size());
    float s = 0;
    for (size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}

int main()
{
    constexpr int n0 = 2, n1 = 3, n2 = 4;
    constexpr int N = n0 * n1 * n2;
    constexpr float eps = 1e-4f;

    // Build a simple host image x = [0,1,2,...]
    std::vector<float> host_x(N);
    for (int i = 0; i < N; ++i)
        host_x[i] = float(i);

    // Test cases: {dir, fixed‑index, f1, f2, y}
    struct TC
    {
        int dir;
        int fixed;
        float f1, f2, y;
    };
    std::vector<TC> tcs = {
        {0, 0, 0.3f, 1.7f, 2.5f},
        {0, 0, -0.6f, 3.7f, 2.5f},
        {0, 1, 1.2f, 2.8f, -1.3f},
        {1, 1, 0.5f, 0.4f, 3.0f},
        {1, 0, 1.8f, 2.1f, -0.7f},
        {2, 2, 0.9f, 1.1f, 4.6f},
        {2, 3, 0.9f, 1.1f, 4.6f},
        {2, 0, 2.3f, 0.6f, -2.2f}};

    // --- PASS 1: managed memory ---
    std::cout << "[PASS 1] Managed memory tests\n";
    {
        float *img_man, *grad_man, *out_man;
        cudaMallocManaged(&img_man, N * sizeof(float));
        cudaMallocManaged(&grad_man, N * sizeof(float));
        cudaMallocManaged(&out_man, sizeof(float));
        // copy host->managed
        for (int i = 0; i < N; ++i)
            img_man[i] = host_x[i];

        for (auto &tc : tcs)
        {
            // zero grad
            for (int i = 0; i < N; ++i)
                grad_man[i] = 0.f;
            cudaDeviceSynchronize();

            // pick the right kernel
            if (tc.dir == 0)
                kernel_test_fixed0<<<1, 1>>>(img_man, grad_man, n0, n1, n2,
                                             tc.fixed, tc.f1, tc.f2,
                                             tc.y, out_man);
            else if (tc.dir == 1)
                kernel_test_fixed1<<<1, 1>>>(img_man, grad_man, n0, n1, n2,
                                             tc.f1, tc.fixed, tc.f2,
                                             tc.y, out_man);
            else
                kernel_test_fixed2<<<1, 1>>>(img_man, grad_man, n0, n1, n2,
                                             tc.f1, tc.f2, tc.fixed,
                                             tc.y, out_man);

            cudaDeviceSynchronize();

            float fwd = out_man[0];
            float lhs = tc.y * fwd;

            // copy grad back to host array
            std::vector<float> host_grad(N);
            for (int i = 0; i < N; ++i)
                host_grad[i] = grad_man[i];
            float rhs = host_dot(host_x, host_grad);

            std::cout << " dir=" << tc.dir
                      << " fixed=" << tc.fixed
                      << " (<A x,y>=" << lhs
                      << ", <x,A^T y>=" << rhs << ")\n";
            assert(fabs(lhs - rhs) < eps);
        }

        cudaFree(img_man);
        cudaFree(grad_man);
        cudaFree(out_man);
    }

    // --- PASS 2: pure device memory ---
    std::cout << "[PASS 2] Device‐only memory tests\n";
    {
        float *d_img, *d_grad, *d_out;
        cudaMalloc(&d_img, N * sizeof(float));
        cudaMalloc(&d_grad, N * sizeof(float));
        cudaMalloc(&d_out, sizeof(float));
        // copy host->device
        cudaMemcpy(d_img, host_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        for (auto &tc : tcs)
        {
            // zero grad on device
            cudaMemset(d_grad, 0, N * sizeof(float));
            cudaDeviceSynchronize();

            if (tc.dir == 0)
                kernel_test_fixed0<<<1, 1>>>(d_img, d_grad, n0, n1, n2,
                                             tc.fixed, tc.f1, tc.f2,
                                             tc.y, d_out);
            else if (tc.dir == 1)
                kernel_test_fixed1<<<1, 1>>>(d_img, d_grad, n0, n1, n2,
                                             tc.f1, tc.fixed, tc.f2,
                                             tc.y, d_out);
            else
                kernel_test_fixed2<<<1, 1>>>(d_img, d_grad, n0, n1, n2,
                                             tc.f1, tc.f2, tc.fixed,
                                             tc.y, d_out);

            cudaDeviceSynchronize();

            float fwd;
            cudaMemcpy(&fwd, d_out, sizeof(float), cudaMemcpyDeviceToHost);
            float lhs = tc.y * fwd;

            // copy grad back
            std::vector<float> host_grad(N);
            cudaMemcpy(host_grad.data(), d_grad, N * sizeof(float), cudaMemcpyDeviceToHost);
            float rhs = host_dot(host_x, host_grad);

            std::cout << " dir=" << tc.dir
                      << " fixed=" << tc.fixed
                      << " (<A x,y>=" << lhs
                      << ", <x,A^T y>=" << rhs << ")\n";
            assert(fabs(lhs - rhs) < eps);
        }

        cudaFree(d_img);
        cudaFree(d_grad);
        cudaFree(d_out);
    }

    std::cout << "All CUDA adjoint‐property tests passed\n";
    return 0;
}
