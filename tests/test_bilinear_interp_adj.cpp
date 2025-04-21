#include "utils.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

// Helper to compute dot product of two vectors of equal size
float dot(const std::vector<float> &a, const std::vector<float> &b)
{
    assert(a.size() == b.size());
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        sum += a[i] * b[i];
    return sum;
}

int main()
{
    // 3D image dims
    const int n0 = 2, n1 = 3, n2 = 4;
    const size_t N = n0 * n1 * n2;

    float eps = 1e-5f; // Tolerance for floating point comparison

    // Create a sample image x with distinct values
    std::vector<float> x(N);
    for (size_t i = 0; i < N; ++i)
        x[i] = float(i + 1); // 1..N

    // Test several fractional locations and directions
    struct TestCase
    {
        int dir;
        int fixed;
        float f1, f2;
        float y;
    };
    std::vector<TestCase> cases = {
        {0, 0, 0.3f, 1.7f, 2.5f},
        {0, 0, -0.6f, 3.7f, 2.5f},
        {0, 1, 1.2f, 2.8f, -1.3f},
        {1, 1, 0.5f, 0.4f, 3.0f},
        {1, 0, 1.8f, 2.1f, -0.7f},
        {2, 2, 0.9f, 1.1f, 4.6f},
        {2, 3, 0.9f, 1.1f, 4.6f},
        {2, 0, 2.3f, 0.6f, -2.2f}};

    for (auto &tc : cases)
    {
        // Forward: compute A x (a scalar) depending on dir
        float fwd = 0.0f;
        if (tc.dir == 0)
        {
            fwd = bilinear_interp_fixed0<float>(x.data(), n0, n1, n2,
                                                tc.fixed, tc.f1, tc.f2);
        }
        else if (tc.dir == 1)
        {
            fwd = bilinear_interp_fixed1<float>(x.data(), n0, n1, n2,
                                                tc.f1, tc.fixed, tc.f2);
        }
        else
        {
            fwd = bilinear_interp_fixed2<float>(x.data(), n0, n1, n2,
                                                tc.f1, tc.f2, tc.fixed);
        }

        // Construct y vector of same length as A x (here 1), y is scalar tc.y
        float lhs = tc.y * fwd; // <A x, y> since Ax is scalar

        // Adjoint: apply A^T y to zero image
        std::vector<float> x_adj(N, 0.0f);
        if (tc.dir == 0)
        {
            bilinear_interp_adj_fixed0<float>(x_adj.data(), n0, n1, n2,
                                              tc.fixed, tc.f1, tc.f2, tc.y);
        }
        else if (tc.dir == 1)
        {
            bilinear_interp_adj_fixed1<float>(x_adj.data(), n0, n1, n2,
                                              tc.f1, tc.fixed, tc.f2, tc.y);
        }
        else
        {
            bilinear_interp_adj_fixed2<float>(x_adj.data(), n0, n1, n2,
                                              tc.f1, tc.f2, tc.fixed, tc.y);
        }

        // Compute <x, A^T y> = dot(x, x_adj)
        float rhs = dot(x, x_adj);

        std::cout << "dir=" << tc.dir
                  << " fixed=" << tc.fixed
                  << " (" << tc.f1 << "," << tc.f2 << ")"
                  << " y=" << tc.y
                  << "  <A x,y>=" << lhs
                  << "  <x,A^T y>=" << rhs << "\n";

        // Compare
        assert(std::fabs(lhs - rhs) < eps);
    }

    std::cout << "Adjoint property tests passed.\n";
    return 0;
}
