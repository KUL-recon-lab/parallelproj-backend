#include "utils.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main()
{
    // Dimensions matching np.arange(24).reshape(2,3,4)
    const int n0 = 2, n1 = 3, n2 = 4;
    float eps = 1e-6f;

    std::vector<float> img(n0 * n1 * n2);
    for (int i = 0; i < n0 * n1 * n2; ++i)
    {
        img[i] = static_cast<float>(i);
    }

    // 1) Test bilinear_interp_fixed0 on plane i0=0
    //    at i_f1=0.5, i_f2=1.5 -> expected = (1 + 5 + 2 + 6) / 4 = 3.5
    {
        int i0 = 0;
        float i_f1 = 0.5f, i_f2 = 1.5f;
        float expected = 3.5f;
        float res = bilinear_interp_fixed0<float>(img.data(), n0, n1, n2, i0, i_f1, i_f2);
        std::cout << "fixed0 (i0=" << i0 << ", i_f1=" << i_f1 << ", i_f2=" << i_f2
                  << ") => res=" << res << ", expected=" << expected << "\n";
        assert(std::fabs(res - expected) < eps);
    }

    // 2) Test bilinear_interp_fixed1 on plane i1=1
    //    at i_f0=0.5, i_f2=2.5 -> expected = (6 + 18 + 7 + 19) / 4 = 12.5
    {
        int i1 = 1;
        float i_f0 = 0.5f, i_f2 = 2.5f;
        float expected = 12.5f;
        float res = bilinear_interp_fixed1<float>(img.data(), n0, n1, n2, i_f0, i1, i_f2);
        std::cout << "fixed1 (i_f0=" << i_f0 << ", i1=" << i1 << ", i_f2=" << i_f2
                  << ") => res=" << res << ", expected=" << expected << "\n";
        assert(std::fabs(res - expected) < eps);
    }

    // 3) Test bilinear_interp_fixed2 on plane i2=2
    //    at i_f0=0.5, i_f1=1.5 -> expected = (6 + 18 + 10 + 22) / 4 = 14.0
    {
        int i2 = 2;
        float i_f0 = 0.5f, i_f1 = 1.5f;
        float expected = 14.0f;
        float res = bilinear_interp_fixed2<float>(img.data(), n0, n1, n2, i_f0, i_f1, i2);
        std::cout << "fixed2 (i_f0=" << i_f0 << ", i_f1=" << i_f1 << ", i2=" << i2
                  << ") => res=" << res << ", expected=" << expected << "\n";
        assert(std::fabs(res - expected) < eps);
    }

    // 4) Test out-of-bounds handling on fixed0 (i1 < 0)
    //    at i0=0, i_f1=-0.5, i_f2=1.0 -> expected = 0.5
    {
        int i0 = 0;
        float i_f1 = -0.5f, i_f2 = 1.0f;
        float expected = 0.5f;
        float res = bilinear_interp_fixed0<float>(img.data(), n0, n1, n2, i0, i_f1, i_f2);
        std::cout << "fixed0 OOB (i0=" << i0 << ", i_f1=" << i_f1 << ", i_f2=" << i_f2
                  << ") => res=" << res << ", expected=" << expected << "\n";
        assert(std::fabs(res - expected) < eps);
    }

    // 5) Test non-.5 fractions for fixed0
    //    at i0=0, i_f1=0.3, i_f2=1.8 -> expected = 3.0
    {
        int i0 = 0;
        float i_f1 = 0.3f, i_f2 = 1.8f;
        float expected = 3.0f;
        float res = bilinear_interp_fixed0<float>(img.data(), n0, n1, n2, i0, i_f1, i_f2);
        std::cout << "fixed0 (i0=" << i0 << ", i_f1=" << i_f1 << ", i_f2=" << i_f2
                  << ") => res=" << res << ", expected=" << expected << "\n";
        assert(std::fabs(res - expected) < eps);
    }

    // 6) Test non-.5 fractions for fixed1
    //    at i1=1, i_f0=0.8, i_f2=2.3 -> expected = 15.9
    {
        int i1 = 1;
        float i_f0 = 0.8f, i_f2 = 2.3f;
        float expected = 15.9f;
        float res = bilinear_interp_fixed1<float>(img.data(), n0, n1, n2, i_f0, i1, i_f2);
        std::cout << "fixed1 (i_f0=" << i_f0 << ", i1=" << i1 << ", i_f2=" << i_f2
                  << ") => res=" << res << ", expected=" << expected << "\n";
        assert(std::fabs(res - expected) < eps);
    }

    // 7) Test non-.5 fractions for fixed2
    //    at i2=1, i_f0=0.9, i_f1=1.4 -> expected = 17.4
    {
        int i2 = 1;
        float i_f0 = 0.9f, i_f1 = 1.4f;
        float expected = 17.4f;
        float res = bilinear_interp_fixed2<float>(img.data(), n0, n1, n2, i_f0, i_f1, i2);
        std::cout << "fixed2 (i_f0=" << i_f0 << ", i_f1=" << i_f1 << ", i2=" << i2
                  << ") => res=" << res << ", expected=" << expected << "\n";
        assert(std::fabs(res - expected) < eps);
    }

    std::cout << "All bilinear interpolation tests passed.\n";
    return 0;
}
