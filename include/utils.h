#pragma once
#include <cmath>
#include "cuda_compat.h"
#include "utils.h" // for atomic_sum

WORKER_QUALIFIER inline void atomic_sum(float *target, float value)
{
#ifdef __CUDACC__
  atomicAdd(target, value);
#else
#pragma omp atomic
  *target += value;
#endif
}

WORKER_QUALIFIER inline unsigned char ray_cube_intersection(float orig0,
                                                            float orig1,
                                                            float orig2,
                                                            float bounds0_min,
                                                            float bounds1_min,
                                                            float bounds2_min,
                                                            float bounds0_max,
                                                            float bounds1_max,
                                                            float bounds2_max,
                                                            float rdir0,
                                                            float rdir1,
                                                            float rdir2,
                                                            float *t1,
                                                            float *t2)
{
  // the inverse of the directional vector
  // using the inverse of the directional vector and IEEE floating point arith standard 754
  // makes sure that 0's in the directional vector are handled correctly
  float invdir0 = 1.f / rdir0;
  float invdir1 = 1.f / rdir1;
  float invdir2 = 1.f / rdir2;

  unsigned char intersec = 1;

  float t11, t12, t21, t22;

  if (invdir0 >= 0)
  {
    *t1 = (bounds0_min - orig0) * invdir0;
    *t2 = (bounds0_max - orig0) * invdir0;
  }
  else
  {
    *t1 = (bounds0_max - orig0) * invdir0;
    *t2 = (bounds0_min - orig0) * invdir0;
  }

  if (invdir1 >= 0)
  {
    t11 = (bounds1_min - orig1) * invdir1;
    t12 = (bounds1_max - orig1) * invdir1;
  }
  else
  {
    t11 = (bounds1_max - orig1) * invdir1;
    t12 = (bounds1_min - orig1) * invdir1;
  }

  if ((*t1 > t12) || (t11 > *t2))
  {
    intersec = 0;
  }
  if (t11 > *t1)
  {
    *t1 = t11;
  }
  if (t12 < *t2)
  {
    *t2 = t12;
  }

  if (invdir2 >= 0)
  {
    t21 = (bounds2_min - orig2) * invdir2;
    t22 = (bounds2_max - orig2) * invdir2;
  }
  else
  {
    t21 = (bounds2_max - orig2) * invdir2;
    t22 = (bounds2_min - orig2) * invdir2;
  }

  if ((*t1 > t22) || (t21 > *t2))
  {
    intersec = 0;
  }
  if (t21 > *t1)
  {
    *t1 = t21;
  }
  if (t22 < *t2)
  {
    *t2 = t22;
  }

  return (intersec);
}

/**
 * Three bilinear interpolate functions for 3d image wit dims (n0,n1,n2) with uniform signatures:
 *   - bilinear_interp_fixed_0: plane at fixed i0, interpolate in (i1,i2)
 *   - bilinear_interp_fixed_1: plane at fixed i1, interpolate in (i0,i2)
 *   - bilinear_interp_fixed_2: plane at fixed i2, interpolate in (i0,i1)
 *
 * All take the full 3D image pointer `img` and dimensions n0,n1,n2 at runtime.
 *
 * If compiled with CUDA, these functions are device functions.
 */

template <typename T = float>
WORKER_QUALIFIER inline T bilinear_interp_fixed0(const T *img,
                                                 int n0, int n1, int n2,
                                                 int i0,
                                                 float i_f1,
                                                 float i_f2)
{
  // get the i0 image plane (contiguous in memory)
  const T *img_plane = img + size_t(i0) * n1 * n2;

  int i1_0 = int(floorf(i_f1));
  int i2_0 = int(floorf(i_f2));
  int i1_1 = i1_0 + 1;
  int i2_1 = i2_0 + 1;

  float w1 = i_f1 - i1_0;
  float w2 = i_f2 - i2_0;

  auto sample = [&](int i1, int i2) -> T
  {
    if (i1 < 0 || i1 >= n1 || i2 < 0 || i2 >= n2)
      return T(0);
    return img_plane[size_t(i1) * n2 + i2];
  };

  T v00 = sample(i1_0, i2_0);
  T v10 = sample(i1_1, i2_0);
  T v01 = sample(i1_0, i2_1);
  T v11 = sample(i1_1, i2_1);

  return v00 * (1 - w1) * (1 - w2) + v10 * (w1) * (1 - w2) + v01 * (1 - w1) * (w2) + v11 * (w1) * (w2);
}

template <typename T = float>
WORKER_QUALIFIER inline T bilinear_interp_fixed1(const T *img,
                                                 int n0, int n1, int n2,
                                                 float i_f0,
                                                 int i1,
                                                 float i_f2)
{
  int i0_0 = int(floorf(i_f0));
  int i2_0 = int(floorf(i_f2));
  int i0_1 = i0_0 + 1;
  int i2_1 = i2_0 + 1;

  float w0 = i_f0 - i0_0;
  float w2 = i_f2 - i2_0;

  auto sample = [&](int i0, int i2) -> T
  {
    if (i0 < 0 || i0 >= n0 || i2 < 0 || i2 >= n2)
      return T(0);
    size_t idx = size_t(i0) * n1 * n2 + size_t(i1) * n2 + i2;
    return img[idx];
  };

  T v00 = sample(i0_0, i2_0);
  T v10 = sample(i0_1, i2_0);
  T v01 = sample(i0_0, i2_1);
  T v11 = sample(i0_1, i2_1);

  return v00 * (1 - w0) * (1 - w2) + v10 * (w0) * (1 - w2) + v01 * (1 - w0) * (w2) + v11 * (w0) * (w2);
}

template <typename T = float>
WORKER_QUALIFIER inline T bilinear_interp_fixed2(const T *img,
                                                 int n0, int n1, int n2,
                                                 float i_f0,
                                                 float i_f1,
                                                 int i2)
{
  int i0_0 = int(floorf(i_f0));
  int i1_0 = int(floorf(i_f1));
  int i0_1 = i0_0 + 1;
  int i1_1 = i1_0 + 1;

  float w0 = i_f0 - i0_0;
  float w1 = i_f1 - i1_0;

  auto sample = [&](int i0, int i1) -> T
  {
    if (i0 < 0 || i0 >= n0 || i1 < 0 || i1 >= n1)
      return T(0);
    size_t idx = size_t(i0) * n1 * n2 + size_t(i1) * n2 + i2;
    return img[idx];
  };

  T v00 = sample(i0_0, i1_0);
  T v10 = sample(i0_1, i1_0);
  T v01 = sample(i0_0, i1_1);
  T v11 = sample(i0_1, i1_1);

  return v00 * (1 - w0) * (1 - w1) + v10 * (w0) * (1 - w1) + v01 * (1 - w0) * (w1) + v11 * (w0) * (w1);
}

/**
 * Three bilinear scatter (adjoint) functions for 3D image dims (n0,n1,n2):
 *   - bilinear_interp_adj_fixed0: plane at fixed i0, scatter into (i1,i2)
 *   - bilinear_interp_adj_fixed1: plane at fixed i1, scatter into (i0,i2)
 *   - bilinear_interp_adj_fixed2: plane at fixed i2, scatter into (i0,i1)
 *
 * All take the full 3D image pointer `img`, dimensions n0,n1,n2,
 * a fixed index, fractional coordinates, and a value to scatter back.
 * Uses atomic_sum to work correctly on CUDA (__device__) and OpenMP.
 */

template <typename T = float>
WORKER_QUALIFIER inline void bilinear_interp_adj_fixed0(
    T *img,
    int n0, int n1, int n2,
    int i0,
    float i_f1,
    float i_f2,
    T val)
{
  // pointer to the i0-th plane
  T *plane = img + size_t(i0) * n1 * n2;

  int i1_0 = int(floorf(i_f1));
  int i2_0 = int(floorf(i_f2));
  int i1_1 = i1_0 + 1;
  int i2_1 = i2_0 + 1;

  float w1 = i_f1 - i1_0;
  float w2 = i_f2 - i2_0;
  float w00 = (1 - w1) * (1 - w2);
  float w10 = (w1) * (1 - w2);
  float w01 = (1 - w1) * (w2);
  float w11 = (w1) * (w2);

  auto inject = [&](int i1, int i2, float w)
  {
    if (i1 < 0 || i1 >= n1 || i2 < 0 || i2 >= n2)
      return;
    atomic_sum(reinterpret_cast<float *>(&plane[size_t(i1) * n2 + i2]), val * w);
  };

  inject(i1_0, i2_0, w00);
  inject(i1_1, i2_0, w10);
  inject(i1_0, i2_1, w01);
  inject(i1_1, i2_1, w11);
}

template <typename T = float>
WORKER_QUALIFIER inline void bilinear_interp_adj_fixed1(
    T *img,
    int n0, int n1, int n2,
    float i_f0,
    int i1,
    float i_f2,
    T val)
{
  int i0_0 = int(floorf(i_f0));
  int i2_0 = int(floorf(i_f2));
  int i0_1 = i0_0 + 1;
  int i2_1 = i2_0 + 1;

  float w0 = i_f0 - i0_0;
  float w2 = i_f2 - i2_0;
  float w00 = (1 - w0) * (1 - w2);
  float w10 = (w0) * (1 - w2);
  float w01 = (1 - w0) * (w2);
  float w11 = (w0) * (w2);

  auto inject = [&](int i0, int i2, float w)
  {
    if (i0 < 0 || i0 >= n0 || i2 < 0 || i2 >= n2)
      return;
    size_t idx = size_t(i0) * n1 * n2 + size_t(i1) * n2 + i2;
    atomic_sum(reinterpret_cast<float *>(&img[idx]), val * w);
  };

  inject(i0_0, i2_0, w00);
  inject(i0_1, i2_0, w10);
  inject(i0_0, i2_1, w01);
  inject(i0_1, i2_1, w11);
}

template <typename T = float>
WORKER_QUALIFIER inline void bilinear_interp_adj_fixed2(
    T *img,
    int n0, int n1, int n2,
    float i_f0,
    float i_f1,
    int i2,
    T val)
{
  int i0_0 = int(floorf(i_f0));
  int i1_0 = int(floorf(i_f1));
  int i0_1 = i0_0 + 1;
  int i1_1 = i1_0 + 1;

  float w0 = i_f0 - i0_0;
  float w1 = i_f1 - i1_0;
  float w00 = (1 - w0) * (1 - w1);
  float w10 = (w0) * (1 - w1);
  float w01 = (1 - w0) * (w1);
  float w11 = (w0) * (w1);

  auto inject = [&](int i0, int i1, float w)
  {
    if (i0 < 0 || i0 >= n0 || i1 < 0 || i1 >= n1)
      return;
    size_t idx = size_t(i0) * n1 * n2 + size_t(i1) * n2 + i2;
    atomic_sum(reinterpret_cast<float *>(&img[idx]), val * w);
  };

  inject(i0_0, i1_0, w00);
  inject(i0_1, i1_0, w10);
  inject(i0_0, i1_1, w01);
  inject(i0_1, i1_1, w11);
}
