#pragma once
#include <cmath>
#include "cuda_compat.h"
#include "utils.h" // for atomic_sum

#ifndef __CUDACC__
#include <math.h>
#endif

WORKER_QUALIFIER inline void atomic_sum(float *target, float value)
{
#ifdef __CUDACC__
  atomicAdd(target, value);
#else
#pragma omp atomic
  *target += value;
#endif
}

/**
 * @brief Ray/cube intersection + Joseph-specific parameters:
 *   - principal axis (0,1,2)
 *   - correction factor
 *   - start/end plane indices
 *
 * Implements the IEEE-754 slab method using fminf/fmaxf to handle +/-0 & inf.
 * Compiles under CUDA (__device__) and standard C++ compilers.
 *
 * If no intersection occurs, `direction` is set to -1 and all outputs
 * (correction, start_plane, end_plane) are reset to defaults.
 * Otherwise fills outputs appropriately.
 *
 * @param xstart       Ray origin (length-3)
 * @param xend         Ray end point (length-3)
 * @param img_origin   World coords of center of voxel (0,0,0)
 * @param voxsize      Voxel sizes (length-3)
 * @param img_dim      Image dimensions [n0,n1,n2]
 * @param[out] direction   Principal axis (0,1,2)
 * @param[out] correction  Correction factor = voxsize[dir]/cos[dir]
 * @param[out] start_plane First plane index to traverse, or -1 if no intersection
 * @param[out] end_plane   Last plane index to traverse, or -1 if no intersection
 */
WORKER_QUALIFIER inline void ray_cube_intersection_joseph(
    const float *xstart,
    const float *xend,
    const float *img_origin,
    const float *voxsize,
    const int *img_dim,
    int &direction,
    float &correction,
    int &start_plane,
    int &end_plane)
{

  // default values - assume no intersection
  direction = 0;
  correction = 1.0f;
  start_plane = -1;
  end_plane = -1;

  // build box bounds: outer faces
  float box_min[3], box_max[3];
  for (int k = 0; k < 3; ++k)
  {
    box_min[k] = img_origin[k] - 0.5f * voxsize[k];
    box_max[k] = box_min[k] + img_dim[k] * voxsize[k];
  }

  // ray vector and its inverse
  float dr[3], inv_dr[3];
  for (int k = 0; k < 3; ++k)
  {
    dr[k] = xend[k] - xstart[k];
    inv_dr[k] = 1.0f / dr[k];
  }

  // slab intersection, parameter t in [0,1]
  float tmin = 0.0f;
  float tmax = 1.0f;
  for (int k = 0; k < 3; ++k)
  {
    float t1 = (box_min[k] - xstart[k]) * inv_dr[k];
    float t2 = (box_max[k] - xstart[k]) * inv_dr[k];
    // use fminf/fmaxf for IEEE-754 compliance
    tmin = fmaxf(tmin, fminf(t1, t2));
    tmax = fminf(tmax, fmaxf(t1, t2));
  }

  if (tmax < tmin)
  {
    // no intersection - direction = 0, start_plane = -1, end_plane = -1
    // correction = 1.0f, all set by default
    return;
  }

  // principal axis & correction
  float dr_sq[3] = {dr[0] * dr[0], dr[1] * dr[1], dr[2] * dr[2]};
  float sum_sq = dr_sq[0] + dr_sq[1] + dr_sq[2];
  // compute squared cosines
  float cos_sq[3] = {dr_sq[0] / sum_sq, dr_sq[1] / sum_sq, dr_sq[2] / sum_sq};

  // choose principal axis based on max squared cosine
  direction = 0;
  if (cos_sq[1] >= cos_sq[0] && cos_sq[1] >= cos_sq[2])
    direction = 1;
  else if (cos_sq[2] >= cos_sq[0] && cos_sq[2] >= cos_sq[1])
    direction = 2;

  // compute correction factor
  correction = voxsize[direction] / sqrtf(cos_sq[direction]);

  // get the first and last voxel planes that are intersected
  // computer the ray cube intersection points
  // note that if xstart or xend are inside the cube, these are xstart / xend
  float xi1 = xstart[direction] + tmin * dr[direction];
  float xi2 = xstart[direction] + tmax * dr[direction];

  // floating point plane values
  float f1 = (xi1 - img_origin[direction]) / voxsize[direction];
  float f2 = (xi2 - img_origin[direction]) / voxsize[direction];

  // if the integer part of f1 and f2 are the same, we are inside one voxel plane
  if ((int)f1 != (int)f2)
  {
    if (f1 > f2)
    {
      float tmp = f1;
      f1 = f2;
      f2 = tmp;
    }
    // first full plane floor(f1)+1
    // ceil(f1) is not completely correct in case f1 is an integer
    start_plane = (int)floorf(f1) + 1;
    // last full plane is floor(exit)
    end_plane = (int)floorf(f2);
  }
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
