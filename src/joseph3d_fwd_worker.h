#pragma once
#include "cuda_compat.h"
#include "utils.h"

WORKER_QUALIFIER inline void joseph3d_fwd_worker(size_t i,
                                                 const float *xstart,
                                                 const float *xend,
                                                 const float *img,
                                                 const float *img_origin,
                                                 const float *voxsize,
                                                 float *p,
                                                 const int *img_dim)
{
  int n0 = img_dim[0];
  int n1 = img_dim[1];
  int n2 = img_dim[2];

  int direction;
  int i0, i1, i2;
  float i0_f, i1_f, i2_f;
  float cf;

  float a0, a1, a2;
  float b0, b1, b2;

  int istart = -1;
  int iend = -1;

  // test whether the ray intersects the image cube
  // if it does not, istart and iend are set to -1
  // if it does, direction is set to the principal axis
  // and istart and iend are set to the first and last voxel planes
  // that are intersected
  // cf is the correction factor voxsize[dir]/cos[dir]
  ray_cube_intersection_joseph(xstart + 3 * i, xend + 3 * i, img_origin, voxsize, img_dim, direction, cf, istart, iend);

  // if the ray does not intersect the image cube, return
  // istart and iend are set to -1
  if (istart == -1)
  {
    return;
  }

  p[i] = 0.0f;

  if (direction == 0)
  {
    b1 = ((xstart[3 * i + 1] - img_origin[1]) / voxsize[1]) +
         (img_origin[0] - xstart[3 * i + 0]) * (xend[3 * i + 1] - xstart[3 * i + 1]) /
             (voxsize[1] * (xend[3 * i + 0] - xstart[3 * i + 0]));

    a1 = voxsize[0] * (xend[3 * i + 1] - xstart[3 * i + 1]) /
         (voxsize[1] * (xend[3 * i + 0] - xstart[3 * i + 0]));

    b2 = ((xstart[3 * i + 2] - img_origin[2]) / voxsize[2]) +
         (img_origin[0] - xstart[3 * i + 0]) * (xend[3 * i + 2] - xstart[3 * i + 2]) /
             (voxsize[2] * (xend[3 * i + 0] - xstart[3 * i + 0]));

    a2 = voxsize[0] * (xend[3 * i + 2] - xstart[3 * i + 2]) /
         (voxsize[2] * (xend[3 * i + 0] - xstart[3 * i + 0]));

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i1_f = istart * a1 + b1;
    i2_f = istart * a2 + b2;

    for (i0 = istart; i0 <= iend; ++i0)
    {
      p[i] += bilinear_interp_fixed0(img, n0, n1, n2, i0, i1_f, i2_f);
      i1_f += a1;
      i2_f += a2;
    }
  }
  else if (direction == 1)
  {
    b0 = ((xstart[3 * i + 0] - img_origin[0]) / voxsize[0]) +
         (img_origin[1] - xstart[3 * i + 1]) * (xend[3 * i + 0] - xstart[3 * i + 0]) /
             (voxsize[0] * (xend[3 * i + 1] - xstart[3 * i + 1]));

    a0 = voxsize[1] * (xend[3 * i + 0] - xstart[3 * i + 0]) /
         (voxsize[0] * (xend[3 * i + 1] - xstart[3 * i + 1]));

    b2 = ((xstart[3 * i + 2] - img_origin[2]) / voxsize[2]) +
         (img_origin[1] - xstart[3 * i + 1]) * (xend[3 * i + 2] - xstart[3 * i + 2]) /
             (voxsize[2] * (xend[3 * i + 1] - xstart[3 * i + 1]));

    a2 = voxsize[1] * (xend[3 * i + 2] - xstart[3 * i + 2]) /
         (voxsize[2] * (xend[3 * i + 1] - xstart[3 * i + 1]));

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i0_f = istart * a0 + b0;
    i2_f = istart * a2 + b2;

    for (i1 = istart; i1 <= iend; ++i1)
    {
      p[i] += bilinear_interp_fixed1(img, n0, n1, n2, i0_f, i1, i2_f);
      i0_f += a0;
      i2_f += a2;
    }
  }
  else if (direction == 2)
  {
    b0 = ((xstart[3 * i + 0] - img_origin[0]) / voxsize[0]) +
         (img_origin[2] - xstart[3 * i + 2]) * (xend[3 * i + 0] - xstart[3 * i + 0]) /
             (voxsize[0] * (xend[3 * i + 2] - xstart[3 * i + 2]));

    a0 = voxsize[0] * (xend[3 * i + 2] - xstart[3 * i + 2]) /
         (voxsize[2] * (xend[3 * i + 0] - xstart[3 * i + 0]));

    b1 = ((xstart[3 * i + 1] - img_origin[1]) / voxsize[1]) +
         (img_origin[2] - xstart[3 * i + 2]) * (xend[3 * i + 1] - xstart[3 * i + 1]) /
             (voxsize[1] * (xend[3 * i + 2] - xstart[3 * i + 2]));

    a1 = voxsize[2] * (xend[3 * i + 1] - xstart[3 * i + 1]) /
         (voxsize[1] * (xend[3 * i + 2] - xstart[3 * i + 2]));

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i0_f = istart * a0 + b0;
    i1_f = istart * a1 + b1;

    for (i2 = istart; i2 <= iend; ++i2)
    {
      p[i] += bilinear_interp_fixed2(img, n0, n1, n2, i0_f, i1_f, i2);
      i0_f += a0;
      i1_f += a1;
    }
  }

  p[i] *= cf;
}
