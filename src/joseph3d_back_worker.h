#pragma once
#include "cuda_compat.h"
#include "utils.h"

WORKER_QUALIFIER inline void joseph3d_back_worker(size_t i,
                                                  const float *xstart,
                                                  const float *xend,
                                                  float *img,
                                                  const float *img_origin,
                                                  const float *voxsize,
                                                  const float *p,
                                                  const int *img_dim)
{
  if (p[i] == 0)
  {
    return;
  }

  int n0 = img_dim[0];
  int n1 = img_dim[1];
  int n2 = img_dim[2];

  float voxsize0 = voxsize[0];
  float voxsize1 = voxsize[1];
  float voxsize2 = voxsize[2];

  float img_origin0 = img_origin[0];
  float img_origin1 = img_origin[1];
  float img_origin2 = img_origin[2];

  int direction;
  int i0, i1, i2;
  float i0_f, i1_f, i2_f;
  float x_pr0, x_pr1, x_pr2;
  float cf;

  float xstart0 = xstart[i * 3 + 0];
  float xstart1 = xstart[i * 3 + 1];
  float xstart2 = xstart[i * 3 + 2];

  int istart = -1;
  int iend = -1;

  float d0 = xend[i * 3 + 0] - xstart0;
  float d1 = xend[i * 3 + 1] - xstart1;
  float d2 = xend[i * 3 + 2] - xstart2;

  // test whether the ray intersects the image cube
  // if it does not, istart and iend are set to -1
  // if it does, direction is set to the principal axis
  // and istart and iend are set to the first and last voxel planes
  // that are intersected
  // cf is the correction factor voxsize[dir]/cos[dir]
  ray_cube_intersection_joseph(xstart + 3 * i, xend + 3 * i, img_origin, voxsize, img_dim, direction, cf, istart, iend);

  float val = cf * p[i];

  if (direction == 0 && istart != -1)
  {
    for (i0 = istart; i0 <= iend; ++i0)
    {
      // get the indices where the ray intersects the image plane
      x_pr1 = xstart1 + (img_origin0 + i0 * voxsize0 - xstart0) * d1 / d0;
      x_pr2 = xstart2 + (img_origin0 + i0 * voxsize0 - xstart0) * d2 / d0;

      i1_f = (x_pr1 - img_origin1) / voxsize1;
      i2_f = (x_pr2 - img_origin2) / voxsize2;
      bilinear_interp_adj_fixed0(img, n0, n1, n2, i0, i1_f, i2_f, val);
    }
  }
  else if (direction == 1 && istart != -1)
  {
    for (i1 = istart; i1 <= iend; ++i1)
    {
      // get the indices where the ray intersects the image plane
      x_pr0 = xstart0 + (img_origin1 + i1 * voxsize1 - xstart1) * d0 / d1;
      x_pr2 = xstart2 + (img_origin1 + i1 * voxsize1 - xstart1) * d2 / d1;

      i0_f = (x_pr0 - img_origin0) / voxsize0;
      i2_f = (x_pr2 - img_origin2) / voxsize2;
      bilinear_interp_adj_fixed1(img, n0, n1, n2, i0_f, i1, i2_f, val);
    }
  }
  else if (direction == 2 && istart != -1)
  {
    for (i2 = istart; i2 <= iend; ++i2)
    {
      // get the indices where the ray intersects the image plane
      x_pr0 = xstart0 + (img_origin2 + i2 * voxsize2 - xstart2) * d0 / d2;
      x_pr1 = xstart1 + (img_origin2 + i2 * voxsize2 - xstart2) * d1 / d2;

      i0_f = (x_pr0 - img_origin0) / voxsize0;
      i1_f = (x_pr1 - img_origin1) / voxsize1;
      bilinear_interp_adj_fixed2(img, n0, n1, n2, i0_f, i1_f, i2, val);
    }
  }
}
