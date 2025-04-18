#pragma once
#include "cuda_compat.h"
#include "utils.h"

// we need to import math.h for the floor, ceil, sqrtf if not compiling for CUDA
#ifndef __CUDACC__
#include <math.h>
#endif

WORKER_QUALIFIER inline void joseph3d_back_worker(size_t i,
                                                  const float *xstart,
                                                  const float *xend,
                                                  float *img,
                                                  const float *img_origin,
                                                  const float *voxsize,
                                                  const float *p,
                                                  const int *img_dim)
{

  int n0 = img_dim[0];
  int n1 = img_dim[1];
  int n2 = img_dim[2];

  float voxsize0 = voxsize[0];
  float voxsize1 = voxsize[1];
  float voxsize2 = voxsize[2];

  float img_origin0 = img_origin[0];
  float img_origin1 = img_origin[1];
  float img_origin2 = img_origin[2];

  if (p[i] != 0)
  {

    float d0, d1, d2, d0_sq, d1_sq, d2_sq;
    float cs0, cs1, cs2, cf;
    float lsq, cos0_sq, cos1_sq, cos2_sq;
    unsigned short direction;
    int i0, i1, i2;
    int i0_f, i1_f, i2_f;
    float x_pr0, x_pr1, x_pr2;

    float xstart0 = xstart[i * 3 + 0];
    float xstart1 = xstart[i * 3 + 1];
    float xstart2 = xstart[i * 3 + 2];

    float xend0 = xend[i * 3 + 0];
    float xend1 = xend[i * 3 + 1];
    float xend2 = xend[i * 3 + 2];

    unsigned char intersec;
    float t1, t2;
    float istart_f, iend_f, tmp;
    int istart, iend;

    // test whether the ray between the two detectors is most parallel
    // with the 0, 1, or 2 axis
    d0 = xend0 - xstart0;
    d1 = xend1 - xstart1;
    d2 = xend2 - xstart2;

    //-----------
    //--- test whether ray and cube intersect
    intersec = ray_cube_intersection(xstart0, xstart1, xstart2,
                                     img_origin0 - 1 * voxsize0, img_origin1 - 1 * voxsize1, img_origin2 - 1 * voxsize2,
                                     img_origin0 + n0 * voxsize0, img_origin1 + n1 * voxsize1, img_origin2 + n2 * voxsize2,
                                     d0, d1, d2, &t1, &t2);

    if (intersec == 1)
    {
      d0_sq = d0 * d0;
      d1_sq = d1 * d1;
      d2_sq = d2 * d2;

      lsq = d0_sq + d1_sq + d2_sq;

      cos0_sq = d0_sq / lsq;
      cos1_sq = d1_sq / lsq;
      cos2_sq = d2_sq / lsq;

      cs0 = sqrtf(cos0_sq);
      cs1 = sqrtf(cos1_sq);
      cs2 = sqrtf(cos2_sq);

      direction = 0;
      if ((cos1_sq >= cos0_sq) && (cos1_sq >= cos2_sq))
      {
        direction = 1;
      }
      if ((cos2_sq >= cos0_sq) && (cos2_sq >= cos1_sq))
      {
        direction = 2;
      }

      if (direction == 0)
      {
        // case where ray is most parallel to the 0 axis
        // we step through the volume along the 0 direction

        // factor for correctiong voxel size and |cos(theta)|
        cf = voxsize0 / cs0;

        //--- check where ray enters / leaves cube
        istart_f = (xstart0 + t1 * d0 - img_origin0) / voxsize0;
        iend_f = (xstart0 + t2 * d0 - img_origin0) / voxsize0;

        if (istart_f > iend_f)
        {
          tmp = iend_f;
          iend_f = istart_f;
          istart_f = tmp;
        }

        istart = (int)floor(istart_f);
        iend = (int)ceil(iend_f);
        if (istart < 0)
        {
          istart = 0;
        }
        if (iend >= n0)
        {
          iend = n0;
        }

        // check in which "plane" the start and end points are
        // we have to do this to avoid that we include voxels
        // that are "outside" the line segment bewteen xstart and xend

        // !! for these calculations we overwrite the istart_f and iend_f variables !!
        istart_f = (xstart0 - img_origin0) / voxsize0;
        iend_f = (xend0 - img_origin0) / voxsize0;

        if (istart_f > iend_f)
        {
          tmp = iend_f;
          iend_f = istart_f;
          istart_f = tmp;
        }

        if (istart < (int)floor(istart_f))
        {
          istart = (int)floor(istart_f);
        }
        if (iend >= (int)ceil(iend_f))
        {
          iend = (int)ceil(iend_f);
        }
        //---

        for (i0 = istart; i0 < iend; i0++)
        {
          // get the indices where the ray intersects the image plane
          x_pr1 = xstart1 + (img_origin0 + i0 * voxsize0 - xstart0) * d1 / d0;
          x_pr2 = xstart2 + (img_origin0 + i0 * voxsize0 - xstart0) * d2 / d0;

          i1_f = (x_pr1 - img_origin1) / voxsize1;
          i2_f = (x_pr2 - img_origin2) / voxsize2;
          bilinear_interp_adj_fixed0(img, n0, n1, n2, i0, i1_f, i2_f, p[i] * cf);
        }
      }
      // ---------------------------------------------------------------------------------
      if (direction == 1)
      {
        // case where ray is most parallel to the 1 axis
        // we step through the volume along the 1 direction

        // factor for correctiong voxel size and |cos(theta)|
        cf = voxsize1 / cs1;

        //--- check where ray enters / leaves cube
        istart_f = (xstart1 + t1 * d1 - img_origin1) / voxsize1;
        iend_f = (xstart1 + t2 * d1 - img_origin1) / voxsize1;

        if (istart_f > iend_f)
        {
          tmp = iend_f;
          iend_f = istart_f;
          istart_f = tmp;
        }

        istart = (int)floor(istart_f);
        iend = (int)ceil(iend_f);
        if (istart < 0)
        {
          istart = 0;
        }
        if (iend >= n1)
        {
          iend = n1;
        }

        // check in which "plane" the start and end points are
        // we have to do this to avoid that we include voxels
        // that are "outside" the line segment bewteen xstart and xend

        // !! for these calculations we overwrite the istart_f and iend_f variables !!
        istart_f = (xstart1 - img_origin1) / voxsize1;
        iend_f = (xend1 - img_origin1) / voxsize1;

        if (istart_f > iend_f)
        {
          tmp = iend_f;
          iend_f = istart_f;
          istart_f = tmp;
        }

        if (istart < (int)floor(istart_f))
        {
          istart = (int)floor(istart_f);
        }
        if (iend >= (int)ceil(iend_f))
        {
          iend = (int)ceil(iend_f);
        }
        //---

        for (i1 = istart; i1 < iend; i1++)
        {
          // get the indices where the ray intersects the image plane
          x_pr0 = xstart0 + (img_origin1 + i1 * voxsize1 - xstart1) * d0 / d1;
          x_pr2 = xstart2 + (img_origin1 + i1 * voxsize1 - xstart1) * d2 / d1;

          i0_f = (x_pr0 - img_origin0) / voxsize0;
          i2_f = (x_pr2 - img_origin2) / voxsize2;
          bilinear_interp_adj_fixed1(img, n0, n1, n2, i0_f, i1, i2_f, p[i] * cf);
        }
      }
      //---------------------------------------------------------------------------------
      if (direction == 2)
      {
        // case where ray is most parallel to the 2 axis
        // we step through the volume along the 2 direction

        // factor for correctiong voxel size and |cos(theta)|
        cf = voxsize2 / cs2;

        //--- check where ray enters / leaves cube
        istart_f = (xstart2 + t1 * d2 - img_origin2) / voxsize2;
        iend_f = (xstart2 + t2 * d2 - img_origin2) / voxsize2;

        if (istart_f > iend_f)
        {
          tmp = iend_f;
          iend_f = istart_f;
          istart_f = tmp;
        }

        istart = (int)floor(istart_f);
        iend = (int)ceil(iend_f);
        if (istart < 0)
        {
          istart = 0;
        }
        if (iend >= n2)
        {
          iend = n2;
        }

        // check in which "plane" the start and end points are
        // we have to do this to avoid that we include voxels
        // that are "outside" the line segment bewteen xstart and xend

        // !! for these calculations we overwrite the istart_f and iend_f variables !!
        istart_f = (xstart2 - img_origin2) / voxsize2;
        iend_f = (xend2 - img_origin2) / voxsize2;

        if (istart_f > iend_f)
        {
          tmp = iend_f;
          iend_f = istart_f;
          istart_f = tmp;
        }

        if (istart < (int)floor(istart_f))
        {
          istart = (int)floor(istart_f);
        }
        if (iend >= (int)ceil(iend_f))
        {
          iend = (int)ceil(iend_f);
        }
        //---

        for (i2 = istart; i2 < iend; i2++)
        {
          // get the indices where the ray intersects the image plane
          x_pr0 = xstart0 + (img_origin2 + i2 * voxsize2 - xstart2) * d0 / d2;
          x_pr1 = xstart1 + (img_origin2 + i2 * voxsize2 - xstart2) * d1 / d2;

          i0_f = (x_pr0 - img_origin0) / voxsize0;
          i1_f = (x_pr1 - img_origin1) / voxsize1;
          bilinear_interp_adj_fixed2(img, n0, n1, n2, i0_f, i1_f, i2, p[i] * cf);
        }
      }
    }
  }
}
