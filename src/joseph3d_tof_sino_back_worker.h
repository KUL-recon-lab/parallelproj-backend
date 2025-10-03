#pragma once
#include "cuda_compat.h"
#include "utils.h"

// Helper: compute TOF weights into caller buffer and scatter normalized contribution.
// No bounds check for MAX_NUM_TOF_WEIGHTS (caller must provide a large enough buffer).
WORKER_QUALIFIER static inline float _gather_back_tof_weights(
    float it_f,
    float max_tof_bin_diff,
    float tofbin_width,
    float sig_tof,
    float *tof_weights, // buffer to hold TOF weights of size at least MAX_NUM_TOF_WEIGHTS
    const float *p,     // TOF projection values
    size_t lor_offset,  // offset at which TOF values for given LOR start in p
    short n_tofbins)
{
  int it_min = static_cast<int>(floorf(it_f - max_tof_bin_diff));
  int it_max = static_cast<int>(ceilf(it_f + max_tof_bin_diff));
  int n_tof_weights = it_max + 1 - it_min;

  float sum_weights = 0.0f;
  for (int k = 0; k < n_tof_weights; ++k)
  {
    float dist = fabsf(it_f - it_min - k) * tofbin_width;
    tof_weights[k] = effective_gaussian_tof_kernel(dist, sig_tof, tofbin_width);
    sum_weights += tof_weights[k];
  }

  // normalize and only take valid TOF bins
  int k_start = (it_min < 0) ? -it_min : 0;
  int k_end = ((it_min + n_tof_weights) > n_tofbins) ? (n_tofbins - it_min) : n_tof_weights;
  for (int k = k_start; k < k_end; ++k)
  {
    toAdd = p[lor_offset + it_min + k] * tof_weights[k];
  }

  toAdd /= sum_weights;

  return toAdd;
}
WORKER_QUALIFIER inline void joseph3d_tof_sino_back_worker(size_t i,
                                                           const float *xstart,
                                                           const float *xend,
                                                           float *img,
                                                           const float *img_origin,
                                                           const float *voxsize,
                                                           const float *p,
                                                           const int *img_dim,
                                                           float tofbin_width,
                                                           const float *sigma_tof,
                                                           const float *tofcenter_offset,
                                                           float n_sigmas,
                                                           short n_tofbins,
                                                           unsigned char lor_dependent_sigma_tof,
                                                           unsigned char lor_dependent_tofcenter_offset)
{
  float tof_lor_sum = 0;
  // check whether the sum over TOF of the TOF sinogram to be backprojcted is > 0
  // if it is 0, we can skip the backprojection of this LOR
  size_t lor_offset = (size_t)i * (size_t)n_tofbins;
  float tof_lor_sum = 0.0f;
  for (short j = 0; j < n_tofbins; ++j)
  {
    tof_lor_sum += p[lor_offset + j];
  }

  if (tof_lor_sum == 0)
  {
    return;
  }

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

  float d0 = xend[3 * i + 0] - xstart[3 * i + 0];
  float d1 = xend[3 * i + 1] - xstart[3 * i + 1];
  float d2 = xend[3 * i + 2] - xstart[3 * i + 2];

  float dr;
  float val;

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

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  ////// calculate TOF-related parameters
  float toAdd;                              // non-TOF contribution to the projection value for a given image plane
  float tof_weights[MAX_NUM_TOF_WEIGHTS];   // buffer to hold TOF weights for a given image plane
  float costheta = voxsize[direction] / cf; // cosine of angle between ray and principal axis

  // get the sigma_tof and tofcenter_offset for this LOR depending on whether they are constant or LOR-dependent
  float sig_tof = lor_dependent_sigma_tof ? sigma_tof[i] : sigma_tof[0];
  float tofcen_offset = lor_dependent_tofcenter_offset ? tofcenter_offset[i] : tofcenter_offset[0];

  // maximum number of TOF bins away from the current TOF bin to consider
  // TOF bins outside this range will have a negligible contribution and will be ignored
  float max_tof_bin_diff = n_sigmas * sig_tof / tofbin_width;

  // sign variable that indicated whether TOF bin numbers increase or decrease when
  // through the image along the principal axis direction
  float sign = (xend[3 * i + direction] >= xstart[3 * i + direction]) ? 1.0 : -1.0;

  // the center of the first TOF bin (TOF bin 0) projected onto the principal axis
  float tof_origin = 0.5 * (xstart[3 * i + direction] + xend[3 * i + direction]) - sign * (n_tofbins / 2 - 0.5) * (tofbin_width * costheta) + tofcen_offset * costheta;
  // slope of TOF bin number as a function of distance along the principal axis
  // the position of the TOF bins projects onto the principal axis is: tof_origin + tof_bin_number*tof_slope
  float tof_slope = sign * tofbin_width * costheta;

  // the TOF bin number of intersection point of the ray with a given image plane along the principal axis is it_f = i*at + bt
  float at = sign * cf / tofbin_width;
  float bt = (img_origin[direction] - tof_origin) / tof_slope;
  float it_f = istart * at + bt;

  //////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // float val = cf * p[i];

  if (direction == 0)
  {
    dr = d0;

    a1 = (d1 * voxsize[direction]) / (voxsize[1] * dr);
    b1 = (xstart[3 * i + 1] - img_origin[1] + d1 * (img_origin[direction] - xstart[3 * i + direction]) / dr) / voxsize[1];

    a2 = (d2 * voxsize[direction]) / (voxsize[2] * dr);
    b2 = (xstart[3 * i + 2] - img_origin[2] + d2 * (img_origin[direction] - xstart[3 * i + direction]) / dr) / voxsize[2];

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i1_f = istart * a1 + b1;
    i2_f = istart * a2 + b2;

    for (i0 = istart; i0 <= iend; ++i0)
    {
      val = cf * _gather_back_tof_weights(it_f, max_tof_bin_diff, tofbin_width, sig_tof, tof_weights, p, lor_offset, n_tofbins);
      bilinear_interp_adj_fixed0(img, n0, n1, n2, i0, i1_f, i2_f, val);
      i1_f += a1;
      i2_f += a2;
      it_f += at;
    }
  }
  else if (direction == 1)
  {
    dr = d1;

    a0 = (d0 * voxsize[direction]) / (voxsize[0] * dr);
    b0 = (xstart[3 * i + 0] - img_origin[0] + d0 * (img_origin[direction] - xstart[3 * i + direction]) / dr) / voxsize[0];

    a2 = (d2 * voxsize[direction]) / (voxsize[2] * dr);
    b2 = (xstart[3 * i + 2] - img_origin[2] + d2 * (img_origin[direction] - xstart[3 * i + direction]) / dr) / voxsize[2];

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i0_f = istart * a0 + b0;
    i2_f = istart * a2 + b2;

    for (i1 = istart; i1 <= iend; ++i1)
    {
      val = cf * _gather_back_tof_weights(it_f, max_tof_bin_diff, tofbin_width, sig_tof, tof_weights, p, lor_offset, n_tofbins);
      bilinear_interp_adj_fixed1(img, n0, n1, n2, i0_f, i1, i2_f, val);
      i0_f += a0;
      i2_f += a2;
      it_f += at;
    }
  }
  else if (direction == 2)
  {
    dr = d2;

    a0 = (d0 * voxsize[direction]) / (voxsize[0] * dr);
    b0 = (xstart[3 * i + 0] - img_origin[0] + d0 * (img_origin[direction] - xstart[3 * i + direction]) / dr) / voxsize[0];

    a1 = (d1 * voxsize[direction]) / (voxsize[1] * dr);
    b1 = (xstart[3 * i + 1] - img_origin[1] + d1 * (img_origin[direction] - xstart[3 * i + direction]) / dr) / voxsize[1];

    // get the intersection points of the ray and the start image plane in voxel coordinates
    i0_f = istart * a0 + b0;
    i1_f = istart * a1 + b1;

    for (i2 = istart; i2 <= iend; ++i2)
    {
      val = cf * _gather_back_tof_weights(it_f, max_tof_bin_diff, tofbin_width, sig_tof, tof_weights, p, lor_offset, n_tofbins);
      bilinear_interp_adj_fixed2(img, n0, n1, n2, i0_f, i1_f, i2, val);
      i0_f += a0;
      i1_f += a1;
      it_f += at;
    }
  }
}
