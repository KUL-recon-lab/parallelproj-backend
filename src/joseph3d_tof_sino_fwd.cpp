#include "parallelproj.h"
#include "joseph3d_tof_sino_fwd_worker.h"

void joseph3d_tof_sino_fwd(const float *xstart,
                           const float *xend,
                           const float *img,
                           const float *img_origin,
                           const float *voxsize,
                           float *p,
                           size_t nvoxels,
                           size_t nlors,
                           const int *img_dim,
                           float tofbin_width,
                           const float *sigma_tof,
                           const float *tofcenter_offset,
                           float n_sigmas,
                           short n_tofbins,
                           unsigned char lor_dependent_sigma_tof,
                           unsigned char lor_dependent_tofcenter_offset,
                           int device_id,
                           int threadsperblock)
{

#pragma omp parallel for
  for (long long i = 0; i < static_cast<long long>(nlors); ++i)
  {
    joseph3d_tof_sino_fwd_worker(i, xstart, xend, img, img_origin, voxsize, p, img_dim, tofbin_width,
                                 sigma_tof, tofcenter_offset, n_sigmas, n_tofbins,
                                 lor_dependent_sigma_tof, lor_dependent_tofcenter_offset);
  }
}
