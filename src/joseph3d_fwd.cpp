#include "parallelproj.h"
#include "joseph3d_fwd_worker.h"
#include "debug.h"

void joseph3d_fwd(const float *xstart,
                  const float *xend,
                  const float *img,
                  const float *img_origin,
                  const float *voxsize,
                  float *p,
                  size_t nvoxels,
                  size_t nlors,
                  const int *img_dim,
                  int device_id,
                  int threadsperblock)
{

#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(nlors); ++i)
    {
        joseph3d_fwd_worker(i, xstart, xend, img, img_origin, voxsize, p, img_dim);
    }
}
