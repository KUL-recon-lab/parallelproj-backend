#include "parallelproj.h"
#include "joseph3d_tof_sino_fwd_worker.h"
#include "debug.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

__global__ void joseph3d_tof_sino_fwd_kernel(const float *xstart,
                                             const float *xend,
                                             const float *img,
                                             const float *img_origin,
                                             const float *voxsize,
                                             float *p,
                                             size_t nlors,
                                             const int *img_dim,
                                             float tofbin_width,
                                             const float *sigma_tof,
                                             const float *tofcenter_offset,
                                             float n_sigmas,
                                             short n_tofbins,
                                             unsigned char lor_dependent_sigma_tof,
                                             unsigned char lor_dependent_tofcenter_offset)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nlors)
    {
        joseph3d_tof_sino_fwd_worker(i, xstart, xend, img, img_origin, voxsize, p, img_dim, tofbin_width,
                                     sigma_tof, tofcenter_offset, n_sigmas, n_tofbins,
                                     lor_dependent_sigma_tof, lor_dependent_tofcenter_offset);
    }
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

void joseph3d_tof_sino_fwd(const float *xstart,
                           const float *xend,
                           const float *img,
                           const float *img_origin,
                           const float *voxsize,
                           float *p,
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
    // Calculate nvoxels from img_dim - img_dim can be device pointer!
    size_t nvoxels = cuda_nvoxels_from_img_dim(img_dim);

    // Set the CUDA device
    if (device_id >= 0)
    {
        cudaSetDevice(device_id);
    }

    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////
    // copy arrays to device if needed
    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////

    // Handle xstart (read mostly)
    float *d_xstart = nullptr;
    bool free_xstart = false;
    handle_cuda_input_array(xstart, &d_xstart, sizeof(float) * nlors * 3, free_xstart, device_id, cudaMemAdviseSetReadMostly);

    // Handle xend (read mostly)
    float *d_xend = nullptr;
    bool free_xend = false;
    handle_cuda_input_array(xend, &d_xend, sizeof(float) * nlors * 3, free_xend, device_id, cudaMemAdviseSetReadMostly);

    // Handle img (read mostly)
    float *d_img = nullptr;
    bool free_img = false;
    handle_cuda_input_array(img, &d_img, sizeof(float) * nvoxels, free_img, device_id, cudaMemAdviseSetReadMostly);

    // Handle img_origin (read mostly)
    float *d_img_origin = nullptr;
    bool free_img_origin = false;
    handle_cuda_input_array(img_origin, &d_img_origin, sizeof(float) * 3, free_img_origin, device_id, cudaMemAdviseSetReadMostly);

    // Handle voxsize (read mostly)
    float *d_voxsize = nullptr;
    bool free_voxsize = false;
    handle_cuda_input_array(voxsize, &d_voxsize, sizeof(float) * 3, free_voxsize, device_id, cudaMemAdviseSetReadMostly);

    // Handle p (write access) - size depends on tofbins
    size_t p_size = sizeof(float) * nlors * n_tofbins;
    float *d_p = nullptr;
    bool free_p = false;
    handle_cuda_input_array(p, &d_p, p_size, free_p, device_id, cudaMemAdviseSetAccessedBy);

    // Handle img_dim (read mostly)
    int *d_img_dim = nullptr;
    bool free_img_dim = false;
    handle_cuda_input_array(img_dim, &d_img_dim, sizeof(int) * 3, free_img_dim, device_id, cudaMemAdviseSetReadMostly);

    // Handle sigma_tof (read mostly)
    float *d_sigma_tof = nullptr;
    bool free_sigma_tof = false;
    size_t sigma_tof_size = lor_dependent_sigma_tof ? sizeof(float) * nlors : sizeof(float);
    handle_cuda_input_array(sigma_tof, &d_sigma_tof, sigma_tof_size, free_sigma_tof, device_id, cudaMemAdviseSetReadMostly);

    // Handle tofcenter_offset (read mostly)
    float *d_tofcenter_offset = nullptr;
    bool free_tofcenter_offset = false;
    size_t tofcenter_offset_size = lor_dependent_tofcenter_offset ? sizeof(float) * nlors : sizeof(float);
    handle_cuda_input_array(tofcenter_offset, &d_tofcenter_offset, tofcenter_offset_size, free_tofcenter_offset, device_id, cudaMemAdviseSetReadMostly);

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    // launch the kernel
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

#ifdef DEBUG
    // get and print the current cuda device ID
    int current_device_id;
    cudaGetDevice(&current_device_id);
    DEBUG_PRINT("Using CUDA device: %d\n", current_device_id);
#endif

    int num_blocks = (int)((nlors + threadsperblock - 1) / threadsperblock);
    joseph3d_tof_sino_fwd_kernel<<<num_blocks, threadsperblock>>>(
        d_xstart, d_xend, d_img, d_img_origin, d_voxsize, d_p, nlors, d_img_dim,
        tofbin_width, d_sigma_tof, d_tofcenter_offset, n_sigmas, n_tofbins,
        lor_dependent_sigma_tof, lor_dependent_tofcenter_offset);
    cudaDeviceSynchronize();

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    // free device memory if needed
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    // Free device memory if it was allocated
    if (free_xstart)
        cudaFree(d_xstart);
    if (free_xend)
        cudaFree(d_xend);
    if (free_img)
        cudaFree(d_img);
    if (free_img_origin)
        cudaFree(d_img_origin);
    if (free_voxsize)
        cudaFree(d_voxsize);
    if (free_p)
    {
        cudaMemcpy(p, d_p, p_size, cudaMemcpyDeviceToHost);
        cudaFree(d_p);
    }
    if (free_img_dim)
        cudaFree(d_img_dim);
    if (free_sigma_tof)
        cudaFree(d_sigma_tof);
    if (free_tofcenter_offset)
        cudaFree(d_tofcenter_offset);
}