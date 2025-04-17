#pragma once
#include <cstddef>
// import parallelproj_export.h to get the PARALLELPROJ_API macro
// needed for __declspec(dllexport) and __declspec(dllimport)
// This is needed for Windows to export the functions in the DLL
#include "parallelproj_export.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Forward projection using the Joseph 3D algorithm.
     *
     * @note All pointers can be host pointers, CUDA device pointers, or CUDA managed pointers.
     *
     * @param xstart Pointer to array of shape [3*nlors] with the coordinates of the start points of the LORs.
     *               The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2.
     *               Units are the ones of voxsize.
     * @param xend   Pointer to array of shape [3*nlors] with the coordinates of the end points of the LORs.
     *               The end coordinates of the n-th LOR are at xend[n*3 + i] with i = 0,1,2.
     *               Units are the ones of voxsize.
     * @param img    Pointer to array of shape [n0*n1*n2] containing the 3D image used for forward projection.
     *               The pixel [i,j,k] is stored at [n1*n2*i + n2*j + k].
     * @param img_origin Pointer to array [x0_0, x0_1, x0_2] of coordinates of the center of the [0,0,0] voxel.
     * @param voxsize Pointer to array [vs0, vs1, vs2] of the voxel sizes.
     * @param p      Pointer to array of length nlors where the forward projection results will be stored.
     * @param nvoxels Number of voxels in the image.
     * @param nlors  Number of geometrical LORs.
     * @param img_dim Pointer to array with dimensions of the image [n0, n1, n2].
     * @param device_id ID of the device to use for computation (default: 0).
     * @param threadsperblock Number of threads per block for GPU computation (default: 64).
     */
    PARALLELPROJ_API void joseph3d_fwd(const float *xstart,
                                       const float *xend,
                                       const float *img,
                                       const float *img_origin,
                                       const float *voxsize,
                                       float *p,
                                       size_t nvoxels,
                                       size_t nlors,
                                       const int *img_dim,
                                       int device_id = 0,
                                       int threadsperblock = 64);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Backprojection using the Joseph 3D algorithm.
     *
     * @note All pointers can be host pointers, CUDA device pointers, or CUDA managed pointers.
     *
     * @param xstart Pointer to array of shape [3*nlors] with the coordinates of the start points of the LORs.
     *               The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2.
     *               Units are the ones of voxsize.
     * @param xend   Pointer to array of shape [3*nlors] with the coordinates of the end points of the LORs.
     *               The end coordinates of the n-th LOR are at xend[n*3 + i] with i = 0,1,2.
     *               Units are the ones of voxsize.
     * @param img    Pointer to array of shape [n0*n1*n2] containing the 3D image used for backprojection (output).
     *               The pixel [i,j,k] is stored at [n1*n2*i + n2*j + k].
     *               !! Values are added to the existing array !!
     * @param img_origin Pointer to array [x0_0, x0_1, x0_2] of coordinates of the center of the [0,0,0] voxel.
     * @param voxsize Pointer to array [vs0, vs1, vs2] of the voxel sizes.
     * @param p      Pointer to array of length nlors with the values to be backprojected.
     * @param nvoxels Number of voxels in the image.
     * @param nlors  Number of geometrical LORs.
     * @param img_dim Pointer to array with dimensions of the image [n0, n1, n2].
     * @param device_id ID of the device to use for computation (default: 0).
     * @param threadsperblock Number of threads per block for GPU computation (default: 64).
     */
    PARALLELPROJ_API void joseph3d_back(const float *xstart,
                                        const float *xend,
                                        float *img,
                                        const float *img_origin,
                                        const float *voxsize,
                                        const float *p,
                                        size_t nvoxels,
                                        size_t nlors,
                                        const int *img_dim,
                                        int device_id = 0,
                                        int threadsperblock = 64);

#ifdef __cplusplus
}
#endif
