/**
 * @file parallelproj.h
 * @brief Header file for the parallelproj backend.
 *
 * @mainpage parallelproj backend documentation
 *
 * - @ref joseph3d_fwd " joseph3d_fwd(): non-TOF Joseph forward projector"
 * - @ref joseph3d_back "joseph3d_back(): non-TOF Joseph back projector"
 */

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

  /** @brief 3D sinogram TOF Joseph forward projector
   *
   * @note All pointers can be host pointers, CUDA device pointers, or CUDA managed pointers.
   *
   *  @param xstart pointer to array of shape [3*nlors] with the coordinates of the start points of the LORs.
   *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2.
   *                Units are the ones of voxsize.
   *  @param xend   pointer array of shape [3*nlors] with the coordinates of the end   points of the LORs.
   *                The end coordinates of the n-th LOR are at xend[n*3 + i] with i = 0,1,2.
   *                Units are the ones of voxsize.
   *  @param img    pointer array of shape [n0*n1*n2] containing the 3D image to be projected.
   *                The voxel [i,j,k] is stored at index n1*n2*i + n2*j + k.
   *  @param img_origin  pointer array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel.
   *  @param voxsize     pointer array [vs0, vs1, vs2] of the voxel sizes.
   *  @param p           pointer to array of length nlors*n_tofbins (output) used to store the projections.
   *                     The ordering is row-major per LOR:
   *                     [LOR0-TOFBIN-0, LOR0-TOFBIN-1, ... LOR0-TOFBIN-(n-1),
   *                      LOR1-TOFBIN-0, LOR1-TOFBIN-1, ... LOR1-TOFBIN-(n-1),
   *                      ...
   *                      LOR(N-1)-TOFBIN-0, LOR(N-1)-TOFBIN-1, ... LOR(N-1)-TOFBIN-(n-1)]
   *  @param nlors       number of geometrical LORs
   *  @param img_dim     array with dimensions of image [n0,n1,n2]
   *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
   *  @param sigma_tof        pointer to array of length 1 or nlors (depending on lor_dependent_sigma_tof)
   *                          with the TOF resolution (sigma) for each LOR in spatial units
   *                          (units of xstart and xend)
   *  @param tofcenter_offset pointer to array of length 1 or nlors (depending on lor_dependent_tofcenter_offset)
   *                          with the offset of the central TOF bin from the midpoint of each LOR in spatial units.
   *                          A positive value means a shift towards the end point of the LOR.
   *  @param n_sigmas         number of sigmas to consider for calculation of TOF kernel
   *  @param n_tofbins        number of TOF bins
   *  @param lor_dependent_sigma_tof unsigned char 0 or 1
   *                                 0 means that the first value in the sigma_tof array is used for all LORs
   *                                 1 (non-zero) means that the TOF resolutions are LOR dependent
   *  @param lor_dependent_tofcenter_offset unsigned char 0 or 1
   *                                        0 means that the first value in the tofcenter_offset array is used for all LORs
   *                                        1 (non-zero) means that the TOF center offsets are LOR dependent
   *  @param device_id ID of the device to use for computation (default: 0).
   *  @param threadsperblock Number of threads per block for GPU computation (default: 64).
   */

  PARALLELPROJ_API void joseph3d_tof_sino_fwd(const float *xstart,
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
   * @brief TOF sinogram backprojection using the Joseph 3D algorithm.
   *
   * @note All pointer arguments may be host pointers, CUDA device pointers, or CUDA managed pointers.
   *       The implementation will safely handle host/device/managed memory for small control arrays
   *       (e.g. img_dim) and copy or access device memory as required.
   *
   * @details The function backprojects a TOF sinogram into a 3D image volume using the Joseph
   *          ray-driven algorithm. The projection data p is organized row-major per LOR:
   *          [LOR0-TOFBIN-0, LOR0-TOFBIN-1, ..., LOR0-TOFBIN-(n_tofbins-1),
   *           LOR1-TOFBIN-0, LOR1-TOFBIN-1, ..., LOR1-TOFBIN-(n_tofbins-1), ...].
   *          Values from p are distributed into img (accumulated, not overwritten).
   *
   * @param xstart Pointer to array of shape [3*nlors] with start coordinates for each LOR
   *               (xstart[n*3 + i], i=0..2). Units are those of @p voxsize.
   * @param xend   Pointer to array of shape [3*nlors] with end coordinates for each LOR
   *               (xend[n*3 + i], i=0..2). Units are those of @p voxsize.
   * @param img    Pointer to array of shape [n0*n1*n2] containing the 3D image to add backprojected
   *               contributions into. The element (i,j,k) is stored at index n1*n2*i + n2*j + k.
   *               Values are added to the existing contents of this array.
   * @param img_origin Pointer to array [x0_0, x0_1, x0_2] giving the coordinates of the center of the
   *                   voxel at index [0,0,0].
   * @param voxsize Pointer to array [vs0, vs1, vs2] specifying voxel sizes in the same units as LOR coords.
   * @param p      Pointer to TOF sinogram data of length nlors * n_tofbins (see details).
   * @param nlors  Number of geometric LORs.
   * @param img_dim Pointer to array [n0, n1, n2] with image dimensions. Can be host/device/managed.
   * @param tofbin_width Width of each TOF bin in spatial units (same units as LOR coordinates).
   * @param sigma_tof Pointer to array of length 1 or nlors (depending on
   *                  lor_dependent_sigma_tof) specifying TOF sigma(s) in spatial units.
   * @param tofcenter_offset Pointer to array of length 1 or nlors (depending on
   *                         lor_dependent_tofcenter_offset) specifying per-LOR offset of the
   *                         central TOF bin from the geometric midpoint (positive towards xend).
   * @param n_sigmas Number of sigmas to consider when evaluating the TOF kernel (controls kernel radius).
   * @param n_tofbins Number of TOF bins per LOR.
   * @param lor_dependent_sigma_tof If non-zero, @p sigma_tof contains one sigma per LOR; otherwise the first
   *                                element is used for all LORs.
   * @param lor_dependent_tofcenter_offset If non-zero, @p tofcenter_offset contains one offset per LOR;
   *                                       otherwise the first element is used for all LORs.
   * @param device_id CUDA device to use (default: 0). If negative, CPU path is used when available.
   * @param threadsperblock Number of CUDA threads per block for GPU execution (default: 64).
   *
   * @return void
   */
  PARALLELPROJ_API void joseph3d_tof_sino_back(const float *xstart,
                                               const float *xend,
                                               float *img,
                                               const float *img_origin,
                                               const float *voxsize,
                                               const float *p,
                                               size_t nlors,
                                               const int *img_dim,
                                               float tofbin_width,
                                               const float *sigma_tof,
                                               const float *tofcenter_offset,
                                               float n_sigmas,
                                               short n_tofbins,
                                               unsigned char lor_dependent_sigma_tof,
                                               unsigned char lor_dependent_tofcenter_offset,
                                               int device_id = 0,
                                               int threadsperblock = 64);
#ifdef __cplusplus
}
#endif
