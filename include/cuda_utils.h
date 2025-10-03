#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

// Derive nvoxels = img_dim[0]*img_dim[1]*img_dim[2] when img_dim may be a
// host, CUDA managed or device pointer. Throws std::invalid_argument or
// std::runtime_error on error; returns computed nvoxels on success.
inline size_t cuda_nvoxels_from_img_dim(const int *img_dim_ptr)
{
    if (!img_dim_ptr)
        throw std::invalid_argument("nvoxels_from_img_dim: img_dim_ptr is null");

    int h_img_dim[3] = {0, 0, 0};
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, static_cast<const void *>(img_dim_ptr));

    // If pointer known to CUDA and points to device/managed memory, copy to host.
    if (err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged))
    {
        err = cudaMemcpy(h_img_dim, img_dim_ptr, 3 * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("nvoxels_from_img_dim: cudaMemcpy failed: ") + cudaGetErrorString(err));
    }
    else
    {
        // Treat as host pointer (or pointer attributes not available) — read directly.
        h_img_dim[0] = img_dim_ptr[0];
        h_img_dim[1] = img_dim_ptr[1];
        h_img_dim[2] = img_dim_ptr[2];
    }

    if (h_img_dim[0] <= 0 || h_img_dim[1] <= 0 || h_img_dim[2] <= 0)
        throw std::invalid_argument("nvoxels_from_img_dim: invalid img_dim values");

    return static_cast<size_t>(h_img_dim[0]) *
           static_cast<size_t>(h_img_dim[1]) *
           static_cast<size_t>(h_img_dim[2]);
}

// Overload for constant input_ptr (const T*)
template <typename T>
void handle_cuda_input_array(const T *input_ptr, T **device_ptr, size_t size, bool &free_flag, int device_id, cudaMemoryAdvise memory_hint);

// Overload for non-constant input_ptr (T*)
template <typename T>
void handle_cuda_input_array(T *input_ptr, T **device_ptr, size_t size, bool &free_flag, int device_id, cudaMemoryAdvise memory_hint);
