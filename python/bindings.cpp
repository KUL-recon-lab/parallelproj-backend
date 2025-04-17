#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <dlpack/dlpack.h>
#include "parallelproj.h"

namespace py = pybind11;

// Helper function to extract raw pointer and shape
template <typename T>
std::pair<T *, std::vector<size_t>> extract_pointer_and_shape(py::object array)
{
    T *raw_ptr = nullptr;
    std::vector<size_t> shape;

    // Handle arrays using the __dlpack__ protocol (default)
    if (py::hasattr(array, "__dlpack__"))
    {
        // Call the __dlpack__ method to get a DLPack tensor
        py::capsule dlpack_capsule = array.attr("__dlpack__")();

        // Extract the DLManagedTensor from the capsule
        auto *managed_tensor = static_cast<DLManagedTensor *>(
            PyCapsule_GetPointer(dlpack_capsule.ptr(), "dltensor"));

        if (!managed_tensor)
        {
            throw std::runtime_error("Failed to extract DLManagedTensor from PyCapsule.");
        }

        // Access the DLTensor from the DLManagedTensor
        DLTensor dltensor = managed_tensor->dl_tensor;

        // Ensure the data type matches
        if (dltensor.dtype.code != kDLFloat || dltensor.dtype.bits != sizeof(T) * 8)
        {
            throw std::invalid_argument("DLPack tensor has an incompatible data type.");
        }

        // Get the raw pointer and shape
        raw_ptr = reinterpret_cast<T *>(dltensor.data);
        shape = std::vector<size_t>(dltensor.shape, dltensor.shape + dltensor.ndim);
    }
    // Handle NumPy arrays
    else if (py::isinstance<py::array_t<T>>(array))
    {
        auto numpy_array = array.cast<py::array_t<T>>();
        raw_ptr = numpy_array.mutable_data();
        shape = std::vector<size_t>(numpy_array.shape(), numpy_array.shape() + numpy_array.ndim());
    }
    // Handle arrays using the __cuda_array_interface__ (e.g. cupy or pytorch gpu tensors)
    else if (py::hasattr(array, "__cuda_array_interface__"))
    {
        auto cuda_interface = array.attr("__cuda_array_interface__");
        raw_ptr = reinterpret_cast<T *>(cuda_interface["data"].cast<std::pair<size_t, bool>>().first);
        shape = cuda_interface["shape"].cast<std::vector<size_t>>();
    }
    // Handle arrays using the __array_interface__ (Python Array API or array_api_strict)
    else
    {
        throw std::invalid_argument("Unsupported array type. Must have __dlpack__, __cuda_array_interface__ or be numpy.");
    }

    return {raw_ptr, shape};
}

// Wrapper for joseph3d_fwd
void joseph3d_fwd_py(py::object xstart,
                     py::object xend,
                     py::object img,
                     py::object img_origin,
                     py::object voxsize,
                     py::object p,
                     int device_id = 0,
                     int threadsperblock = 64)
{
    // Extract raw pointers and shapes
    auto [xstart_ptr, xstart_shape] = extract_pointer_and_shape<float>(xstart);
    auto [xend_ptr, xend_shape] = extract_pointer_and_shape<float>(xend);
    auto [img_ptr, img_shape] = extract_pointer_and_shape<float>(img);
    auto [img_origin_ptr, img_origin_shape] = extract_pointer_and_shape<float>(img_origin);
    auto [voxsize_ptr, voxsize_shape] = extract_pointer_and_shape<float>(voxsize);
    auto [p_ptr, p_shape] = extract_pointer_and_shape<float>(p);

    // Validate shapes
    if (xstart_shape.size() < 2 || xstart_shape.back() != 3)
    {
        throw std::invalid_argument("xstart must have at least 2 dims and shape (..., 3)");
    }
    if (xend_shape.size() < 2 || xend_shape.back() != 3)
    {
        throw std::invalid_argument("xend must have at least 2 dims and shape (..., 3)");
    }
    if (img_shape.size() != 3)
    {
        throw std::invalid_argument("img must be a 3D array");
    }
    // Validate that p.shape == xstart.shape[:-1]
    if (p_shape.size() != xstart_shape.size() - 1 ||
        !std::equal(p_shape.begin(), p_shape.end(), xstart_shape.begin()))
    {
        throw std::invalid_argument("p must have a shape equal to xstart.shape[:-1]");
    }
    if (img_origin_shape.size() != 1 || img_origin_shape[0] != 3)
    {
        throw std::invalid_argument("img_origin must be a 1D array with 3 elements");
    }
    if (voxsize_shape.size() != 1 || voxsize_shape[0] != 3)
    {
        throw std::invalid_argument("voxsize must be a 1D array with 3 elements");
    }

    // Calculate nlors using xstart_shape (multiply shape except the last dimension)
    size_t nlors = std::accumulate(xstart_shape.begin(), xstart_shape.end() - 1, static_cast<size_t>(1), std::multiplies<size_t>());
    int img_dim[3] = {static_cast<int>(img_shape[0]), static_cast<int>(img_shape[1]), static_cast<int>(img_shape[2])};
    size_t nvoxels = img_dim[0] * img_dim[1] * img_dim[2];

    // Call the C++ function
    joseph3d_fwd(xstart_ptr, xend_ptr, img_ptr, img_origin_ptr, voxsize_ptr, p_ptr, nvoxels, nlors, img_dim, device_id, threadsperblock);
}

// Wrapper for joseph3d_back
void joseph3d_back_py(py::object xstart,
                      py::object xend,
                      py::object img,
                      py::object img_origin,
                      py::object voxsize,
                      py::object p,
                      int device_id = 0,
                      int threadsperblock = 64)
{
    // Extract raw pointers and shapes
    auto [xstart_ptr, xstart_shape] = extract_pointer_and_shape<float>(xstart);
    auto [xend_ptr, xend_shape] = extract_pointer_and_shape<float>(xend);
    auto [img_ptr, img_shape] = extract_pointer_and_shape<float>(img);
    auto [img_origin_ptr, img_origin_shape] = extract_pointer_and_shape<float>(img_origin);
    auto [voxsize_ptr, voxsize_shape] = extract_pointer_and_shape<float>(voxsize);
    auto [p_ptr, p_shape] = extract_pointer_and_shape<float>(p);

    // Validate shapes
    if (xstart_shape.size() < 2 || xstart_shape.back() != 3)
    {
        throw std::invalid_argument("xstart must have at least 2 dims and shape (..., 3)");
    }
    if (xend_shape.size() < 2 || xend_shape.back() != 3)
    {
        throw std::invalid_argument("xend must have at least 2 dims and shape (..., 3)");
    }
    if (img_shape.size() != 3)
    {
        throw std::invalid_argument("img must be a 3D array");
    }
    // Validate that p.shape == xstart.shape[:-1]
    if (p_shape.size() != xstart_shape.size() - 1 ||
        !std::equal(p_shape.begin(), p_shape.end(), xstart_shape.begin()))
    {
        throw std::invalid_argument("p must have a shape equal to xstart.shape[:-1]");
    }
    if (img_origin_shape.size() != 1 || img_origin_shape[0] != 3)
    {
        throw std::invalid_argument("img_origin must be a 1D array with 3 elements");
    }
    if (voxsize_shape.size() != 1 || voxsize_shape[0] != 3)
    {
        throw std::invalid_argument("voxsize must be a 1D array with 3 elements");
    }

    // Calculate nlors using xstart_shape (multiply shape except the last dimension)
    size_t nlors = std::accumulate(xstart_shape.begin(), xstart_shape.end() - 1, static_cast<size_t>(1), std::multiplies<size_t>());
    int img_dim[3] = {static_cast<int>(img_shape[0]), static_cast<int>(img_shape[1]), static_cast<int>(img_shape[2])};
    size_t nvoxels = img_dim[0] * img_dim[1] * img_dim[2];

    // Call the C++ function
    joseph3d_back(xstart_ptr, xend_ptr, img_ptr, img_origin_ptr, voxsize_ptr, p_ptr, nvoxels, nlors, img_dim, device_id, threadsperblock);
}

// Pybind11 module definition
PYBIND11_MODULE(parallelproj_backend, m)
{
    m.doc() = "Python bindings for parallelproj backend";

    // Expose the project version as __version__
#ifdef PROJECT_VERSION
    m.attr("__version__") = PROJECT_VERSION;
#else
    m.attr("__version__") = "unknown";
#endif

    // Expose the PARALLELPROJ_CUDA definition as a Python constant
#ifdef PARALLELPROJ_CUDA
    m.attr("PARALLELPROJ_CUDA") = PARALLELPROJ_CUDA;
#else
    m.attr("PARALLELPROJ_CUDA") = 0; // Default to 0 if not defined
#endif

    m.def("joseph3d_fwd", &joseph3d_fwd_py, R"pbdoc(
    Non-TOF forward projection using the Joseph 3D algorithm. (adjoint of joseph3d_back())

    Parameters:
    -----------
    xstart : array-like
        array of size [...,3] with the world coordinates of the start points of the LORs.

    xend : array-like
        array of size [...,3] with the world coordinates of the end points of the LORs.

    img : array-like
        3D array of shape [n0,n1,n2] containing the 3D image used for forward projection.

    img_origin : array-like
        array [x0_0, x0_1, x0_2] with the world coordinates of the center of the [0,0,0] voxel.

    voxsize : array-like
        array [vs0, vs1, vs2] of the voxel sizes (same units as world coordinates).

    p : array-like
        array of size [...] where the forward projection results will be stored.

    device_id : int, optional
        ID of the device to use for computation (default: 0).

    threadsperblock : int, optional
        Number of threads per block for GPU computation (default: 64).

    Returns:
    --------
    None
)pbdoc",
          py::arg("xstart"), py::arg("xend"), py::arg("img"), py::arg("img_origin"),
          py::arg("voxsize"), py::arg("p"), py::arg("device_id") = 0, py::arg("threadsperblock") = 64);

    m.def("joseph3d_back", &joseph3d_back_py, R"pbdoc(
    Non-TOF back projection using the Joseph 3D algorithm (adjoint of joseph3_fwd).

    Parameters:
    -----------
    xstart : array-like
        Array of size [...,3] with the world coordinates of the start points of the LORs.

    xend : array-like
        Array of size [...,3] with the world coordinates of the end points of the LORs.

    img : array-like
        3D array of shape [n0,n1,n2] containing the 3D image used for back projection (output).
        The pixel [i,j,k] is stored at [n1*n2*i + n2*j + k].
        !! Values are added to the existing array !!

    img_origin : array-like
        Array [x0_0, x0_1, x0_2] with the world coordinates of the center of the [0,0,0] voxel.

    voxsize : array-like
        Array [vs0, vs1, vs2] of the voxel sizes (same units as world coordinates).

    p : array-like
        Array of size [...] containing the values to be back projected.

    device_id : int, optional
        ID of the device to use for computation (default: 0).

    threadsperblock : int, optional
        Number of threads per block for GPU computation (default: 64).

    Returns:
    --------
    None
)pbdoc",
          py::arg("xstart"), py::arg("xend"), py::arg("img"), py::arg("img_origin"),
          py::arg("voxsize"), py::arg("p"), py::arg("device_id") = 0, py::arg("threadsperblock") = 64);
}
