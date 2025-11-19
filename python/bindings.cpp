#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "parallelproj.h"

namespace nb = nanobind;
using namespace nb::literals;

// ND c-contiguous float arrays
using ConstFloatNDArray = nb::ndarray<const float, nb::c_contig>;
using FloatNDArray = nb::ndarray<float, nb::c_contig>;
// 3D c-contiguous float arrays
using ConstFloat3DArray = nb::ndarray<const float, nb::c_contig, nb::ndim<3>>;
using Float3DArray = nb::ndarray<float, nb::c_contig, nb::ndim<3>>;
// 1D c-contiguous float arrays
using ConstFloat1D3ELArray = nb::ndarray<const float, nb::c_contig, nb::shape<3>>;

// Wrapper for joseph3d_fwd
void joseph3d_fwd_py(ConstFloatNDArray xstart,
                     ConstFloatNDArray xend,
                     ConstFloat3DArray img,
                     ConstFloat1D3ELArray img_origin,
                     ConstFloat1D3ELArray voxsize,
                     FloatNDArray p,
                     int device_id = 0,
                     int threadsperblock = 64)
{
  // 1 check that ndim of xstart and xend >=2 and last dim ==3
  if (xstart.ndim() < 2 || xstart.shape(xstart.ndim() - 1) != 3)
    throw std::invalid_argument("xstart must have at least 2 dims and shape (..., 3)");

  // 2 check that xstart and xend have same ndim and shape
  if (xstart.ndim() != xend.ndim())
    throw std::invalid_argument("xstart and xend must have the same number of dimensions");
  for (size_t i = 0; i < xstart.ndim(); ++i)
  {
    if (xstart.shape(i) != xend.shape(i))
      throw std::invalid_argument("xstart and xend must have the same shape");
  }

  // 3 check that the shape of p matches xstart.shape[:-1]
  if (p.ndim() != xstart.ndim() - 1)
    throw std::invalid_argument("p must have a shape equal to xstart.shape[:-1]");
  for (size_t i = 0; i < p.ndim(); ++i)
  {
    if (p.shape(i) != xstart.shape(i))
      throw std::invalid_argument("p must have a shape equal to xstart.shape[:-1]");
  }

  // 4 check that xstart, xend, img, img_origin, voxsize, p have the same device type
  if (xstart.device_type() != xend.device_type() ||
      xstart.device_type() != img.device_type() ||
      xstart.device_type() != img_origin.device_type() ||
      xstart.device_type() != voxsize.device_type() ||
      xstart.device_type() != p.device_type())
  {
    throw std::invalid_argument("All input arrays must be on the same device type");
  }

  // 5 check that xstart, xend, img, img_origin, voxsize, p have the same device ID
  if (xstart.device_id() != xend.device_id() ||
      xstart.device_id() != img.device_id() ||
      xstart.device_id() != img_origin.device_id() ||
      xstart.device_id() != voxsize.device_id() ||
      xstart.device_id() != p.device_id())
  {
    throw std::invalid_argument("All input arrays must be on the same device ID");
  }

  // 6 check that img_origin and voxsize have length 3
  if (img_origin.shape(0) != 3)
    throw std::invalid_argument("img_origin must be a 1D array with 3 elements");
  if (voxsize.shape(0) != 3)
    throw std::invalid_argument("voxsize must be a 1D array with 3 elements");

  // 7 compute the number of LORs as the product of all dimensions except the last
  // the last dimension must be 3 (3 floating point values per LOR endpoint)
  size_t nlors = 1;
  for (size_t i = 0; i < xstart.ndim() - 1; ++i)
  {
    nlors *= xstart.shape(i);
  }

  int img_dim[3] = {static_cast<int>(img.shape(0)),
                    static_cast<int>(img.shape(1)),
                    static_cast<int>(img.shape(2))};

  joseph3d_fwd(xstart.data(), xend.data(), img.data(), img_origin.data(), voxsize.data(), p.data(), nlors, img_dim, device_id, threadsperblock);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_back
void joseph3d_back_py(ConstFloatNDArray xstart,
                      ConstFloatNDArray xend,
                      Float3DArray img,
                      ConstFloat1D3ELArray img_origin,
                      ConstFloat1D3ELArray voxsize,
                      ConstFloatNDArray p,
                      int device_id = 0,
                      int threadsperblock = 64)
{
  // 1 check that ndim of xstart and xend >=2 and last dim ==3
  if (xstart.ndim() < 2 || xstart.shape(xstart.ndim() - 1) != 3)
    throw std::invalid_argument("xstart must have at least 2 dims and shape (..., 3)");

  // 2 check that xstart and xend have same ndim and shape
  if (xstart.ndim() != xend.ndim())
    throw std::invalid_argument("xstart and xend must have the same number of dimensions");
  for (size_t i = 0; i < xstart.ndim(); ++i)
  {
    if (xstart.shape(i) != xend.shape(i))
      throw std::invalid_argument("xstart and xend must have the same shape");
  }

  // 3 check that the shape of p matches xstart.shape[:-1]
  if (p.ndim() != xstart.ndim() - 1)
    throw std::invalid_argument("p must have a shape equal to xstart.shape[:-1]");
  for (size_t i = 0; i < p.ndim(); ++i)
  {
    if (p.shape(i) != xstart.shape(i))
      throw std::invalid_argument("p must have a shape equal to xstart.shape[:-1]");
  }

  // 4 check that xstart, xend, img, img_origin, voxsize, p have the same device type
  if (xstart.device_type() != xend.device_type() ||
      xstart.device_type() != img.device_type() ||
      xstart.device_type() != img_origin.device_type() ||
      xstart.device_type() != voxsize.device_type() ||
      xstart.device_type() != p.device_type())
  {
    throw std::invalid_argument("All input arrays must be on the same device type");
  }

  // 5 check that xstart, xend, img, img_origin, voxsize, p have the same device ID
  if (xstart.device_id() != xend.device_id() ||
      xstart.device_id() != img.device_id() ||
      xstart.device_id() != img_origin.device_id() ||
      xstart.device_id() != voxsize.device_id() ||
      xstart.device_id() != p.device_id())
  {
    throw std::invalid_argument("All input arrays must be on the same device ID");
  }

  // 6 check that img_origin and voxsize have length 3
  if (img_origin.shape(0) != 3)
    throw std::invalid_argument("img_origin must be a 1D array with 3 elements");
  if (voxsize.shape(0) != 3)
    throw std::invalid_argument("voxsize must be a 1D array with 3 elements");

  // 7 compute the number of LORs as the product of all dimensions except the last
  // the last dimension must be 3 (3 floating point values per LOR endpoint)
  size_t nlors = 1;
  for (size_t i = 0; i < xstart.ndim() - 1; ++i)
  {
    nlors *= xstart.shape(i);
  }

  int img_dim[3] = {static_cast<int>(img.shape(0)),
                    static_cast<int>(img.shape(1)),
                    static_cast<int>(img.shape(2))};

  joseph3d_back(xstart.data(), xend.data(), img.data(), img_origin.data(), voxsize.data(), p.data(), nlors, img_dim, device_id, threadsperblock);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_tof_sino_fwd
void joseph3d_tof_sino_fwd_py(ConstFloatNDArray xstart,
                              ConstFloatNDArray xend,
                              ConstFloat3DArray img,
                              ConstFloat1D3ELArray img_origin,
                              ConstFloat1D3ELArray voxsize,
                              FloatNDArray p,
                              float tofbin_width,
                              ConstFloatNDArray sigma_tof,
                              ConstFloatNDArray tofcenter_offset,
                              short n_tofbins,
                              float n_sigmas = 3.0f,
                              int device_id = 0,
                              int threadsperblock = 64)
{
  bool lor_dependent_sigma_tof;
  bool lor_dependent_tofcenter_offset;

  // 1 check that ndim of xstart and xend >=2 and last dim ==3
  if (xstart.ndim() < 2 || xstart.shape(xstart.ndim() - 1) != 3)
    throw std::invalid_argument("xstart must have at least 2 dims and shape (..., 3)");

  // 2 check that xstart and xend have same ndim and shape
  if (xstart.ndim() != xend.ndim())
    throw std::invalid_argument("xstart and xend must have the same number of dimensions");
  for (size_t i = 0; i < xstart.ndim(); ++i)
  {
    if (xstart.shape(i) != xend.shape(i))
      throw std::invalid_argument("xstart and xend must have the same shape");
  }

  // 3 check p has same ndim as xstart
  if (p.ndim() != xstart.ndim())
    throw std::invalid_argument("p must have same number of dimensions as xstart");
  for (size_t i = 0; i < (p.ndim() - 1); ++i)
  {
    if (p.shape(i) != xstart.shape(i))
      throw std::invalid_argument("shape of p[:-1] must match shape of xstart[:-1]");
  }
  // check that p.shape[-1] == n_tofbins
  if (p.shape(p.ndim() - 1) != static_cast<size_t>(n_tofbins))
    throw std::invalid_argument("last dimension of p must equal n_tofbins");

  // 4 check that xstart, xend, img, img_origin, voxsize, p have the same device type
  if (xstart.device_type() != xend.device_type() ||
      xstart.device_type() != img.device_type() ||
      xstart.device_type() != img_origin.device_type() ||
      xstart.device_type() != voxsize.device_type() ||
      xstart.device_type() != p.device_type())
  {
    throw std::invalid_argument("All input arrays must be on the same device type");
  }

  // 5 check that xstart, xend, img, img_origin, voxsize, p have the same device ID
  if (xstart.device_id() != xend.device_id() ||
      xstart.device_id() != img.device_id() ||
      xstart.device_id() != img_origin.device_id() ||
      xstart.device_id() != voxsize.device_id() ||
      xstart.device_id() != p.device_id())
  {
    throw std::invalid_argument("All input arrays must be on the same device ID");
  }

  // 6 check that img_origin and voxsize have length 3
  if (img_origin.shape(0) != 3)
    throw std::invalid_argument("img_origin must be a 1D array with 3 elements");
  if (voxsize.shape(0) != 3)
    throw std::invalid_argument("voxsize must be a 1D array with 3 elements");

  // 7 compute the number of LORs as the product of all dimensions except the last
  // the last dimension must be 3 (3 floating point values per LOR endpoint)
  size_t nlors = 1;
  for (size_t i = 0; i < xstart.ndim() - 1; ++i)
  {
    nlors *= xstart.shape(i);
  }

  int img_dim[3] = {static_cast<int>(img.shape(0)),
                    static_cast<int>(img.shape(1)),
                    static_cast<int>(img.shape(2))};

  // check that the shape of sigma_tof is either [1,] or xstart.shape[:-1]
  if (sigma_tof.ndim() == 1 && sigma_tof.shape(0) == 1)
  {
    lor_dependent_sigma_tof = false;
  }
  else if (sigma_tof.ndim() == xstart.ndim() - 1)
  {
    for (size_t i = 0; i < (sigma_tof.ndim()); ++i)
    {
      if (sigma_tof.shape(i) != xstart.shape(i))
        throw std::invalid_argument("shape of sigma_tof must match shape of xstart[:-1] or be scalar");
    }
    lor_dependent_sigma_tof = true;
  }
  else
  {
    throw std::invalid_argument("shape of sigma_tof must match shape of xstart[:-1] or be scalar");
  }

  // check that the shape of tofcenter_offset is either [1,] or xstart.shape[:-1]
  if (tofcenter_offset.ndim() == 1 && tofcenter_offset.shape(0) == 1)
  {
    lor_dependent_tofcenter_offset = false;
  }
  else if (tofcenter_offset.ndim() == xstart.ndim() - 1)
  {
    for (size_t i = 0; i < (tofcenter_offset.ndim()); ++i)
    {
      if (tofcenter_offset.shape(i) != xstart.shape(i))
        throw std::invalid_argument("shape of tofcenter_offset must match shape of xstart[:-1] or be scalar");
    }
    lor_dependent_tofcenter_offset = true;
  }
  else
  {
    throw std::invalid_argument("shape of tofcenter_offset must match shape of xstart[:-1] or be scalar");
  }

  joseph3d_tof_sino_fwd(xstart.data(),
                        xend.data(),
                        img.data(),
                        img_origin.data(),
                        voxsize.data(),
                        p.data(),
                        nlors,
                        img_dim,
                        tofbin_width,
                        sigma_tof.data(),
                        tofcenter_offset.data(),
                        n_sigmas,
                        n_tofbins,
                        static_cast<unsigned char>(lor_dependent_sigma_tof ? 1 : 0),
                        static_cast<unsigned char>(lor_dependent_tofcenter_offset ? 1 : 0),
                        device_id,
                        threadsperblock);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Wrapper for joseph3d_tof_sino_back
void joseph3d_tof_back_fwd_py(ConstFloatNDArray xstart,
                              ConstFloatNDArray xend,
                              Float3DArray img,
                              ConstFloat1D3ELArray img_origin,
                              ConstFloat1D3ELArray voxsize,
                              ConstFloatNDArray p,
                              float tofbin_width,
                              ConstFloatNDArray sigma_tof,
                              ConstFloatNDArray tofcenter_offset,
                              short n_tofbins,
                              float n_sigmas = 3.0f,
                              int device_id = 0,
                              int threadsperblock = 64)
{
  bool lor_dependent_sigma_tof;
  bool lor_dependent_tofcenter_offset;

  // 1 check that ndim of xstart and xend >=2 and last dim ==3
  if (xstart.ndim() < 2 || xstart.shape(xstart.ndim() - 1) != 3)
    throw std::invalid_argument("xstart must have at least 2 dims and shape (..., 3)");

  // 2 check that xstart and xend have same ndim and shape
  if (xstart.ndim() != xend.ndim())
    throw std::invalid_argument("xstart and xend must have the same number of dimensions");
  for (size_t i = 0; i < xstart.ndim(); ++i)
  {
    if (xstart.shape(i) != xend.shape(i))
      throw std::invalid_argument("xstart and xend must have the same shape");
  }

  // 3 check p has same ndim as xstart
  if (p.ndim() != xstart.ndim())
    throw std::invalid_argument("p must have same number of dimensions as xstart");
  for (size_t i = 0; i < (p.ndim() - 1); ++i)
  {
    if (p.shape(i) != xstart.shape(i))
      throw std::invalid_argument("shape of p[:-1] must match shape of xstart[:-1]");
  }
  // check that p.shape[-1] == n_tofbins
  if (p.shape(p.ndim() - 1) != static_cast<size_t>(n_tofbins))
    throw std::invalid_argument("last dimension of p must equal n_tofbins");

  // 4 check that xstart, xend, img, img_origin, voxsize, p have the same device type
  if (xstart.device_type() != xend.device_type() ||
      xstart.device_type() != img.device_type() ||
      xstart.device_type() != img_origin.device_type() ||
      xstart.device_type() != voxsize.device_type() ||
      xstart.device_type() != p.device_type())
  {
    throw std::invalid_argument("All input arrays must be on the same device type");
  }

  // 5 check that xstart, xend, img, img_origin, voxsize, p have the same device ID
  if (xstart.device_id() != xend.device_id() ||
      xstart.device_id() != img.device_id() ||
      xstart.device_id() != img_origin.device_id() ||
      xstart.device_id() != voxsize.device_id() ||
      xstart.device_id() != p.device_id())
  {
    throw std::invalid_argument("All input arrays must be on the same device ID");
  }

  // 6 check that img_origin and voxsize have length 3
  if (img_origin.shape(0) != 3)
    throw std::invalid_argument("img_origin must be a 1D array with 3 elements");
  if (voxsize.shape(0) != 3)
    throw std::invalid_argument("voxsize must be a 1D array with 3 elements");

  // 7 compute the number of LORs as the product of all dimensions except the last
  // the last dimension must be 3 (3 floating point values per LOR endpoint)
  size_t nlors = 1;
  for (size_t i = 0; i < xstart.ndim() - 1; ++i)
  {
    nlors *= xstart.shape(i);
  }

  int img_dim[3] = {static_cast<int>(img.shape(0)),
                    static_cast<int>(img.shape(1)),
                    static_cast<int>(img.shape(2))};

  // check that the shape of sigma_tof is either [1,] or xstart.shape[:-1]
  if (sigma_tof.ndim() == 1 && sigma_tof.shape(0) == 1)
  {
    lor_dependent_sigma_tof = false;
  }
  else if (sigma_tof.ndim() == xstart.ndim() - 1)
  {
    for (size_t i = 0; i < (sigma_tof.ndim()); ++i)
    {
      if (sigma_tof.shape(i) != xstart.shape(i))
        throw std::invalid_argument("shape of sigma_tof must match shape of xstart[:-1] or be scalar");
    }
    lor_dependent_sigma_tof = true;
  }
  else
  {
    throw std::invalid_argument("shape of sigma_tof must match shape of xstart[:-1] or be scalar");
  }

  // check that the shape of tofcenter_offset is either [1,] or xstart.shape[:-1]
  if (tofcenter_offset.ndim() == 1 && tofcenter_offset.shape(0) == 1)
  {
    lor_dependent_tofcenter_offset = false;
  }
  else if (tofcenter_offset.ndim() == xstart.ndim() - 1)
  {
    for (size_t i = 0; i < (tofcenter_offset.ndim()); ++i)
    {
      if (tofcenter_offset.shape(i) != xstart.shape(i))
        throw std::invalid_argument("shape of tofcenter_offset must match shape of xstart[:-1] or be scalar");
    }
    lor_dependent_tofcenter_offset = true;
  }
  else
  {
    throw std::invalid_argument("shape of tofcenter_offset must match shape of xstart[:-1] or be scalar");
  }

  joseph3d_tof_sino_back(xstart.data(),
                         xend.data(),
                         img.data(),
                         img_origin.data(),
                         voxsize.data(),
                         p.data(),
                         nlors,
                         img_dim,
                         tofbin_width,
                         sigma_tof.data(),
                         tofcenter_offset.data(),
                         n_sigmas,
                         n_tofbins,
                         static_cast<unsigned char>(lor_dependent_sigma_tof ? 1 : 0),
                         static_cast<unsigned char>(lor_dependent_tofcenter_offset ? 1 : 0),
                         device_id,
                         threadsperblock);
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

NB_MODULE(parallelproj_backend, m)
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

  m.def("joseph3d_fwd", &joseph3d_fwd_py,
        "xstart"_a.noconvert(), "xend"_a.noconvert(), "img"_a.noconvert(),
        "img_origin"_a.noconvert(), "voxsize"_a.noconvert(), "p"_a.noconvert(),
        "device_id"_a = 0, "threadsperblock"_a = 64,
        "Missing DOCSTRING for joseph3d_fwd");

  m.def("joseph3d_back", &joseph3d_back_py,
        "xstart"_a.noconvert(), "xend"_a.noconvert(), "img"_a.noconvert(),
        "img_origin"_a.noconvert(), "voxsize"_a.noconvert(), "p"_a.noconvert(),
        "device_id"_a = 0, "threadsperblock"_a = 64,
        "Missing DOCSTRING for joseph3d_back");

  m.def("joseph3d_tof_sino_fwd", &joseph3d_tof_sino_fwd_py,
        "xstart"_a.noconvert(), "xend"_a.noconvert(), "img"_a.noconvert(),
        "img_origin"_a.noconvert(), "voxsize"_a.noconvert(), "p"_a.noconvert(),
        "tofbin_width"_a,
        "sigma_tof"_a.noconvert(),
        "tofcenter_offset"_a.noconvert(),
        "n_tofbins"_a,
        "n_sigmas"_a = 3.0f,
        "device_id"_a = 0, "threadsperblock"_a = 64,
        "Missing DOCSTRING for joseph3d_tof_sino_fwd");

  m.def("joseph3d_tof_sino_back", &joseph3d_tof_back_fwd_py,
        "xstart"_a.noconvert(), "xend"_a.noconvert(), "img"_a.noconvert(),
        "img_origin"_a.noconvert(), "voxsize"_a.noconvert(), "p"_a.noconvert(),
        "tofbin_width"_a,
        "sigma_tof"_a.noconvert(),
        "tofcenter_offset"_a.noconvert(),
        "n_tofbins"_a,
        "n_sigmas"_a = 3.0f,
        "device_id"_a = 0, "threadsperblock"_a = 64,
        "Missing DOCSTRING for joseph3d_tof_sino_back");
}

//// Wrapper for joseph3d_tof_sino_fwd
// void joseph3d_tof_sino_fwd_py(py::object xstart,
//                               py::object xend,
//                               py::object img,
//                               py::object img_origin,
//                               py::object voxsize,
//                               py::object p,
//                               float tofbin_width,
//                               py::object sigma_tof,
//                               py::object tofcenter_offset,
//                               short n_tofbins,
//                               float n_sigmas = 3.0f,
//                               bool lor_dependent_sigma_tof = false,
//                               bool lor_dependent_tofcenter_offset = false,
//                               int device_id = 0,
//                               int threadsperblock = 64)
//{
//   // Extract raw pointers and shapes
//   auto [xstart_ptr, xstart_shape] = extract_pointer_and_shape<float>(xstart);
//   auto [xend_ptr, xend_shape] = extract_pointer_and_shape<float>(xend);
//   auto [img_ptr, img_shape] = extract_pointer_and_shape<float>(img);
//   auto [img_origin_ptr, img_origin_shape] = extract_pointer_and_shape<float>(img_origin);
//   auto [voxsize_ptr, voxsize_shape] = extract_pointer_and_shape<float>(voxsize);
//   auto [p_ptr, p_shape] = extract_pointer_and_shape<float>(p);
//   auto [sigma_ptr, sigma_shape] = extract_pointer_and_shape<float>(sigma_tof);
//   auto [tofcenter_ptr, tofcenter_shape] = extract_pointer_and_shape<float>(tofcenter_offset);
//
//   // Validate shapes (common checks)
//   if (xstart_shape.size() < 2 || xstart_shape.back() != 3)
//   {
//     throw std::invalid_argument("xstart must have at least 2 dims and shape (..., 3)");
//   }
//   if (xend_shape.size() < 2 || xend_shape.back() != 3)
//   {
//     throw std::invalid_argument("xend must have at least 2 dims and shape (..., 3)");
//   }
//   if (img_shape.size() != 3)
//   {
//     throw std::invalid_argument("img must be a 3D array");
//   }
//   if (img_origin_shape.size() != 1 || img_origin_shape[0] != 3)
//   {
//     throw std::invalid_argument("img_origin must be a 1D array with 3 elements");
//   }
//   if (voxsize_shape.size() != 1 || voxsize_shape[0] != 3)
//   {
//     throw std::invalid_argument("voxsize must be a 1D array with 3 elements");
//   }
//
//   // Calculate nlors using xstart_shape (multiply shape except the last dimension)
//   size_t nlors = std::accumulate(xstart_shape.begin(), xstart_shape.end() - 1, static_cast<size_t>(1), std::multiplies<size_t>());
//
//   // Validate p has shape xstart.shape[:-1] + [n_tofbins]
//   if (p_shape.size() != xstart_shape.size())
//   {
//     throw std::invalid_argument("p must have same number of dimensions as xstart and last dim == n_tofbins");
//   }
//   for (size_t d = 0; d + 1 < xstart_shape.size(); ++d)
//   {
//     if (p_shape[d] != xstart_shape[d])
//       throw std::invalid_argument("p shape must match xstart.shape[:-1]");
//   }
//   if (p_shape.back() != static_cast<size_t>(n_tofbins))
//   {
//     throw std::invalid_argument("last dimension of p must equal n_tofbins");
//   }
//
//   int img_dim[3] = {static_cast<int>(img_shape[0]), static_cast<int>(img_shape[1]), static_cast<int>(img_shape[2])};
//
//   // validate sigma_tof / tofcenter_offset shapes wrt lor_dependent flags
//   if (lor_dependent_sigma_tof)
//   {
//     if (sigma_shape.size() == 0 || sigma_shape[0] != nlors)
//       throw std::invalid_argument("sigma_tof must have length nlors when lor_dependent_sigma_tof is true");
//   }
//   else
//   {
//     // allow scalar or length >=1 (we will use sigma_ptr[0])
//     if (sigma_shape.size() > 0 && sigma_shape[0] < 1)
//       throw std::invalid_argument("sigma_tof must contain at least one element");
//   }
//
//   if (lor_dependent_tofcenter_offset)
//   {
//     if (tofcenter_shape.size() == 0 || tofcenter_shape[0] != nlors)
//       throw std::invalid_argument("tofcenter_offset must have length nlors when lor_dependent_tofcenter_offset is true");
//   }
//   else
//   {
//     if (tofcenter_shape.size() > 0 && tofcenter_shape[0] < 1)
//       throw std::invalid_argument("tofcenter_offset must contain at least one element");
//   }
//
//   // Call the C++ function
//   joseph3d_tof_sino_fwd(xstart_ptr,
//                         xend_ptr,
//                         img_ptr,
//                         img_origin_ptr,
//                         voxsize_ptr,
//                         p_ptr,
//                         nlors,
//                         img_dim,
//                         tofbin_width,
//                         sigma_ptr,
//                         tofcenter_ptr,
//                         n_sigmas,
//                         n_tofbins,
//                         static_cast<unsigned char>(lor_dependent_sigma_tof ? 1 : 0),
//                         static_cast<unsigned char>(lor_dependent_tofcenter_offset ? 1 : 0),
//                         device_id,
//                         threadsperblock);
// }
//
//// Wrapper for joseph3d_tof_sino_back
// void joseph3d_tof_sino_back_py(py::object xstart,
//                                py::object xend,
//                                py::object img,
//                                py::object img_origin,
//                                py::object voxsize,
//                                py::object p,
//                                float tofbin_width,
//                                py::object sigma_tof,
//                                py::object tofcenter_offset,
//                                short n_tofbins,
//                                float n_sigmas = 3.0f,
//                                bool lor_dependent_sigma_tof = false,
//                                bool lor_dependent_tofcenter_offset = false,
//                                int device_id = 0,
//                                int threadsperblock = 64)
//{
//   // Extract raw pointers and shapes
//   auto [xstart_ptr, xstart_shape] = extract_pointer_and_shape<float>(xstart);
//   auto [xend_ptr, xend_shape] = extract_pointer_and_shape<float>(xend);
//   auto [img_ptr, img_shape] = extract_pointer_and_shape<float>(img);
//   auto [img_origin_ptr, img_origin_shape] = extract_pointer_and_shape<float>(img_origin);
//   auto [voxsize_ptr, voxsize_shape] = extract_pointer_and_shape<float>(voxsize);
//   auto [p_ptr, p_shape] = extract_pointer_and_shape<float>(p);
//   auto [sigma_ptr, sigma_shape] = extract_pointer_and_shape<float>(sigma_tof);
//   auto [tofcenter_ptr, tofcenter_shape] = extract_pointer_and_shape<float>(tofcenter_offset);
//
//   // Validate shapes (common checks)
//   if (xstart_shape.size() < 2 || xstart_shape.back() != 3)
//     throw std::invalid_argument("xstart must have at least 2 dims and shape (..., 3)");
//   if (xend_shape.size() < 2 || xend_shape.back() != 3)
//     throw std::invalid_argument("xend must have at least 2 dims and shape (..., 3)");
//   if (img_shape.size() != 3)
//     throw std::invalid_argument("img must be a 3D array");
//   if (img_origin_shape.size() != 1 || img_origin_shape[0] != 3)
//     throw std::invalid_argument("img_origin must be a 1D array with 3 elements");
//   if (voxsize_shape.size() != 1 || voxsize_shape[0] != 3)
//     throw std::invalid_argument("voxsize must be a 1D array with 3 elements");
//
//   // Calculate nlors using xstart_shape (multiply shape except the last dimension)
//   size_t nlors = std::accumulate(xstart_shape.begin(), xstart_shape.end() - 1, static_cast<size_t>(1), std::multiplies<size_t>());
//
//   // Validate p has shape xstart.shape[:-1] + [n_tofbins]
//   if (p_shape.size() != xstart_shape.size())
//     throw std::invalid_argument("p must have same number of dimensions as xstart and last dim == n_tofbins");
//   for (size_t d = 0; d + 1 < xstart_shape.size(); ++d)
//   {
//     if (p_shape[d] != xstart_shape[d])
//       throw std::invalid_argument("p shape must match xstart.shape[:-1]");
//   }
//   if (p_shape.back() != static_cast<size_t>(n_tofbins))
//     throw std::invalid_argument("last dimension of p must equal n_tofbins");
//
//   int img_dim[3] = {static_cast<int>(img_shape[0]), static_cast<int>(img_shape[1]), static_cast<int>(img_shape[2])};
//
//   // validate sigma_tof / tofcenter_offset shapes wrt lor_dependent flags
//   if (lor_dependent_sigma_tof)
//   {
//     if (sigma_shape.size() == 0 || sigma_shape[0] != nlors)
//       throw std::invalid_argument("sigma_tof must have length nlors when lor_dependent_sigma_tof is true");
//   }
//   else
//   {
//     if (sigma_shape.size() > 0 && sigma_shape[0] < 1)
//       throw std::invalid_argument("sigma_tof must contain at least one element");
//   }
//
//   if (lor_dependent_tofcenter_offset)
//   {
//     if (tofcenter_shape.size() == 0 || tofcenter_shape[0] != nlors)
//       throw std::invalid_argument("tofcenter_offset must have length nlors when lor_dependent_tofcenter_offset is true");
//   }
//   else
//   {
//     if (tofcenter_shape.size() > 0 && tofcenter_shape[0] < 1)
//       throw std::invalid_argument("tofcenter_offset must contain at least one element");
//   }
//
//   // Call the C++ function
//   joseph3d_tof_sino_back(xstart_ptr,
//                          xend_ptr,
//                          img_ptr,
//                          img_origin_ptr,
//                          voxsize_ptr,
//                          p_ptr,
//                          nlors,
//                          img_dim,
//                          tofbin_width,
//                          sigma_ptr,
//                          tofcenter_ptr,
//                          n_sigmas,
//                          n_tofbins,
//                          static_cast<unsigned char>(lor_dependent_sigma_tof ? 1 : 0),
//                          static_cast<unsigned char>(lor_dependent_tofcenter_offset ? 1 : 0),
//                          device_id,
//                          threadsperblock);
// }
//
//// Pybind11 module definition
// PYBIND11_MODULE(parallelproj_backend, m)
//{
//   m.doc() = "Python bindings for parallelproj backend";
//
//   // Expose the project version as __version__
// #ifdef PROJECT_VERSION
//   m.attr("__version__") = PROJECT_VERSION;
// #else
//   m.attr("__version__") = "unknown";
// #endif
//
//   // Expose the PARALLELPROJ_CUDA definition as a Python constant
// #ifdef PARALLELPROJ_CUDA
//   m.attr("PARALLELPROJ_CUDA") = PARALLELPROJ_CUDA;
// #else
//   m.attr("PARALLELPROJ_CUDA") = 0; // Default to 0 if not defined
// #endif
//
//   m.def("joseph3d_fwd", &joseph3d_fwd_py, R"pbdoc(
//     Non-TOF forward projection using the Joseph 3D algorithm. (adjoint of joseph3d_back())
//
//     Parameters:
//     -----------
//     xstart : array-like
//         array of size [...,3] with the world coordinates of the start points of the LORs.
//
//     xend : array-like
//         array of size [...,3] with the world coordinates of the end points of the LORs.
//
//     img : array-like
//         3D array of shape [n0,n1,n2] containing the 3D image used for forward projection.
//
//     img_origin : array-like
//         array [x0_0, x0_1, x0_2] with the world coordinates of the center of the [0,0,0] voxel.
//
//     voxsize : array-like
//         array [vs0, vs1, vs2] of the voxel sizes (same units as world coordinates).
//
//     p : array-like
//         array of size [xstart.shape[:-1]] where the forward projection results will be stored.
//
//     device_id : int, optional
//         ID of the device to use for computation (default: 0).
//
//     threadsperblock : int, optional
//         Number of threads per block for GPU computation (default: 64).
//
//     Returns:
//     --------
//     None
//)pbdoc",
//         py::arg("xstart"), py::arg("xend"), py::arg("img"), py::arg("img_origin"),
//         py::arg("voxsize"), py::arg("p"), py::arg("device_id") = 0, py::arg("threadsperblock") = 64);
//
//   m.def("joseph3d_back", &joseph3d_back_py, R"pbdoc(
//     Non-TOF back projection using the Joseph 3D algorithm (adjoint of joseph3_fwd).
//
//     Parameters:
//     -----------
//     xstart : array-like
//         Array of size [...,3] with the world coordinates of the start points of the LORs.
//
//     xend : array-like
//         Array of size [...,3] with the world coordinates of the end points of the LORs.
//
//     img : array-like
//         3D array of shape [n0,n1,n2] containing the 3D image used for back projection (output).
//         The pixel [i,j,k] is stored at [n1*n2*i + n2*j + k].
//         !! Values are added to the existing array !!
//
//     img_origin : array-like
//         Array [x0_0, x0_1, x0_2] with the world coordinates of the center of the [0,0,0] voxel.
//
//     voxsize : array-like
//         Array [vs0, vs1, vs2] of the voxel sizes (same units as world coordinates).
//
//     p : array-like
//         Array of size [xstart.shape[:-1]] containing the values to be back projected.
//
//     device_id : int, optional
//         ID of the device to use for computation (default: 0).
//
//     threadsperblock : int, optional
//         Number of threads per block for GPU computation (default: 64).
//
//     Returns:
//     --------
//     None
//)pbdoc",
//         py::arg("xstart"), py::arg("xend"), py::arg("img"), py::arg("img_origin"),
//         py::arg("voxsize"), py::arg("p"), py::arg("device_id") = 0, py::arg("threadsperblock") = 64);
//
//   m.def("joseph3d_tof_sino_fwd", &joseph3d_tof_sino_fwd_py, R"pbdoc(
//     sinogram TOF forward projection using the Joseph 3D algorithm and a Gaussian TOF model integrated over TOF bins
//
//     Parameters:
//     -----------
//     xstart : array-like
//         array of size [...,3] with the world coordinates of the start points of the LORs.
//
//     xend : array-like
//         array of size [...,3] with the world coordinates of the end points of the LORs.
//
//     img : array-like
//         3D array of shape [n0,n1,n2] containing the 3D image used for forward projection.
//
//     img_origin : array-like
//         array [x0_0, x0_1, x0_2] with the world coordinates of the center of the [0,0,0] voxel.
//
//     voxsize : array-like
//         array [vs0, vs1, vs2] of the voxel sizes (same units as world coordinates).
//
//     p : array-like
//         array of size [xstart.shape[:-1],n_tofbins] where the forward projection results will be stored.
//
//     tofbin_width : float
//         Width of the TOF bins.
//
//     sigma_tof : array-like
//         Array of size [n_tofbins] with the sigma values for the TOF Gaussian.
//
//     tofcenter_offset : array-like
//         Array of size [n_tofbins] with the center offsets for the TOF Gaussian.
//
//     n_tofbins : int
//         Number of TOF bins.
//
//     n_sigmas : float, optional
//         Number of sigmas beyond which the TOF Gaussian is truncated.
//         TOF weights are normalized to sum to 1 within +/- n_sigmas.
//         Default is 3.0.
//
//     lor_dependent_sigma_tof : bool, optional
//         If True, sigma_tof is interpreted as a Lorentz-dependent array of length nlors.
//         Default is False.
//
//     lor_dependent_tofcenter_offset : bool, optional
//         If True, tofcenter_offset is interpreted as a Lorentz-dependent array of length nlors.
//         Default is False.
//
//     device_id : int, optional
//         ID of the device to use for computation (default: 0).
//
//     threadsperblock : int, optional
//         Number of threads per block for GPU computation (default: 64).
//
//     Returns:
//     --------
//     None
//)pbdoc",
//         py::arg("xstart"), py::arg("xend"), py::arg("img"), py::arg("img_origin"),
//         py::arg("voxsize"), py::arg("p"), py::arg("tofbin_width"), py::arg("sigma_tof"),
//         py::arg("tofcenter_offset"), py::arg("n_tofbins"), py::arg("n_sigmas") = 3.0f,
//         py::arg("lor_dependent_sigma_tof") = false, py::arg("lor_dependent_tofcenter_offset") = false,
//         py::arg("device_id") = 0, py::arg("threadsperblock") = 64);
//
//   m.def("joseph3d_tof_sino_back", &joseph3d_tof_sino_back_py, R"pbdoc(
//     TOF sinogram backprojection using the Joseph 3D algorithm.
//
//     Parameters:
//     -----------
//     xstart : array-like
//         array of size [...,3] with the world coordinates of the start points of the LORs.
//     xend : array-like
//         array of size [...,3] with the world coordinates of the end points of the LORs.
//     img : array-like
//         3D array of shape [n0,n1,n2] containing the image to accumulate into (output).
//     img_origin : array-like
//         array [x0_0, x0_1, x0_2] with the world coordinates of the center of the [0,0,0] voxel.
//     voxsize : array-like
//         array [vs0, vs1, vs2] of the voxel sizes (same units as world coordinates).
//     p : array-like
//         array of size [xstart.shape[:-1], n_tofbins] containing TOF sinogram data.
//     tofbin_width : float
//         Width of the TOF bins.
//     sigma_tof : array-like
//         Array of size [1] or [nlors] depending on lor_dependent_sigma_tof.
//     tofcenter_offset : array-like
//         Array of size [1] or [nlors] depending on lor_dependent_tofcenter_offset.
//     n_tofbins : int
//         Number of TOF bins per LOR.
//     n_sigmas : float, optional
//         Number of sigmas for TOF kernel radius (default 3.0).
//     lor_dependent_sigma_tof : bool, optional
//         If True, sigma_tof has length nlors.
//     lor_dependent_tofcenter_offset : bool, optional
//         If True, tofcenter_offset has length nlors.
//     device_id : int, optional
//         CUDA device id (default 0).
//     threadsperblock : int, optional
//         Threads per block for GPU (default 64).
//)pbdoc",
//         py::arg("xstart"), py::arg("xend"), py::arg("img"), py::arg("img_origin"),
//         py::arg("voxsize"), py::arg("p"), py::arg("tofbin_width"), py::arg("sigma_tof"),
//         py::arg("tofcenter_offset"), py::arg("n_tofbins"), py::arg("n_sigmas") = 3.0f,
//         py::arg("lor_dependent_sigma_tof") = false, py::arg("lor_dependent_tofcenter_offset") = false,
//         py::arg("device_id") = 0, py::arg("threadsperblock") = 64);
// }
//
