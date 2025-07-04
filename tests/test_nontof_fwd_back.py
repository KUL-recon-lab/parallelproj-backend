import math
import numpy as np
import parallelproj_backend as pp

from types import ModuleType

from .config import pytestmark


def test_forward_and_back_projection(xp: ModuleType, dev: str):
    """Test the forward projection."""

    ############################################################################
    # setup data

    img_dim = (2, 3, 4)
    voxsize = xp.asarray([4.0, 3.0, 2.0], dtype=xp.float32, device=dev)
    img_origin = (
        -0.5 * xp.asarray(img_dim, dtype=xp.float32, device=dev) + 0.5
    ) * voxsize

    # Read the image from file
    img = xp.reshape(
        xp.asarray(np.loadtxt("img.txt", dtype=np.float32), device=dev), img_dim
    )

    # Read the ray start and end coordinates from file
    vstart = xp.reshape(
        xp.asarray(np.loadtxt("vstart.txt", dtype=np.float32), device=dev), (2, 5, 3)
    )
    vend = xp.reshape(
        xp.asarray(np.loadtxt("vend.txt", dtype=np.float32), device=dev), (2, 5, 3)
    )

    # Calculate the start and end coordinates in world coordinates
    xstart = vstart * voxsize + img_origin
    xend = vend * voxsize + img_origin

    # Read the expected forward values from file
    expected_fwd_vals = xp.reshape(
        xp.asarray(np.loadtxt("expected_fwd_vals.txt", dtype=np.float32), device=dev),
        xstart.shape[:-1],
    )

    ############################################################################

    # Allocate memory for forward projection results
    img_fwd = xp.zeros(xstart.shape[:-1], dtype=xp.float32, device=dev)

    # Perform forward projection
    pp.joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd)

    # Check if we got the expected results
    eps = 1e-7
    assert (
        float(xp.max(xp.abs(img_fwd - expected_fwd_vals))) < eps
    ), "Forward projection test failed."

    # Allocate memory for back projection results
    bimg = xp.zeros(img_dim, dtype=xp.float32, device=dev)
    ones = xp.ones(xstart.shape[:-1], dtype=xp.float32, device=dev)

    # Perform back projection
    pp.joseph3d_back(xstart, xend, bimg, img_origin, voxsize, ones)

    # Check the results
    ip1 = float(xp.sum(img * bimg))
    ip2 = float(xp.sum(img_fwd * ones))

    eps = 1e-7
    assert abs(ip1 - ip2) / abs(ip1) < eps, "Back projection test failed."


def test_box_projection(xp: ModuleType, dev: str):
    """test forward projection through a uniform box along different axes / angles"""

    # generate an image box full of ones with side length 100mm but non-uniform voxel size
    img_dim = (50, 100, 25)
    voxel_size = xp.asarray([2.0, 1.0, 4.0], device=dev, dtype=xp.float32)
    img_origin = -50 + 0.5 * voxel_size

    # Allocate memory for back projection results
    img = xp.ones(img_dim, dtype=xp.float32, device=dev)

    # create xstart and xend arrays
    xstart = xp.asarray(
        [
            [100, 0, 0],  # exp val 100
            [50, 0, 0],  # exp val 100
            [0, 50, 0],  # exp val 100
            [0, 0, 50],  # exp val 100
            [40, 0, 0],  # exp val 80
            [0, 40, 0],  # exp val 80
            [0, 0, 40],  # exp val 80
            [50, 5, 0],  # exp val sqrt(100^2 + 9^2) = 100.404
            [0, 50, 5],  # exp val sqrt(100^2 + 9^2) = 100.404
            [5, 0, 50],  # exp val sqrt(100^2 + 9^2) = 100.404
            [50, 5, -2],  # exp val sqrt(100^2 + 9^2 + 5^2) = 100.528
            [-2, 50, 5],  # exp val sqrt(100^2 + 9^2 + 5^2) = 100.528
            [5, -2, 50],  # exp val sqrt(100^2 + 9^2 + 5^2) = 100.528
        ],
        device=dev,
        dtype=xp.float32,
    )

    xend = xp.asarray(
        [
            [-100, 0, 0],
            [-50, 0, 0],
            [0, -50, 0],
            [0, 0, -50],
            [-40, 0, 0],
            [0, -40, 0],
            [0, 0, -40],
            [-50, -4, 0],
            [0, -50, -4],
            [-4, 0, -50],
            [-50, -4, 3],
            [3, -50, -4],
            [-4, 3, -50],
        ],
        device=dev,
        dtype=xp.float32,
    )

    exp_vals = xp.asarray(
        [
            100.0,
            100.0,
            100.0,
            100.0,
            80.0,
            80.0,
            80.0,
            math.sqrt(100**2 + 9**2),  # 100.404
            math.sqrt(100**2 + 9**2),  # 100.404
            math.sqrt(100**2 + 9**2),  # 100.404
            math.sqrt(100**2 + 9**2 + 5**2),  # 100.528
            math.sqrt(100**2 + 9**2 + 5**2),  # 100.528
            math.sqrt(100**2 + 9**2 + 5**2),  # 100.528
        ],
        device=dev,
        dtype=xp.float32,
    )

    img_fwd = xp.zeros(xstart.shape[0], dtype=xp.float32, device = dev)

    # Perform back projection
    pp.joseph3d_fwd(xstart, xend, img, img_origin, voxel_size, img_fwd)

    eps = 1e-5
    assert (
        float(xp.max(xp.abs(img_fwd - exp_vals))) < eps
    ), "Forward box projection test failed."
