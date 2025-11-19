import pytest
import math
import random
import parallelproj_backend as ppb

from types import ModuleType
from .config import pytestmark



def effective_tof_kernel(dx: float, sigma_t: float, tbin_width: float) -> float:
    """Gaussian integrated over a tof bin width."""
    sqrt2 = 1.414213562373095
    return 0.5 * (
        math.erf((dx + 0.5 * tbin_width) / (sqrt2 * sigma_t))
        - math.erf((dx - 0.5 * tbin_width) / (sqrt2 * sigma_t))
    )


@pytest.mark.parametrize("direc, sigma_tof, num_tofbins, tof_center_offset", [
    (0, 4.5, 41, 0.0),
    (1, 4.5, 41, 0.0),
    (2, 4.5, 41, 0.0),
    (0, 3.5, 41, 0.0),
    (0, 3.5, 40, 0.0),
    (0, 8.5, 41, 0.0),
    (0, 3.5, 41, -2.5),
    (0, 3.5, 41, 2.5),
])
def test_tof_sino_fwd(
    xp: ModuleType,
    dev: str,
    direc: int,
    sigma_tof: float,
    num_tofbins: int,
    tof_center_offset: float,
    voxsize: tuple[float, float, float] = (2.2, 2.5, 2.7),
    vox_num: int = 7,
    tofbin_width: float = 3.0,
    num_sigmas: float = 3.0,
    nvox: int = 19,
    verbose: bool = False,
):

    img_dim = tuple(nvox if i == direc else 1 for i in range(3))
    xstart = tuple(60.0 if i == direc else 0 for i in range(3))
    xend = tuple(-60.0 if i == direc else 0 for i in range(3))

    ###########################################

    n0, n1, n2 = img_dim

    img_origin = (
        (-(n0 / 2) + 0.5) * voxsize[0],
        (-(n1 / 2) + 0.5) * voxsize[1],
        (-(n2 / 2) + 0.5) * voxsize[2],
    )

    d0 = xend[0] - xstart[0]
    d1 = xend[1] - xstart[1]
    d2 = xend[2] - xstart[2]

    # %%
    # get the correction factor cf

    dr_sq = (d0**2, d1**2, d2**2)
    direction = max(range(3), key=lambda i: dr_sq[i])

    sum_sq = sum(dr_sq)
    cos_sq = dr_sq[direction] / sum_sq

    cf = voxsize[direction] / (cos_sq**0.5)
    # %%

    #####################################
    #####################################
    #####################################

    # TOF related calculations

    istart = 0
    iend = img_dim[direction] - 1

    # max tof bin diffference where kernel is effectively non-zero
    # max_tof_bin_diff = num_sigmas * max(sigma_tof, tofbin_width) / tofbin_width
    costheta: float = voxsize[direction] / cf
    max_tof_bin_diff: float = num_sigmas * sigma_tof / (tofbin_width)

    # calculate the where the TOF bins are located along the projected line
    # in world coordinates
    sign: int = 1 if xend[direction] >= xstart[direction] else -1
    # the tof bin centers (in world coordinates projected to the axis along which we step through the volume)
    # are at it*a_tof + b_tof for it in range(num_tofbins)
    tof_origin: float = (
        0.5 * (xstart[direction] + xend[direction])
        - sign * (0.5 * num_tofbins - 0.5) * (tofbin_width * costheta)
        + tof_center_offset * costheta
    )
    tof_slope: float = sign * (tofbin_width * costheta)

    ### TOF offset and increment per voxel step in direction
    at: float = sign * cf / tofbin_width
    bt: float = (img_origin[direction] - tof_origin) / tof_slope

    # it_f is the index of the TOF bin at the current plane
    it_f: float = istart * at + bt

    #####################################
    #####################################
    #####################################

    img = xp.zeros(img_dim, dtype=xp.float32, device=dev)
    if direction == 0:
        img[vox_num, 0, 0] = 1.0
    elif direction == 1:
        img[0, vox_num, 0] = 1.0
    elif direction == 2:
        img[0, 0, vox_num] = 1.0
    else:
        raise ValueError("direction must be 0, 1, or 2")

    p_tof_ref = xp.zeros((1, num_tofbins), dtype=xp.float32, device=dev)

    for i in range(istart, iend + 1):
        # min and max tof bin for which we have to calculate tof weights
        it_min = math.floor(it_f - max_tof_bin_diff)
        it_max = math.ceil(it_f + max_tof_bin_diff)

        tof_weights = xp.zeros(it_max + 1 - it_min, device=dev, dtype=xp.float32)
        for k, it in enumerate(range(it_min, it_max + 1)):
            dist = abs(it_f - it) * tofbin_width
            tof_weights[k] = effective_tof_kernel(dist, sigma_tof, tofbin_width)

        tof_weights /= xp.sum(tof_weights)

        for k, it in enumerate(range(it_min, it_max + 1)):
            if direction == 0:
                p_tof_ref[0, it] += tof_weights[k] * cf * img[i,0,0]
            elif direction == 1:
                p_tof_ref[0, it] += tof_weights[k] * cf * img[0,i,0]
            elif direction == 2:
                p_tof_ref[0, it] += tof_weights[k] * cf * img[0,0,i]
            else:
                raise ValueError("direction must be 0, 1, or 2")

        it_f += at

    # %%

    # parallelproj-backend based sinogram TOF forward projection

    p_tof = xp.zeros((1, num_tofbins), dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_sino_fwd(
        xp.asarray([xstart], dtype=xp.float32, device=dev),
        xp.asarray([xend], dtype=xp.float32, device=dev),
        img,
        xp.asarray(img_origin, dtype=xp.float32, device=dev),
        xp.asarray(voxsize, dtype=xp.float32, device=dev),
        p_tof,
        tofbin_width,
        xp.asarray([sigma_tof], dtype=xp.float32, device=dev),
        xp.asarray([tof_center_offset], dtype=xp.float32, device=dev),
        num_tofbins,
        n_sigmas=num_sigmas,
    )

    if verbose:
        for i in range(num_tofbins):
            print(
                f"bin {i:2d}: proj_ref = {p_tof_ref[0,i]:.5E}, proj_ppb = {p_tof[0,i]:.5E}"
            )

    # check whether the projection is equal to the expected one
    for i in range(num_tofbins):
        assert math.isclose(float(p_tof_ref[0, i]), float(p_tof[0, i]), abs_tol=1e-5)

    # since we are forward projecting an image of a single voxel containg a value of 1
    # the sum over TOF should be the voxel size (if we have enough TOF bins)
    assert math.isclose(float(xp.sum(p_tof)), voxsize[direction], abs_tol=1e-6)

@pytest.mark.parametrize("sigma_tof, num_tofbins", [(4.5, 41), (4.5, 40), (8.5, 41), (2.5, 41)])
def test_tof_sino_adjointness(
    xp: ModuleType,
    dev: str,
    sigma_tof: float,
    num_tofbins: int,
    voxsize: tuple[float, float, float] = (2.2, 2.5, 2.7),
    tofbin_width: float = 3.0,
    num_sigmas: float = 3.0,
    tof_center_offset: float = 0.0,
    nvox: int = 19,
    verbose: bool = False,
    nlors = 200):

    #------
    random.seed(42)
    img_dim = (nvox, nvox, nvox)

    n0, n1, n2 = img_dim
    img_origin = (
        (-(n0 / 2) + 0.5) * voxsize[0],
        (-(n1 / 2) + 0.5) * voxsize[1],
        (-(n2 / 2) + 0.5) * voxsize[2],
    )


    img = xp.zeros(img_dim, dtype=xp.float32, device=dev)
    # fill the image with uniform random values using python's random module
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                img[i, j, k] = random.uniform(0.0, 1.0)

    xstart = xp.zeros((nlors, 3), dtype=xp.float32, device=dev)
    xend = xp.zeros((nlors, 3), dtype=xp.float32, device=dev)

    # fill xstart and xend with random points on a sphere with radius 45
    r = 45.0
    for i in range(nlors):
        theta = random.uniform(0.0, math.pi)
        phi = random.uniform(0.0, 2.0 * math.pi)
        xstart[i, 0] = r * math.sin(theta) * math.cos(phi)
        xstart[i, 1] = r * math.sin(theta) * math.sin(phi)
        xstart[i, 2] = r * math.cos(theta)

        theta = random.uniform(0.0, math.pi)
        phi = random.uniform(0.0, 2.0 * math.pi)
        xend[i, 0] = r * math.sin(theta) * math.cos(phi)
        xend[i, 1] = r * math.sin(theta) * math.sin(phi)
        xend[i, 2] = r * math.cos(theta)


    # simulate LOR-dependent TOF resolution and center offsets
    sigma_tof_array = xp.zeros(nlors, dtype=xp.float32, device=dev)
    for i in range(nlors):
        sigma_tof_array[i] = sigma_tof * random.uniform(0.9,1.1)

    tof_center_offset_array = xp.zeros(nlors, dtype=xp.float32, device=dev)
    for i in range(nlors):
        tof_center_offset_array[i] = tof_center_offset + random.uniform(-2.0,2.0)

    img_fwd = xp.zeros((nlors, num_tofbins), dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_sino_fwd(
        xstart,
        xend,
        img,
        xp.asarray(img_origin, dtype=xp.float32, device=dev),
        xp.asarray(voxsize, dtype=xp.float32, device=dev),
        img_fwd,
        tofbin_width,
        sigma_tof_array,
        tof_center_offset_array,
        num_tofbins,
        n_sigmas=num_sigmas,
    )

    # back project a random TOF sinogram
    y = xp.zeros((nlors, num_tofbins), dtype=xp.float32, device=dev)
    for i in range(nlors):
        for j in range(num_tofbins):
            y[i, j] = random.uniform(0.0, 1.0)

    y_back = xp.zeros(img_dim, dtype=xp.float32, device=dev)
    ppb.joseph3d_tof_sino_back(
        xstart,
        xend,
        y_back,
        xp.asarray(img_origin, dtype=xp.float32, device=dev),
        xp.asarray(voxsize, dtype=xp.float32, device=dev),
        y,
        tofbin_width,
        sigma_tof_array,
        tof_center_offset_array,
        num_tofbins,
        n_sigmas=num_sigmas,
    )

    # test the adjointness property
    innerprod1 = float(xp.sum(img_fwd * y))
    innerprod2 = float(xp.sum(img * y_back))

    if verbose:
        print(f"Inner product 1: {innerprod1:.5E}")
        print(f"Inner product 2: {innerprod2:.5E}")

    try:
        assert math.isclose(innerprod1, innerprod2, abs_tol=3e-4)
    except:
        print(sigma_tof, num_tofbins)
        breakpoint()

    # do a non-TOF forward projection and check whether the sum over TOF bins equals the non-TOF projection
    img_fwd_nontof = xp.zeros(nlors, dtype=xp.float32, device=dev)
    ppb.joseph3d_fwd(
        xstart,
        xend,
        img,
        xp.asarray(img_origin, dtype=xp.float32, device=dev),
        xp.asarray(voxsize, dtype=xp.float32, device=dev),
        img_fwd_nontof
    )

    img_fwd_sum_tof = xp.sum(img_fwd, axis=-1)

    for i in range(nlors):
        assert math.isclose(float(img_fwd_sum_tof[i]), float(img_fwd_nontof[i]), abs_tol=2e-3)
