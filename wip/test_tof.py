import parallelproj_backend as ppb
import math
import random

if ppb.PARALLELPROJ_CUDA:
    import cupy as xp
else:
    import numpy as xp


def effective_tof_kernel(dx: float, sigma_t: float, tbin_width: float) -> float:
    """Gaussian integrated over a tof bin width."""
    sqrt2 = 1.414213562373095
    return 0.5 * (
        math.erf((dx + 0.5 * tbin_width) / (sqrt2 * sigma_t))
        - math.erf((dx - 0.5 * tbin_width) / (sqrt2 * sigma_t))
    )


def test_tof_sino_fwd(
    voxsize: tuple[float, float, float] = (2.2, 2.5, 2.7),
    vox_num: int = 7,
    tofbin_width: float = 3.0,
    sigma_tof: float = 4.5,
    num_sigmas: float = 3.0,
    tof_center_offset: float = 1.0,
    num_tofbins: int = 41,
    nvox: int = 19,
    direc=2,
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

    img = xp.zeros(img_dim, dtype=xp.float32)
    img.ravel()[vox_num] = 1.0
    p_tof_ref = xp.zeros((1, num_tofbins), dtype=xp.float32)

    for i in range(istart, iend + 1):
        # min and max tof bin for which we have to calculate tof weights
        it_min = math.floor(it_f - max_tof_bin_diff)
        it_max = math.ceil(it_f + max_tof_bin_diff)

        tof_weights = xp.zeros(it_max + 1 - it_min)
        for k, it in enumerate(range(it_min, it_max + 1)):
            dist = abs(it_f - it) * tofbin_width
            tof_weights[k] = effective_tof_kernel(dist, sigma_tof, tofbin_width)

        tof_weights /= tof_weights.sum()

        for k, it in enumerate(range(it_min, it_max + 1)):
            p_tof_ref[0, it] += tof_weights[k] * cf * float(img.ravel()[i])

        it_f += at

    # %%

    # parallelproj-backend based sinogram TOF forward projection

    p_tof = xp.zeros((1, num_tofbins), dtype=xp.float32)
    ppb.joseph3d_tof_sino_fwd(
        xp.array([xstart], dtype=xp.float32),
        xp.array([xend], dtype=xp.float32),
        img,
        xp.array(img_origin, dtype=xp.float32),
        xp.array(voxsize, dtype=xp.float32),
        p_tof,
        tofbin_width,
        xp.array([sigma_tof], dtype=xp.float32),
        xp.array([tof_center_offset], dtype=xp.float32),
        num_tofbins,
        n_sigmas=num_sigmas,
    )

    if verbose:
        for i in range(num_tofbins):
            print(
                f"bin {i:2d}: proj_ref = {p_tof_ref[0,i]:.5E}, proj_ppb = {p_tof[0,i]:.5E}"
            )

    # check whether the projection is equal to the expected one
    assert xp.all(xp.isclose(p_tof_ref, p_tof, atol=1e-6))

    # since we are forward projecting an image of a single voxel containg a value of 1
    # the sum over TOF should be the voxel size (if we have enough TOF bins)
    assert xp.isclose(float(xp.sum(p_tof)), voxsize[direction], atol=1e-6)

def test_tof_sino_adjointness(
    voxsize: tuple[float, float, float] = (2.2, 2.5, 2.7),
    tofbin_width: float = 3.0,
    sigma_tof: float = 4.5,
    num_sigmas: float = 3.0,
    tof_center_offset: float = 0.0,
    num_tofbins: int = 41,
    nvox: int = 19,
    verbose: bool = False,
    nlors = 500):

    #------
    random.seed(42)
    img_dim = (nvox, nvox, nvox)

    n0, n1, n2 = img_dim
    img_origin = (
        (-(n0 / 2) + 0.5) * voxsize[0],
        (-(n1 / 2) + 0.5) * voxsize[1],
        (-(n2 / 2) + 0.5) * voxsize[2],
    )


    img = xp.zeros(img_dim, dtype=xp.float32)
    # fill the image with uniform random values using python's random module
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                img[i, j, k] = random.uniform(0.0, 1.0)

    xstart = xp.zeros((nlors, 3), dtype=xp.float32)
    xend = xp.zeros((nlors, 3), dtype=xp.float32)

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
    sigma_tof_array = xp.zeros(nlors, dtype=xp.float32)
    for i in range(nlors):
        sigma_tof_array[i] = sigma_tof * random.uniform(0.9,1.1)

    tof_center_offset_array = xp.zeros(nlors, dtype=xp.float32)
    for i in range(nlors):
        tof_center_offset_array[i] = tof_center_offset + random.uniform(-2.0,2.0)

    img_fwd = xp.zeros((nlors, num_tofbins), dtype=xp.float32)
    ppb.joseph3d_tof_sino_fwd(
        xstart,
        xend,
        img,
        xp.array(img_origin, dtype=xp.float32),
        xp.array(voxsize, dtype=xp.float32),
        img_fwd,
        tofbin_width,
        sigma_tof_array,
        tof_center_offset_array,
        num_tofbins,
        n_sigmas=num_sigmas,
    )

    # back project a random TOF sinogram
    y = xp.zeros((nlors, num_tofbins), dtype=xp.float32)
    for i in range(nlors):
        for j in range(num_tofbins):
            y[i, j] = random.uniform(0.0, 1.0)

    y_back = xp.zeros(img_dim, dtype=xp.float32)
    ppb.joseph3d_tof_sino_back(
        xstart,
        xend,
        y_back,
        xp.array(img_origin, dtype=xp.float32),
        xp.array(voxsize, dtype=xp.float32),
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
    assert xp.isclose(innerprod1, innerprod2)

    # do a non-TOF forward projection and check whether the sum over TOF bins equals the non-TOF projection
    img_fwd_nontof = xp.zeros(nlors, dtype=xp.float32)
    ppb.joseph3d_fwd(
        xstart,
        xend,
        img,
        xp.array(img_origin, dtype=xp.float32),
        xp.array(voxsize, dtype=xp.float32),
        img_fwd_nontof
    )

    xp.all(xp.isclose(xp.sum(img_fwd, -1), img_fwd_nontof))
