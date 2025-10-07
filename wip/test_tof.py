import parallelproj_backend as ppb
import numpy as np
import math

if ppb.PARALLELPROJ_CUDA:
    import cupy as xp
else:
    import numpy as xp
from scipy.special import erf


def effective_tof_kernel(dx: float, sigma_t: float, tbin_width: float) -> float:
    """Gaussian integrated over a tof bin width."""
    sqrt2 = 1.414213562373095
    return 0.5 * (
        erf((dx + 0.5 * tbin_width) / (sqrt2 * sigma_t))
        - erf((dx - 0.5 * tbin_width) / (sqrt2 * sigma_t))
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
    p_tof_ref = xp.zeros((1, num_tofbins), dtype=np.float32)

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

    assert xp.all(xp.isclose(p_tof_ref, p_tof, atol=1e-6))


## %%
## backprojection
#
# q_tof = xp.zeros(p_tof.shape, dtype=xp.float32)
# q_tof[0, num_tofbins // 2] = 1.0
#
#
# img_back_tof = xp.zeros(img_dim, dtype=xp.float32)
# ppb.joseph3d_tof_sino_back(
#    xp.array([xstart], dtype=xp.float32),
#    xp.array([xend], dtype=xp.float32),
#    img_back_tof,
#    xp.array(img_origin, dtype=xp.float32),
#    xp.array(voxsize, dtype=xp.float32),
#    q_tof,
#    tofbin_width,
#    xp.array([sigma_tof], dtype=xp.float32),
#    xp.array([tof_center_offset], dtype=xp.float32),
#    num_tofbins,
#    n_sigmas=num_sigmas,
# )
