import math

import parallelproj_backend as ppb

if ppb.PARALLELPROJ_CUDA:
    import cupy as xp
else:
    import numpy as xp
from scipy.special import erf


def effective_tof_kernel(dx: float, sigma_t: float, tbin_width: float) -> float:
    """Gaussian integrated over a tof bin width."""
    sqrt2 = math.sqrt(2.0)
    return 0.5 * (
        erf((dx + 0.5 * tbin_width) / (sqrt2 * sigma_t))
        - erf((dx - 0.5 * tbin_width) / (sqrt2 * sigma_t))
    )


img_dim = (19, 1, 1)
voxsize = (2, 2, 2)

xstart = (-60, 0, 0)
xend = (60, 0, 0)

tofbin_width: float = 3.0
sigma_tof: float = 4.0
num_sigmas: float = 3.0
tof_center_offset: float = 0.0

show_fig = True

num_tofbins: int | None = None

###########################################

if num_tofbins is None:
    ray_length = math.sqrt(
        (xend[0] - xstart[0]) ** 2
        + (xend[1] - xstart[1]) ** 2
        + (xend[2] - xstart[2]) ** 2
    )
    num_tofbins = math.ceil(ray_length / tofbin_width)

    # ensure num_tofbins is odd
    if num_tofbins % 2 == 0:
        num_tofbins += 1


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


for i in range(istart, iend + 1):
    # min and max tof bin for which we have to calculate tof weights
    it_min = math.floor(it_f - max_tof_bin_diff)
    it_max = math.ceil(it_f + max_tof_bin_diff)

    tof_weights = xp.zeros(it_max + 1 - it_min)
    for k, it in enumerate(range(it_min, it_max + 1)):
        dist = abs(it_f - it) * tofbin_width
        tof_weights[k] = effective_tof_kernel(dist, sigma_tof, tofbin_width)

    tof_weights /= tof_weights.sum()

    trunc_tof_sum = 0.0
    for k, it in enumerate(range(it_min, it_max + 1)):
        if it >= 0 and it < num_tofbins:
            trunc_tof_sum += tof_weights[k]

    ################################ diagnostics
    x_dir = i * voxsize[direction] + img_origin[direction]
    # print(i, num_tofbins, x_dir, it_f, it_min, it_max, tof_weights.sum(), trunc_tof_sum)

    ################################ diagnostics

    it_f += at

# %%
img = xp.zeros(img_dim, dtype=xp.float32)
img[img_dim[0] // 2, 0, 0] = 1.0

p_nontof = xp.zeros(1, dtype=xp.float32)
p_tof = xp.zeros((1, num_tofbins), dtype=xp.float32)

ppb.joseph3d_fwd(
    xp.array([xstart], dtype=xp.float32),
    xp.array([xend], dtype=xp.float32),
    img,
    xp.array(img_origin, dtype=xp.float32),
    xp.array(voxsize, dtype=xp.float32),
    p_nontof,
)

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

# %%
# backprojection

q_tof = xp.zeros(p_tof.shape, dtype=xp.float32)
q_tof[0, num_tofbins // 2] = 1.0


img_back_tof = xp.zeros(img_dim, dtype=xp.float32)
ppb.joseph3d_tof_sino_back(
    xp.array([xstart], dtype=xp.float32),
    xp.array([xend], dtype=xp.float32),
    img_back_tof,
    xp.array(img_origin, dtype=xp.float32),
    xp.array(voxsize, dtype=xp.float32),
    q_tof,
    tofbin_width,
    xp.array([sigma_tof], dtype=xp.float32),
    xp.array([tof_center_offset], dtype=xp.float32),
    num_tofbins,
    n_sigmas=num_sigmas,
)
