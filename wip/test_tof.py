import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


def effective_tof_kernel(dx: float, sigma_t: float, tbin_width: float) -> float:
    """Gaussian integrated over a tof bin width."""
    sqrt2 = math.sqrt(2.0)
    return 0.5 * (
        erf((dx + 0.5 * tbin_width) / (sqrt2 * sigma_t))
        - erf((dx - 0.5 * tbin_width) / (sqrt2 * sigma_t))
    )


img_dim = (5, 5, 5)
voxsize = (2, 2, 2)

xstart = (0, 0, 50)
xend = (0, 0, -50)

tofbin_width: float = 2.0
sigma_tof: float = 8.0
num_sigmas: float = 3.0
tof_center_offset: float = 0.0

show_fig = False

num_tofbins: int | None = None

###########################################

if num_tofbins is None:
    ray_length = math.sqrt(
        (xend[0] - xstart[0]) ** 2
        + (xend[1] - xstart[1]) ** 2
        + (xend[2] - xstart[2]) ** 2
    )
    num_tofbins = math.ceil(ray_length / tofbin_width)


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


# %%
# step through the volume plane by plane

# assert direction == 0
#
# dr = dr_sq[direction] ** 0.5
#
##### ONLY VALID FOR direction == 0 ####
# a1 = (d1 * voxsize[direction]) / (voxsize[1] * dr)
# b1 = (
#    xstart[1] - img_origin[1] + d1 * (img_origin[direction] - xstart[direction]) / dr
# ) / voxsize[1]
#
# a2 = (d2 * voxsize[direction]) / (voxsize[2] * dr)
# b2 = (
#    xstart[2] - img_origin[2] + d2 * (img_origin[direction] - xstart[direction]) / dr
# ) / voxsize[2]
##### ONLY VALID FOR direction == 0 ####
#
# istart = 0
# iend = img_dim[direction] - 1
#
# i1_f = istart * a1 + b1
# i2_f = istart * a2 + b2

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
    - sign * (num_tofbins / 2 - 0.5) * (tofbin_width * costheta)
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
    # print(f"{i0:03}, {i1_f:7.2f}, {i2_f:7.2f}, {x0_f:7.2f}, {x1_f:7.2f}, {x2_f:7.2f}")
    # p[i] += bilinear_interp_fixed0(img, n0, n1, n2, i0, i1_f, i2_f);

    # min and max tof bin for which we have to calculate tof weights
    it_min = math.floor(it_f - max_tof_bin_diff)
    it_max = math.ceil(it_f + max_tof_bin_diff)

    tof_weights = np.zeros(it_max + 1 - it_min)
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
    print(i, num_tofbins, x_dir, it_f, it_min, it_max, tof_weights.sum(), trunc_tof_sum)

    ################################ diagnostics

    it_f += at

# %%
import parallelproj_backend as ppb

img = np.zeros(img_dim, dtype=np.float32)
img[2, 2, 2] = 1.0

p_nontof = np.zeros(1, dtype=np.float32)
p_tof = np.zeros((1, num_tofbins), dtype=np.float32)

ppb.joseph3d_fwd(
    np.array([xstart], dtype=np.float32),
    np.array([xend], dtype=np.float32),
    img,
    np.array(img_origin, dtype=np.float32),
    np.array(voxsize, dtype=np.float32),
    p_nontof,
)

ppb.joseph3d_tof_sino_fwd(
    np.array([xstart], dtype=np.float32),
    np.array([xend], dtype=np.float32),
    img,
    np.array(img_origin, dtype=np.float32),
    np.array(voxsize, dtype=np.float32),
    p_tof,
    tofbin_width,
    np.array([sigma_tof], dtype=np.float32),
    np.array([tof_center_offset], dtype=np.float32),
    num_tofbins,
    n_sigmas=num_sigmas,
)

# %%
if show_fig:
    fig = plt.figure(figsize=(8, 8), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")

    ax.plot([xstart[0], xend[0]], [xstart[1], xend[1]], [xstart[2], xend[2]], "r-")
    ax.plot([xstart[0], xend[0]], [xstart[1], xstart[1]], [xstart[2], xstart[2]], "b-")

    ax.scatter(xstart[0], xstart[1], xstart[2], c="r", marker="x")
    ax.scatter(xend[0], xend[1], xend[2], c="r")
    ax.scatter(xend[0], xstart[1], xstart[2], c="b")

    ax.scatter(img_origin[0], xstart[1], xstart[2], marker=".", c="k")
    ax.scatter(
        img_origin[0] + voxsize[0] * (img_dim[0] - 1),
        xstart[1],
        xstart[2],
        marker=".",
        c="k",
    )

    ax.scatter(
        [i * tof_slope + tof_origin for i in range(num_tofbins)],
        num_tofbins * [xstart[1]],
        num_tofbins * [xstart[2]],
        marker="x",
    )

    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_zlim(-60, 60)

    fig2, ax2 = plt.subplots(figsize=(8, 6), layout="constrained")
    ax2.plot(tof_weights, "o-")

    plt.show()
