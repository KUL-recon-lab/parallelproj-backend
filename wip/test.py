import math

img_dim = (250, 270, 280)
voxsize = (2, 2, 2)

xstart = (300, 2, -1)
xend = (-300, -3, 4)

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

num_tofbins = 28
tofbin_width = 20.0
sigma_tof = 30.0
num_sigmas = 3.5
max_tof_bin_diff = num_sigmas * max(sigma_tof, tofbin_width) / tofbin_width

costheta = voxsize[direction] / cf


xm = 0.5 * (xstart[direction] + xend[direction])
sign = 1 if xend[direction] >= xstart[direction] else -1

# the tof bin centers (in world coordinates) are at it*a_tof + b_tof for it in range(num_tofbins)
b_tof = xm - sign * (num_tofbins / 2 - 0.5) * (tofbin_width * costheta)
a_tof = sign * (tofbin_width * costheta)


# %%
# step through the volume plane by plane

assert direction == 0

dr = dr_sq[direction] ** 0.5

#### ONLY VALID FOR direction == 0 ####
a1 = (d1 * voxsize[direction]) / (voxsize[1] * dr)
b1 = (
    xstart[1] - img_origin[1] + d1 * (img_origin[direction] - xstart[direction]) / dr
) / voxsize[1]

a2 = (d2 * voxsize[direction]) / (voxsize[2] * dr)
b2 = (
    xstart[2] - img_origin[2] + d2 * (img_origin[direction] - xstart[direction]) / dr
) / voxsize[2]
#### ONLY VALID FOR direction == 0 ####


istart = 0
iend = img_dim[direction] - 1

i1_f = istart * a1 + b1
i2_f = istart * a2 + b2

for i in range(istart, iend + 1):
    # print(f"{i0:03}, {i1_f:7.2f}, {i2_f:7.2f}, {x0_f:7.2f}, {x1_f:7.2f}, {x2_f:7.2f}")
    # p[i] += bilinear_interp_fixed0(img, n0, n1, n2, i0, i1_f, i2_f);

    i1_f += a1
    i2_f += a2

    x_dir = i * voxsize[direction] + img_origin[direction]
    it_f = (x_dir - b_tof) / a_tof

    # min and max tof bin for which we have to calculate tof weights
    it_min = math.floor(it_f - max_tof_bin_diff)
    if it_min < 0:
        it_min = 0
    if it_min > (num_tofbins - 1):
        it_min = num_tofbins - 1

    it_max = math.ceil(it_f + max_tof_bin_diff)
    if it_max < 0:
        it_max = 0
    if it_max > (num_tofbins - 1):
        it_max = num_tofbins - 1

    print(i, x_dir, it_f, it_min, it_max)
