img_dim = (21, 21, 21)
voxsize = (2, 2, 2)

n0, n1, n2 = img_dim

img_origin = (
    (-(n0 / 2) + 0.5) * voxsize[0],
    (-(n1 / 2) + 0.5) * voxsize[1],
    (-(n2 / 2) + 0.5) * voxsize[2],
)

direction = 0

xstart = (-20, 2, -1)
xend = (22, -3, 4)

d0 = xend[0] - xstart[0]
d1 = xend[1] - xstart[1]
d2 = xend[2] - xstart[2]


istart = 0
iend = img_dim[direction] - 1

# %%
# get the correction factor cf

dr_sq = (d0**2, d1**2, d2**2)

sum_sq = sum(dr_sq)
cos_sq = dr_sq[direction] / sum_sq

cf = voxsize[direction] / (cos_sq**0.5)

# %%

num_tof_bins: int = 8
tofbin_width: float = 6.0

# calculate the fraction of a tof bin that we increase when we step forward a plane through the volume
# i_tof is the signed tof bin index
delta_i_tof = cf / tofbin_width

# TODO: calculate "float" tof bin index of start voxel
# this can be used to determine the min/max tofbin to consider
# and also to calculate the distance to a given tof bin which is tof_bin_width x i_diff

# calculate the 3 world coordinates of num_tof_bins points that are along the ray and separated by tofbin_width
# this points should be "symmetric" around the center point between xstart and xend

tof_bins = []
for unsigned_tof_bin in range(num_tof_bins):
    signed_tof_bin = (
        unsigned_tof_bin - (num_tof_bins // 2) + 0.5 * (1 - num_tof_bins % 2)
    )
    print(unsigned_tof_bin, signed_tof_bin)

# %%
# step trought the volume plane by plane

assert direction == 0

dr = d0

a1 = (d1 * voxsize[direction]) / (voxsize[1] * dr)
b1 = (
    xstart[1] - img_origin[1] + d1 * (img_origin[direction] - xstart[direction]) / dr
) / voxsize[1]

a2 = (d2 * voxsize[direction]) / (voxsize[2] * dr)
b2 = (
    xstart[2] - img_origin[2] + d2 * (img_origin[direction] - xstart[direction]) / dr
) / voxsize[2]

i1_f = istart * a1 + b1
i2_f = istart * a2 + b2

# the world coordinates of the current point along the ray
x0_f = istart * voxsize[0] + img_origin[0]
x1_f = i1_f * voxsize[1] + img_origin[1]
x2_f = i2_f * voxsize[2] + img_origin[2]

dx0 = voxsize[0]
dx1 = a1 * voxsize[1]
dx2 = a2 * voxsize[2]

for i0 in range(istart, iend + 1):
    # print(f"{i0:03}, {i1_f:7.2f}, {i2_f:7.2f}, {x0_f:7.2f}, {x1_f:7.2f}, {x2_f:7.2f}")
    # p[i] += bilinear_interp_fixed0(img, n0, n1, n2, i0, i1_f, i2_f);

    i1_f += a1
    i2_f += a2

    x0_f += dx0
    x1_f += dx1
    x2_f += dx2
