import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.special import erf

# angle between the ray and the stepping direction (costheta = voxsize[dir] / cf)
costheta = 0.85

# Plot with current parameters
xs = [450.0, 0, 0]
xe = [-250.0, 0, 0]
dir = 0
fwhm_tof = 120.0
tofbin_width = fwhm_tof / 3.0
num_tofbins = int(0.95 * abs(xe[dir] - xs[dir]) / (tofbin_width * costheta)) + 1

sigma_tof = fwhm_tof / (2 * math.sqrt(2 * math.log(2)))
dmax = 3.5 * max(sigma_tof, tofbin_width)

img_origin = -228
voxsize = 2.0
i0 = 110

renormalize = False
# %%
# plot the TOF kernel


def effective_tof_kernel(dx):
    """Gaussian integrated over a tof bin width."""
    return 0.5 * (
        erf((dx + 0.5 * tofbin_width) / (np.sqrt(2) * sigma_tof))
        - erf((dx - 0.5 * tofbin_width) / (np.sqrt(2) * sigma_tof))
    )


# %%
# build the LUT for the effective TOF kernel
d_samples, dd = np.linspace(0, dmax, 100, endpoint=True, retstep=True)
effective_tof_kernel_lut = effective_tof_kernel(d_samples)

# %%
# calculate the word coordinates of the tof bin center (x_center_tofbin)


xm = 0.5 * (xs[dir] + xe[dir])
sign = 1 if xe[dir] >= xs[dir] else -1

b_tof = xm - sign * (num_tofbins / 2 - 0.5) * (tofbin_width * costheta)
a_tof = sign * (tofbin_width * costheta)

# %%
# world coordinate of the voxel center for voxel i0
x_vox = img_origin + i0 * voxsize
i_tof_vox = (x_vox - b_tof) / a_tof

imin = math.ceil(i_tof_vox - dmax / tofbin_width)
imax = math.floor(i_tof_vox + dmax / tofbin_width)

if imin < 0:
    imin = 0

if imax > (num_tofbins - 1):
    imax = num_tofbins - 1

#####
tof_weights = []
######

for it in range(imin, imax + 1):
    # calculate the lookup index for the effective TOF kernel LUT
    i_tof_lut_f = abs(it - i_tof_vox) * tofbin_width / dd
    # set tof weight by using linear interpolation in the LUT
    i_t1 = int(i_tof_lut_f)

    if (i_t1 + 1) >= len(effective_tof_kernel_lut):
        tof_weight = effective_tof_kernel_lut[i_t1]
    else:
        tof_weight = effective_tof_kernel_lut[i_t1] * (
            i_t1 + 1 - i_tof_lut_f
        ) + effective_tof_kernel_lut[i_t1 + 1] * (i_tof_lut_f - i_t1)

    ####
    tof_weights.append(tof_weight)
    ####

print(sum(tof_weights))

# %%

# %%
# visualize the binning
x_center_tofbin = [b_tof + a_tof * k for k in range(num_tofbins)]

fig, ax = plt.subplots(1, 1, figsize=(12, 4), layout="constrained")

offset_b = -sign * (num_tofbins / 2) * tofbin_width * costheta
boundaries = [
    xm + offset_b + sign * k * tofbin_width * costheta for k in range(num_tofbins + 1)
]

span_min = min(xs[dir], xe[dir], boundaries[0])
span_max = max(xs[dir], xe[dir], boundaries[-1])
ax.hlines(0, span_min, span_max, linewidth=2)
for b in boundaries:
    ax.vlines(b, -0.1, 0.1, color="orange")
ax.vlines(xm, -0.1, 0.1, color="black", ls=":")
ax.scatter(x_center_tofbin, [0] * num_tofbins, marker="o", color="orange")
for i, x in enumerate(x_center_tofbin):
    tcol = "black"
    if imin <= i <= imax:
        tcol = "orange"
    plt.text(
        x, -0.05, f"{i:02}", ha="center", va="bottom", color=tcol, fontsize="x-small"
    )
ax.scatter([xs[dir], xe[dir]], [0, 0], marker=".", s=100, color=plt.cm.tab10(0))
ax.plot([x_vox - dmax, x_vox + dmax], [0, 0], color="red", ls="-")


for i, it in enumerate(range(imin, imax + 1)):
    ax.plot([a_tof * it + b_tof], [tof_weights[i]], marker=".", color="black")
    ax.vlines(a_tof * it + b_tof, 0, tof_weights[i], color="black")

ax.vlines(x_vox, -0.02, 0.02, color="red")
ax.vlines(x_vox - dmax, -0.03, 0.03, color="red")
ax.vlines(x_vox + dmax, -0.03, 0.03, color="red")
ax.text(xs[dir], -0.02, "xs", ha="center", va="top")
ax.text(xe[dir], -0.02, "xe", ha="center", va="top")
ax.set_xlabel("Position")
ax.set_title(
    f"FWHM tof {fwhm_tof:.1f} - tofbin width {tofbin_width:.1f} - dmax/sigma_tof {dmax/sigma_tof:.1f}",
    fontsize="medium",
)
ax.grid(ls=":")


fig.show()
