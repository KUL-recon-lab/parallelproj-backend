#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def draw_plane_bbox(ax, direction=2, position=0.5, bound=(-10, 10), color="green"):
    """Draws the bounding box (edges) of a 2D plane in 3D space with correct corner order."""
    low, high = bound

    # Fixed axis stays at 'position'; vary the other two in this order to get a rectangle
    # Order: (low, low) → (high, low) → (high, high) → (low, high) → back to (low, low)
    corners = []
    for i, j in [(low, low), (high, low), (high, high), (low, high), (low, low)]:
        point = [0, 0, 0]
        point[direction] = position
        point[(direction + 1) % 3] = i
        point[(direction + 2) % 3] = j
        corners.append(point)

    corners = np.array(corners)
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color=color)


def ray_cube_intersection(xstart, xend, img_origin, voxsize, img_dim):

    boxmin = img_origin - 0.5 * voxsize
    boxmax = boxmin + img_dim * voxsize

    dr = xend - xstart
    inv_dr = 1 / dr

    tmin = 0.0
    tmax = np.inf

    for i in range(3):
        t1 = (boxmin[i] - xstart[i]) * inv_dr[i]
        t2 = (boxmax[i] - xstart[i]) * inv_dr[i]

        tmin = max(tmin, min(t1, t2))
        tmax = min(tmax, max(t1, t2))

    # additional calculations useful for the joseph projector
    dr_sq = dr**2
    cos_sq = dr_sq / np.sum(dr_sq)

    # principal axis of the ray
    direction: int = int(np.argmax(cos_sq))
    # correction factor that accounts for the voxel size and obliqueness of the ray
    correction_factor: float = float(voxsize[direction]) / float(
        np.sqrt(cos_sq[direction])
    )

    if tmax > 1:
        tmax = 1.0

    start_plane = -1
    end_plane = -1

    if tmin < tmax:
        xi1 = xstart + tmin * dr
        xi2 = xstart + tmax * dr

        tmp1: float = (xi1[direction] - img_origin[direction]) / voxsize[direction]
        tmp2: float = (xi2[direction] - img_origin[direction]) / voxsize[direction]

        if int(tmp1) != int(tmp2):
            if tmp1 > tmp2:
                tmp1, tmp2 = tmp2, tmp1

            start_plane: int = int(tmp1 + 1)
            end_plane: int = int(np.floor(tmp2))

    return direction, correction_factor, start_plane, end_plane


if __name__ == "__main__":

    xstart = np.array([1.0, -8.0, 2.0])
    xend = np.array([1.0, 8.0, -2.0])

    voxsize = np.array([2.0, 3.0, 1.0])
    img_dim = np.array([4, 3, 7])
    img_origin = np.array([-3.0, -4.0, -3.5])

    # print xstart, xend
    print("xstart:", xstart)
    print("xend:", xend)

    direction, cf, start_plane, end_plane = ray_cube_intersection(
        xstart, xend, img_origin, voxsize, img_dim
    )

    # print the plane position along the principal axis
    print("direction:", direction)
    for i in range(img_dim[direction]):
        pos = img_origin[direction] + i * voxsize[direction]
        print(f"plane position {i}: {pos}")

    print("start_plane:", start_plane)
    print("end_plane:", end_plane)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # plot the image bounding box
    # Calculate the corners of the bounding box
    boxmin = img_origin - 0.5 * voxsize
    boxmax = boxmin + img_dim * voxsize

    # Define the vertices of the box
    x = [boxmin[0], boxmax[0]]
    y = [boxmin[1], boxmax[1]]
    z = [boxmin[2], boxmax[2]]

    # Plot the edges of the box
    for i in range(2):
        for j in range(2):
            ax.plot([x[0], x[1]], [y[i], y[i]], [z[j], z[j]], color="black")
            ax.plot([x[i], x[i]], [y[0], y[1]], [z[j], z[j]], color="black")
            ax.plot([x[i], x[i]], [y[j], y[j]], [z[0], z[1]], color="black")

    # Plot the ray
    ax.plot(
        [xstart[0], xend[0]], [xstart[1], xend[1]], [xstart[2], xend[2]], color="red"
    )

    ax.scatter(xstart[0], xstart[1], xstart[2], color="red", marker="x")
    ax.scatter(xend[0], xend[1], xend[2], color="red", marker="x")

    # set equal aspect ratio
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    q = np.linspace(-10, 10, 2)
    Q1, Q2 = np.meshgrid(q, q)

    # draw all planes along the principal axis
    for i in range(img_dim[direction]):
        pos = img_origin[direction] + i * voxsize[direction]
        draw_plane_bbox(ax, direction=direction, position=pos, bound=(-10, 10))

    fig.show()
