#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray


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


def ray_cube_intersection(
    xstart: NDArray[np.float64],
    xend: NDArray[np.float64],
    img_origin: NDArray[np.float64],
    voxsize: NDArray[np.float64],
    img_dim: NDArray[np.int32],
) -> tuple[int, float, int, int]:
    """ray cube intersection function for Joseph projector

    Parameters
    ----------
    xstart : NDArray[np.float64]
        3 element array with the starting point of the ray
    xend : NDArray[np.float64]
        3 element array with the end point of the ray
    img_origin : NDArray[np.float64]
        3 element array with the origin of the image volume (world coordinates of 0,0,0 voxel)
    voxsize : NDArray[np.float64]
        3 element array with the size of the voxels in world coordinates
    img_dim : NDArray[np.int32]
        3 element array with the dimensions of the image volume

    Returns
    -------
    tuple[int, float, int, int]
        direction of the ray (0, 1, or 2), correction factor, start plane, end plane
    """

    # get the bounding box of the image volume (outside of all voxels)
    boxmin = img_origin - 0.5 * voxsize
    boxmax = boxmin + img_dim * voxsize

    # get the direction and inverse direction of the ray
    dr = xend - xstart
    inv_dr = 1 / dr

    # check if the ray intersects the bounding box
    tmin = 0.0
    tmax = 1.0

    for i in range(3):
        t1 = (boxmin[i] - xstart[i]) * inv_dr[i]
        t2 = (boxmax[i] - xstart[i]) * inv_dr[i]

        tmin = max(tmin, min(t1, t2))
        tmax = min(tmax, max(t1, t2))

    # if tmin < tmax, then the ray intersects the bounding box
    # we do a few additional calculations to things that are useful for the joseph projector
    dr_sq = dr**2
    cos_sq = dr_sq / np.sum(dr_sq)

    # principal axis of the ray
    direction: int = int(np.argmax(cos_sq))
    # correction factor that accounts for the voxel size and obliqueness of the ray
    correction_factor: float = float(voxsize[direction]) / float(
        np.sqrt(cos_sq[direction])
    )

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

    show_plots = False
    verbose = False

    voxsize = np.array([2.0, 1.0, 4.0])
    img_dim = np.array([6, 12, 3])
    img_origin = -voxsize * img_dim / 2 + 0.5 * voxsize + 1

    test_cases = [
        (
            np.array([-14.0, 0.0, 0.0]),  # xstart
            np.array([14.0, 0.0, 0.0]),  # xend
            0,  # expected_start_plane
            5,  # expected_end_plane
            0,  # expected_direction
        ),
        (
            np.array([14.0, 0.0, 0.0]),  # xstart
            np.array([-14.0, 0.0, 0.0]),  # xend
            0,  # expected_start_plane
            5,  # expected_end_plane
            0,  # expected_direction
        ),
        (
            np.array([-14.0, 0.0, 0.0]),  # xstart
            np.array([2.0, 0.0, 0.0]),  # xend
            0,  # expected_start_plane
            3,  # expected_end_plane
            0,  # expected_direction
        ),
        (
            np.array([2.0, 0.0, 0.0]),  # xstart
            np.array([14.0, 0.0, 0.0]),  # xend
            4,  # expected_start_plane
            5,  # expected_end_plane
            0,  # expected_direction
        ),
        (
            np.array([0.0, -14.0, 0.0]),  # xstart
            np.array([0.0, 14.0, 0.0]),  # xend
            0,  # expected_start_plane
            11,  # expected_end_plane
            1,  # expected_direction
        ),
        (
            np.array([0.0, 14.0, 0.0]),  # xstart
            np.array([0.0, -14.0, 0.0]),  # xend
            0,  # expected_start_plane
            11,  # expected_end_plane
            1,  # expected_direction
        ),
        (
            np.array([0.0, 0.0, -14.0]),  # xstart
            np.array([0.0, 0.0, 14.0]),  # xend
            0,  # expected_start_plane
            2,  # expected_end_plane
            2,  # expected_direction
        ),
        (
            np.array([0.0, 0.0, -14.0]),  # xstart
            np.array([0.0, 0.0, 1.0]),  # xend
            0,  # expected_start_plane
            1,  # expected_end_plane
            2,  # expected_direction
        ),
        (
            np.array([0.0, 0.0, 1.0]),  # xstart
            np.array([0.0, 0.0, 14.0]),  # xend
            2,  # expected_start_plane
            2,  # expected_end_plane
            2,  # expected_direction
        ),
        (
            np.array([-14.0, -14.0, -14.0]),  # xstart
            np.array([0.0, 0.0, 1.0]),  # xend
            0,  # expected_start_plane
            1,  # expected_end_plane
            2,  # expected_direction
        ),
        (
            np.array([-15.1, -14.0, -14.0]),  # xstart
            np.array([0.0, 0.0, 1.0]),  # xend
            0,  # expected_start_plane
            2,  # expected_end_plane
            0,  # expected_direction
        ),
        (
            np.array([0.0, 0.0, 1.0]),  # xstart
            np.array([-15.1, -14.0, -14.0]),  # xend
            0,  # expected_start_plane
            2,  # expected_end_plane
            0,  # expected_direction
        ),
        (
            np.array([-4.5, 7.0, 0.0]),  # xstart # ray touching the cube, but excluded
            np.array([6.5, 7.0, 0.0]),  # xend
            -1,  # expected_start_plane
            -1,  # expected_end_plane
            0,  # expected_direction
        ),
        (
            np.array([-4.5, -5.0, 0.0]),  # xstart # ray touching the cube, but included
            np.array([6.5, -5.0, 0.0]),  # xend
            0,  # expected_start_plane
            5,  # expected_end_plane
            0,  # expected_direction
        ),
    ]

    for (
        xstart,
        xend,
        expected_start_plane,
        expected_end_plane,
        expected_direction,
    ) in test_cases:

        # print xstart, xend
        direction, cf, start_plane, end_plane = ray_cube_intersection(
            xstart, xend, img_origin, voxsize, img_dim
        )

        assert start_plane == expected_start_plane
        assert end_plane == expected_end_plane
        assert direction == expected_direction

        d = xend - xstart
        assert cf == float(
            voxsize[direction] / (np.abs(d[direction]) / np.linalg.norm(d))
        )

        if verbose:
            print("xstart:", xstart)
            print("xend:", xend)
            print("direction:", direction)
            for i in range(img_dim[direction]):
                pos = img_origin[direction] + i * voxsize[direction]
                print(f"plane position {i}: {pos}")

            print("start_plane:", start_plane)
            print("end_plane:", end_plane)
            print()

        if show_plots:
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
                [xstart[0], xend[0]],
                [xstart[1], xend[1]],
                [xstart[2], xend[2]],
                color="red",
            )

            ax.scatter(xstart[0], xstart[1], xstart[2], color="red", marker="o")
            ax.scatter(xend[0], xend[1], xend[2], color="red", marker="x")

            # set equal aspect ratio
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            ax.set_zlim(-15, 15)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            q = np.linspace(-10, 10, 2)
            Q1, Q2 = np.meshgrid(q, q)

            # draw all planes along the principal axis
            for i in range(img_dim[direction]):
                pos = img_origin[direction] + i * voxsize[direction]
                if i == start_plane:
                    col = "blue"
                elif i == end_plane:
                    col = "orange"
                else:
                    col = "green"
                draw_plane_bbox(
                    ax, direction=direction, position=pos, bound=(-10, 10), color=col
                )

            fig.show()
