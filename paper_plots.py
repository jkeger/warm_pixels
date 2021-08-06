"""Paper and test plots etc in addition to hst_warm_pixels.py

Parameters
----------
--pdf, -p
    Save as pdf not png.

--run, -r : str (opt.)
    Which function(s) to run, chosen by the function name or a substring of the
    name. Accepts multiple values. Defaults to run all.
"""

import numpy as np
import os
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
import matplotlib.patheffects as path_effects

from pixel_lines import PixelLineCollection
from warm_pixels import find_warm_pixels

from hst_warm_pixels import *
from misc import *

d_07_2020 = Dataset("07_2020")  # 2020/12/03, day 6852, 12 images


# ========
# Utilities
# ========
def prep_parser():
    """Prepare the sys args parser."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--pdf",
        action="store_true",
        default=False,
        required=False,
        help="Save as pdf not png.",
    )

    parser.add_argument(
        "-r",
        "--run",
        nargs="*",
        default=["all"],
        required=False,
        help="Which function(s) to run.",
    )

    return parser


def run(name):
    """Whether to run the function with this name."""
    # Run if either running all or the name (or part of it) was provided
    return args.run == ["all"] or any([run in name for run in args.run])


def save_fig(Fp_save, do_pdf=False):
    """Save a figure and print the file path"""
    if do_pdf:
        Fp_save += ".pdf"
    else:
        Fp_save += ".png"
    plt.savefig(Fp_save, dpi=200)
    print("Saved %s" % Fp_save[-64:])


# ========
# Functions
# ========
def example_image_zooms(do_pdf=False):
    """Example HST ACS image with CTI trails"""

    image_path, quadrant = d_07_2020.path + "jdrwc3fcq_raw.fits", "D"

    # Load the image
    image = aa.acs.ImageACS.from_fits(
        file_path=image_path,
        quadrant_letter=quadrant,
        bias_subtract_via_bias_file=True,
        bias_subtract_via_prescan=True,
    ).native

    # Figure
    fig = plt.figure(figsize=(18, 9), constrained_layout=False)
    widths = [1.3, 1, 0.13, 0.06]
    heights = [1, 1]
    gs = fig.add_gridspec(nrows=2, ncols=4, width_ratios=widths, height_ratios=heights)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[:, 3])
    gs.update(wspace=0.01, hspace=0)

    # Zoom regions
    n_row, n_col = image.shape
    col_0 = n_col - 256
    col_1 = col_0 + 200
    row1_0 = 95
    row1_1 = row1_0 + 120
    row2_0 = n_row - 153
    row2_1 = row2_0 + 120

    # Plot the image and zooms
    vmin, vmax = 0, 450
    im1 = ax1.imshow(X=image, aspect="equal", vmin=vmin, vmax=vmax)
    im2 = ax2.imshow(
        X=image[row1_0:row1_1, col_0:col_1],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        extent=[col_0, col_1, row1_1, row1_0],
    )
    im3 = ax3.imshow(
        X=image[row2_0:row2_1, col_0:col_1],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        extent=[col_0, col_1, row2_1, row2_0],
    )

    # Zoom lines
    c_zm = "0.7"
    lw_zm = 1.4
    ax1.plot(
        [col_0, col_1, col_1, col_0, col_0],
        [row1_0, row1_0, row1_1, row1_1, row1_0],
        c=c_zm,
        lw=lw_zm,
    )
    ax1.plot(
        [col_0, col_1, col_1, col_0, col_0],
        [row2_0, row2_0, row2_1, row2_1, row2_0],
        c=c_zm,
        lw=lw_zm,
    )
    for xyA, xyB, axB in [
        [(col_0, row1_0), (0, 1), ax2],
        [(col_0, row1_1), (0, 0), ax2],
        [(col_0, row2_0), (0, 1), ax3],
        [(col_0, row2_1), (0, 0), ax3],
    ]:
        ax1.add_artist(
            ConnectionPatch(
                xyA=xyA,
                xyB=xyB,
                coordsA=ax1.transData,
                coordsB=axB.transAxes,
                color=c_zm,
                lw=lw_zm,
            )
        )

    # Axes etc
    cbar = plt.colorbar(im1, cax=cax, extend="max")
    cbar.set_label(r"Flux (e$^-$)")
    ax2.xaxis.set_visible(False)
    ax2.yaxis.tick_right()
    ax3.yaxis.tick_right()
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    for ax in [ax1.xaxis, ax1.yaxis]:
        ax.set_major_locator(mpl.ticker.MultipleLocator(base=500))
        ax.set_minor_locator(mpl.ticker.MultipleLocator(base=100))
    for ax in [ax2.xaxis, ax2.yaxis, ax3.xaxis, ax3.yaxis]:
        ax.set_major_locator(mpl.ticker.MultipleLocator(base=50))
        ax.set_minor_locator(mpl.ticker.MultipleLocator(base=10))
    cbar.ax.minorticks_on()
    cbar.ax.yaxis.set_ticks(np.arange(vmin, vmax + 1, 100))
    cbar.ax.yaxis.set_ticks(np.arange(50, vmax + 1, 100), minor=True)
    for ax in [ax1, ax2, ax3, cax]:
        set_large_ticks(ax)

    # Save
    save_fig("example_image_zooms", do_pdf)


def found_warm_pixels(do_pdf=False):
    """Example HST ACS image with identified warm pixels"""

    image_path, quadrant = d_07_2020.path + "jdrwc3fcq_raw.fits", "D"

    # Load the image
    image = aa.acs.ImageACS.from_fits(
        file_path=image_path,
        quadrant_letter=quadrant,
        bias_subtract_via_bias_file=True,
        bias_subtract_via_prescan=True,
    ).native

    # Load warm pixels
    poss_warm_pixels = PixelLineCollection()
    poss_warm_pixels.load(d_07_2020.saved_lines(quadrant))
    warm_pixels = PixelLineCollection()
    warm_pixels.load(d_07_2020.saved_consistent_lines(quadrant))

    # Figure
    fig = plt.figure(figsize=(10, 9), constrained_layout=False)
    widths = [1, 0.04]
    heights = [1, 1]
    gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=widths, height_ratios=heights)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[:, 1])
    gs.update(wspace=-0.2, hspace=0.02)

    # Zoom regions
    n_row, n_col = image.shape
    col_0 = n_col - int(256 * 3 / 4) + 1
    col_1 = col_0 + int(200 / 3)
    row1_0 = 96
    row1_1 = row1_0 + int(120 / 3)
    row2_0 = n_row - 154
    row2_1 = row2_0 + int(120 / 3)

    # Plot the image and zooms
    vmin, vmax = 0, 450
    im1 = ax1.imshow(
        X=image[row1_0:row1_1, col_0:col_1],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        extent=[col_0, col_1, row1_1, row1_0],
    )
    im2 = ax2.imshow(
        X=image[row2_0:row2_1, col_0:col_1],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        extent=[col_0, col_1, row2_1, row2_0],
    )

    # Warm pixels
    for wp, do_trail in [[poss_warm_pixels, False], [warm_pixels, True]]:
        # Select warm pixels in the regions and above the minimum flux
        rows = wp.locations[:, 0]
        cols = wp.locations[:, 1]
        flux_min = ut.flux_bins[0]
        sel1 = np.where(
            (cols > col_0)
            & (cols < col_1)
            & (rows > row1_0)
            & (rows < row1_1)
            & (wp.fluxes > flux_min)
        )[0]
        sel2 = np.where(
            (cols > col_0)
            & (cols < col_1)
            & (rows > row2_0)
            & (rows < row2_1)
            & (wp.fluxes > flux_min)
        )[0]

        if do_trail:
            c = "#ff1100"
            zorder = 99
        else:
            c = "#ee7711"
            zorder = 88

        # Plot the warm pixels (and trails)
        for ax, sel in [[ax1, sel1], [ax2, sel2]]:
            for i in sel:
                # Outline warm pixel
                c_0 = cols[i] - 0.03
                c_1 = cols[i] + 1.03
                r_0 = rows[i] - 0.03
                r_1 = rows[i] + 1.03
                ax.plot(
                    [c_0, c_1, c_1, c_0, c_0],
                    [r_0, r_0, r_1, r_1, r_0],
                    c=c,
                    lw=0.7,
                    path_effects=[
                        path_effects.Stroke(linewidth=1.4, foreground="k"),
                        path_effects.Normal(),
                    ],
                    zorder=zorder,
                )

                if do_trail:
                    # Outline trail, rounded edges
                    r_0 += 1
                    r_1 += ut.trail_length
                    ax.plot(
                        [c_0, c_0, c_1, c_1],
                        [r_0, r_1, r_1, r_0],
                        c="w",
                        lw=0.7,
                        path_effects=[
                            path_effects.Stroke(linewidth=1.4, foreground="k"),
                            path_effects.Normal(),
                        ],
                    )

    # Axes etc
    cbar = plt.colorbar(im1, cax=cax, extend="max")
    cbar.set_label(r"Flux (e$^-$)")
    ax1.xaxis.set_visible(False)
    ax2.set_xlabel("Column")
    ax1.set_ylabel("Row")
    ax2.set_ylabel("Row")
    ax1.set_xlim(col_0, col_1)
    ax2.set_xlim(col_0, col_1)
    ax1.set_ylim(row1_1, row1_0)
    ax2.set_ylim(row2_1, row2_0)
    for ax in [ax1.xaxis, ax1.yaxis, ax2.xaxis, ax2.yaxis]:
        ax.set_major_locator(mpl.ticker.MultipleLocator(base=10))
        ax.set_minor_locator(mpl.ticker.MultipleLocator(base=5))
    cbar.ax.minorticks_on()
    cbar.ax.yaxis.set_ticks(np.arange(vmin, vmax + 1, 100))
    cbar.ax.yaxis.set_ticks(np.arange(50, vmax + 1, 100), minor=True)
    for ax in [ax1, ax2, cax]:
        set_large_ticks(ax)

    # Save
    save_fig("found_warm_pixels", do_pdf)


# ========
# Main
# ========
if __name__ == "__main__":
    # Parse arguments
    parser = prep_parser()
    args = parser.parse_args()

    # Run functions
    if run("example_image_zooms"):
        example_image_zooms(args.pdf)
    if run("found_warm_pixels"):
        found_warm_pixels(args.pdf)
