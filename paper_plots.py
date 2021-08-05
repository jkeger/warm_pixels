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
    plt.savefig(Fp_save)
    print("Saved %s" % Fp_save[-64:])


# ========
# Functions
# ========
def example_image_trails(do_pdf=False):
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
    widths = [1.3, 1, 0.13, 0.07]
    heights = [1, 1]
    gs = fig.add_gridspec(nrows=2, ncols=4, width_ratios=widths, height_ratios=heights)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[:, 3])
    gs.update(wspace=0.01, hspace=0)

    # Zoom regions
    n_row, n_col = image.shape
    col_0 = n_col - 255
    col_1 = col_0 + 200
    row_a_0 = 95
    row_a_1 = row_a_0 + 120
    row_b_0 = n_row - 153
    row_b_1 = row_b_0 + 120

    # Plot the image and zooms
    im1 = ax1.imshow(X=image, aspect="equal", vmin=0, vmax=500)
    im2 = ax2.imshow(
        X=image[row_a_0:row_a_1, col_0:col_1],
        aspect="equal",
        vmin=0,
        vmax=500,
        extent=[col_0, col_1, row_a_1, row_a_0],
    )
    im3 = ax3.imshow(
        X=image[row_b_0:row_b_1, col_0:col_1],
        aspect="equal",
        vmin=0,
        vmax=500,
        extent=[col_0, col_1, row_b_1, row_b_0],
    )

    # Zoom lines
    c_zm = "0.7"
    lw_zm = 1.4
    ax1.plot(
        [col_0, col_1, col_1, col_0, col_0],
        [row_a_0, row_a_0, row_a_1, row_a_1, row_a_0],
        c=c_zm,
        lw=lw_zm,
    )
    ax1.plot(
        [col_0, col_1, col_1, col_0, col_0],
        [row_b_0, row_b_0, row_b_1, row_b_1, row_b_0],
        c=c_zm,
        lw=lw_zm,
    )
    for xyA, xyB, axB in [
        [(col_0, row_a_0), (0, 1), ax2],
        [(col_0, row_a_1), (0, 0), ax2],
        [(col_0, row_b_0), (0, 1), ax3],
        [(col_0, row_b_1), (0, 0), ax3],
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
    for ax in [ax1, ax2, ax3, cax]:
        set_large_ticks(ax)

    # Save
    save_fig("example_image_trails", do_pdf)


# ========
# Main
# ========
if __name__ == "__main__":
    # Parse arguments
    parser = prep_parser()
    args = parser.parse_args()

    # Run functions
    if run("example_image_trails"):
        example_image_trails(args.pdf)
