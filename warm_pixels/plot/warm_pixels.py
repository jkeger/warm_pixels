import logging
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from warm_pixels import hst_utilities as ut
from warm_pixels import misc
from warm_pixels.misc import plot_hist
from warm_pixels.model.quadrant import Quadrant
from warm_pixels.pixel_lines import PixelLineCollection

logger = logging.getLogger(
    __name__
)


def plot_warm_pixels(image_quadrant, save_path=None):
    """Plot an image and mark the locations of warm pixels.

    stack_dataset_warm_pixels() must first be run for the dataset.

    Parameters
    ----------
    image_quadrant
        One quadrant of an image

    save_path : str (opt.)
        The file path for saving the figure. If None, then show the figure.
    """
    # Plot the image and the found warm pixels
    plt.figure()

    image = image_quadrant.array()
    warm_pixels = PixelLineCollection(
        image_quadrant.warm_pixels(),
    )

    im = plt.imshow(
        X=image,
        aspect="equal",
        vmin=0,
        vmax=500
    )
    try:
        plt.scatter(
            warm_pixels.locations[:, 1] + 0.5,
            warm_pixels.locations[:, 0] + 0.5,
            marker=".",
            c="r",
            edgecolor="none",
            s=0.1,
            alpha=0.7,
        )
    except Exception as e:
        logger.exception(e)

    plt.colorbar(im)
    plt.axis("off")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=500)
        plt.close()


def plot_warm_pixel_distributions(quadrants: List[Quadrant], save_path=None):
    """Plot histograms of the properties of premade warm pixel trails.

    find_dataset_warm_pixels() and find_consistent_warm_pixels() must first be
    run for the dataset.

    Parameters
    ----------
    quadrants
        A list of quadrants (A, B, C, D) from the image to plot..

    save_path : str (opt.)
        The file path for saving the figure. If None, then show the figure.
    """
    # Tile four histograms
    plt.figure()
    gs = GridSpec(nrows=2, ncols=2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])

    if len(quadrants) > 1:
        colours = misc.A1_c
    else:
        colours = ["k"]

    # Load
    warm_pixels = sum(quadrant.consistent_lines() for quadrant in quadrants)

    # Set bins for all quadrants
    n_row_bins = 15
    n_flux_bins = 30
    n_background_bins = 10
    n_date_bins = 10

    row_min = np.amin(warm_pixels.locations[:, 0])
    row_max = np.amax(warm_pixels.locations[:, 0])
    row_bins = np.linspace(row_min, row_max, n_row_bins + 1)

    flux_min = np.amin(warm_pixels.fluxes)
    flux_max = np.amax(warm_pixels.fluxes)
    flux_bins = np.logspace(np.log10(flux_min), np.log10(flux_max), n_flux_bins + 1)

    background_min = np.amin(warm_pixels.backgrounds)
    background_max = np.amax(warm_pixels.backgrounds)
    background_bins = np.linspace(background_min, background_max, n_background_bins + 1)

    date_min = np.amin(warm_pixels.dates - ut.date_acs_launch)
    date_max = np.amax(warm_pixels.dates - ut.date_acs_launch)
    date_bins = np.linspace(date_min, date_max, n_date_bins + 1)

    # Plot each quadrant separately
    for quadrant, c in zip(
            quadrants,
            colours
    ):
        warm_pixels = quadrant.consistent_lines()
        # Data
        row_hist, row_bin_edges = np.histogram(
            warm_pixels.locations[:, 0], bins=row_bins
        )
        flux_hist, flux_bin_edges = np.histogram(warm_pixels.fluxes, bins=flux_bins)
        background_hist, background_bin_edges = np.histogram(
            warm_pixels.backgrounds, bins=background_bins
        )
        date_hist, date_bin_edges = np.histogram(
            warm_pixels.dates - ut.date_acs_launch, bins=date_bins
        )

        # Plot
        plot_hist(ax1, row_hist, row_bin_edges, c=c)
        plot_hist(ax2, flux_hist, flux_bin_edges, c=c, label=quadrant)
        plot_hist(ax3, background_hist, background_bin_edges, c=c)
        plot_hist(ax4, date_hist, date_bin_edges, c=c)

    ax2.legend(fontsize=12)

    # Axes
    ax1.set_xlabel("Row")
    ax2.set_xlabel(r"Flux (e$^-$)")
    ax3.set_xlabel(r"Background (e$^-$)")
    ax4.set_xlabel("Days Since ACS Launch")
    ax1.set_ylabel("Number of Warm Pixels")
    ax3.set_ylabel("Number of Warm Pixels")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax3.ticklabel_format(useOffset=False, axis="x")
    ax4.ticklabel_format(useOffset=False, axis="x")

    misc.nice_plot(ax1)
    misc.nice_plot(ax2)
    misc.nice_plot(ax3)
    misc.nice_plot(ax4)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()
        print("Saved", save_path.stem)
