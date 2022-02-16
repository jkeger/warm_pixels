"""Primary and plotting functions for hst_warm_pixels.py"""
import logging

import numpy as np

from warm_pixels import hst_utilities as ut
from warm_pixels.pixel_lines import PixelLine, PixelLineCollection

logger = logging.getLogger(
    __name__
)


def find_consistent_warm_pixels(dataset, quadrant, flux_min=None, flux_max=None):
    """Find the consistent warm pixels in a dataset.

    find_dataset_warm_pixels() must first be run for the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    quadrant : str (opt.)
        The quadrant (A, B, C, D) of the image to load.

    flux_min, flux_max : float (opt.)
        If provided, then before checking for consistent pixels, discard any
        with fluxes outside of these limits.

    Saves
    -----
    warm_pixels : PixelLineCollection
        The set of consistent warm pixel trails, saved to
        dataset.saved_consistent_lines().
    """
    # Load
    warm_pixels = PixelLineCollection()
    warm_pixels.load(dataset.saved_lines(quadrant))

    # Ignore warm pixels below the minimum flux
    if flux_min is not None:
        sel_flux = np.where(
            (warm_pixels.fluxes > flux_min) & (warm_pixels.fluxes < flux_max)
        )[0]
        print("Kept %d bounded fluxes of %d" % (len(sel_flux), warm_pixels.n_lines))
        warm_pixels.lines = warm_pixels.lines[sel_flux]
        print("    ", end="")

    # Find the warm pixels present in at least e.g. 2/3 of the images
    consistent_lines = warm_pixels.find_consistent_lines(
        fraction_present=ut.fraction_present
    )
    print(
        "Found %d consistents of %d possibles"
        % (len(consistent_lines), warm_pixels.n_lines)
    )

    # Extract the consistent warm pixels
    warm_pixels.lines = warm_pixels.lines[consistent_lines]

    # Save
    warm_pixels.save(dataset.saved_consistent_lines(quadrant))


def stack_dataset_warm_pixels(dataset, quadrants):
    """Stack a set of premade warm pixel trails into bins.

    find_dataset_warm_pixels() and find_consistent_warm_pixels() must first be
    run for the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    quadrants : [str]
        The list of quadrants (A, B, C, D) of the images to load, combined
        together if more than one provided.

    use_corrected : bool (opt.)
        If True, then use the corrected images with CTI removed instead.

    Saves
    -----
    stacked_lines : PixelLineCollection
        The set of stacked pixel trails, saved to dataset.saved_stacked_lines().
    """
    # Load
    warm_pixels = PixelLineCollection()
    # Append data from each quadrant
    for quadrant in quadrants:
        warm_pixels.load(dataset.saved_consistent_lines(quadrant))

    # Subtract preceeding pixels in each line before stacking
    #
    # RJM - global background estimates may not be accurate, so keeping this information at this stage.
    #       It is always possible to do this subtraction after stacking.
    #
    # for i in range(warm_pixels.n_lines):
    #    warm_pixels.lines[i].data[ut.trail_length + 1 :] -= warm_pixels.lines[i].data[
    #        : ut.trail_length
    #    ][::-1]

    # Stack the lines in bins by distance from readout and total flux
    (
        stacked_lines,
        row_bins,
        flux_bins,
        date_bins,
        background_bins,
    ) = warm_pixels.generate_stacked_lines_from_bins(
        n_row_bins=ut.n_row_bins,
        flux_bins=ut.flux_bins,
        n_background_bins=ut.n_background_bins,
        return_bin_info=True,
    )
    print(
        "Stacked lines in %d bins"
        % (ut.n_row_bins * ut.n_flux_bins * ut.n_background_bins)
    )

    # Save
    stacked_lines.save(dataset.saved_stacked_lines(quadrants))
    np.savez(
        dataset.saved_stacked_info(quadrants),
        row_bins,
        flux_bins,
        date_bins,
        background_bins,
    )
