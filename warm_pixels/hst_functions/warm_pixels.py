"""Primary and plotting functions for hst_warm_pixels.py"""
import logging

import numpy as np

from warm_pixels import hst_utilities as ut
from warm_pixels.pixel_lines import PixelLineCollection

logger = logging.getLogger(
    __name__
)


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
