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


def extract_consistent_warm_pixels_corrected(dataset, quadrant):
    """Extract the corresponding warm pixels from the corrected images with CTI
    removed, in the same locations as the orignal consistent warm pixels.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    quadrant : str (opt.)
        The quadrant (A, B, C, D) of the image to load.

    Saves
    -----
    warm_pixels_cor : PixelLineCollection
        The set of consistent warm pixel trails, saved to
        dataset.saved_consistent_lines(use_corrected=True).
    """
    # Load original warm pixels for the whole dataset
    warm_pixels = PixelLineCollection()
    warm_pixels.load(dataset.saved_consistent_lines(quadrant))

    # Corrected images
    warm_pixels_cor = PixelLineCollection()
    for i, image in enumerate(dataset):
        image_name = image.name
        print(
            f"\r    {image_name}_cor_{quadrant} ({i + 1} of {len(dataset)}) ",
            end="",
            flush=True,
        )

        # Load the image
        array = image.corrected().load_quadrant(quadrant)

        # Select consistent warm pixels found from this image
        image_name_q = image_name + "_%s" % quadrant
        sel = np.where(warm_pixels.origins == image_name_q)[0]
        for i in sel:
            line = warm_pixels.lines[i]
            row, column = line.location

            # Copy the original metadata but take the data from the corrected image
            warm_pixels_cor.append(
                PixelLine(
                    data=array[
                         row - ut.trail_length: row + ut.trail_length + 1, column
                         ],
                    origin=line.origin,
                    location=line.location,
                    date=line.date,
                    background=line.background,
                )
            )

    print("Extracted %d lines" % warm_pixels_cor.n_lines)

    # Save
    warm_pixels_cor.save(dataset.saved_consistent_lines(quadrant, use_corrected=True))


def stack_dataset_warm_pixels(dataset, quadrants, use_corrected=False):
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
        warm_pixels.load(dataset.saved_consistent_lines(quadrant, use_corrected))

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
    stacked_lines.save(dataset.saved_stacked_lines(quadrants, use_corrected))
    np.savez(
        dataset.saved_stacked_info(quadrants, use_corrected),
        row_bins,
        flux_bins,
        date_bins,
        background_bins,
    )
