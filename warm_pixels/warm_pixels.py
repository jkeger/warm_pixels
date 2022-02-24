from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter

from warm_pixels import hst_utilities as ut
from warm_pixels.hst_functions import plot_warm_pixels
from warm_pixels.pixel_lines import PixelLine, PixelLineCollection


def find_dataset_warm_pixels(dataset, quadrant):
    """Find the possible warm pixels in all images in a dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    quadrant : str (opt.)
        The quadrant (A, B, C, D) of the image to load.

    Saves
    -----
    warm_pixels : PixelLineCollection
        The set of warm pixel trails, saved to dataset.saved_lines().
    """
    # Initialise the collection of warm pixel trails
    warm_pixels = PixelLineCollection()
    print("")

    # Find the warm pixels in each image
    for i, image in enumerate(dataset.images):
        image_name = image.name
        print(
            "    %s_%s (%d of %d): "
            % (image_name, quadrant, i + 1, len(dataset)),
            end="",
            flush=True,
        )

        # Load the image
        array = image.load_quadrant(quadrant)

        date = 2400000.5 + array.header.modified_julian_date

        image_name_q = image_name + "_%s" % quadrant

        # Find the warm pixel trails
        new_warm_pixels = find_warm_pixels(
            image=array,
            trail_length=ut.trail_length,
            n_parallel_overscan=20,
            n_serial_prescan=24,
            origin=image_name_q,
            date=date,
        )
        print("Found %d possible warm pixels " % len(new_warm_pixels))

        # Plot
        plot_warm_pixels(
            array,
            PixelLineCollection(new_warm_pixels),
            save_path=dataset.output_path / image_name_q,
        )

        # Add them to the collection
        warm_pixels.append(new_warm_pixels)

    return warm_pixels


def find_warm_pixels(
        image,
        trail_length=9,
        n_parallel_overscan=0,
        n_serial_prescan=0,
        ignore_bad_columns=True,
        bad_column_factor=3.5,
        bad_column_loops=5,
        smooth_width=3,
        unsharp_masking_factor=6,
        flux_min: Optional[float] = None,
        origin=None,
        date=None,
):
    """Find warm (and hot) pixels in an image.

    Parameters
    ----------
    image : [[float]]
        The input array of pixel values.

        The first dimension is the "row" index, the second is the "column"
        index. By default (for parallel clocking), charge is transfered "up"
        from row n to row 0 along each independent column. i.e. the readout
        register is above row 0.

    trail_length : int
        The number of pixels in a trail, not including the warm pixel itself.
        The warm pixel itself and the same number of preceding pixels are also
        included in the saved line.

    n_parallel_overscan : int
        The number of rows in the overscan region of the input image. i.e. the
        final rows, furthest from the readout register, beyond the physical
        image. They should not contain warm pixels so will be ignored.

    n_serial_prescan : int
        The number of rows in the overscan region of the input image. i.e. the
        first columns, closest to the readout register, before the physical
        image. They should not contain warm pixels so will be ignored.

    ignore_bad_columns : bool
        Check for and ignore bad columns wiped out by extremely hot pixels.

    bad_column_factor : float
        Columns with a mean value more than this number of standard deviations
        above the overall median will be discarded, to avoid columns wiped out
        by extremely hot pixels.

    bad_column_loops : int
        The number of times to check for columns having means close enough to
        the overall median, updating the median each time.

    smooth_width : int
        The width of the window (in pixels) for calculating a smoothed image,
        used to find delta function-like warm pixels.

    unsharp_masking_factor : float
        Pixels must be this many times brighter than their neighbours in the
        smoothed image to be counted as warm pixels.

    flux_min
        Pixels below this value AFTER background subtraction will be ignored.
        Defaults to None to not ignore any pixels. Set to 0 to ignore warm
        pixels below the background.

    origin : str
        An identifier for the origin (e.g. image name) of the data, for the
        PixelLine objects' metadata.

    date : float
        The Julian date for the image, for the PixelLine objects' metadata.

    Returns
    -------
    warm_pixels : [PixelLine]
        A list of the warm pixels and associated data as PixelLine objects.
    """
    n_rows, n_columns = image.shape

    # Pixels flagged with 0 will be ignored
    where_not_ignored = np.ones_like(image)

    # List of not-ignored column indices, initially all
    good_columns = np.arange(n_columns)

    # Mean of each column
    column_means = np.mean(image, axis=0)

    # Identify bad columns to ignore, due to a really bright object or a pixel
    # hot enough to wipe everything above it, so have a high mean value
    if ignore_bad_columns:
        # Initialise to ignore all columns
        where_not_ignored *= 0

        # Remove columns with means far away from the median
        for i in range(bad_column_loops):
            median = np.median(column_means[good_columns])
            stddev = np.std(column_means[good_columns])
            # Keep columns with means close to the median
            good_columns = good_columns[
                abs(column_means[good_columns] - median) < bad_column_factor * stddev
                ]

        # Don't ignore the good columns
        where_not_ignored[:, good_columns] = 1

    # Subtract background
    background = 2.5 * np.median(column_means[good_columns]) - 1.5 * np.mean(
        column_means[good_columns]
    )
    image_no_bg = image - background

    # Don't ignore low-flux pixels if requested
    if flux_min is None:
        flux_min = np.nanmin(image_no_bg)

    # Unsharp mask image
    image_smooth = uniform_filter(image_no_bg, size=smooth_width)

    # Ignore the very top of the CCD since we can't get full trails
    where_not_ignored[: trail_length + 1, :] = 0
    # Ignore parallel overscan
    where_not_ignored[-(n_parallel_overscan + trail_length + 1):, :] = 0
    # Ignore serial prescan
    where_not_ignored[:, :n_serial_prescan] = 0

    # Calculate the maximum of the neighbouring pixels in the same column for
    # each pixel, not including that pixel
    nearby_maxima = np.maximum.reduce(
        [np.roll(image_no_bg, i + 1, axis=0) for i in range(trail_length)]
        + [np.roll(image_no_bg, -(i + 1), axis=0) for i in range(trail_length)]
    )

    # Identify warm pixels
    warm_pixel_locations = np.argwhere(
        # Not in ignored regions
        (where_not_ignored.astype(bool))
        # Local maximum
        & (image_no_bg > nearby_maxima)
        & (image_no_bg > np.roll(image_no_bg, 1, axis=1))
        & (image_no_bg > np.roll(image_no_bg, -1, axis=1))
        # Still local maximum after unsharp masking
        & (image_no_bg > unsharp_masking_factor * np.roll(image_smooth, 1, axis=0))
        & (image_no_bg > unsharp_masking_factor * np.roll(image_smooth, -1, axis=0))
        & (image_no_bg > unsharp_masking_factor * np.roll(image_smooth, 1, axis=1))
        & (image_no_bg > unsharp_masking_factor * np.roll(image_smooth, -1, axis=1))
        # Above the minimum flux above background
        & (image_no_bg >= flux_min)
    )
    n_warm_pixels = len(warm_pixel_locations)

    if n_warm_pixels == 0:
        return []

    # Assemble the list of warm pixel data
    warm_pixels = []
    for location in warm_pixel_locations:
        row, column = location

        warm_pixels.append(
            PixelLine(
                data=image[row - trail_length: row + trail_length + 1, column],
                origin=origin,
                location=[row, column],
                date=date,
                background=background,
            )
        )

    return warm_pixels
