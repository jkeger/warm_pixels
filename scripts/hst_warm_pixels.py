#!/usr/bin/env python

# noinspection PyUnresolvedReferences
"""
Find, stack, and plot warm pixels in multiple datasets of HST ACS images.

See functions.py and utilities.py.

Full pipeline:
+ For each dataset of images:
    + For each image (and quadrant), find possible warm pixels
    + Find consistent warm pixels
    + Plot distributions of the warm pixels
    + Stack the warm pixel trails in bins
    + Plot the stacked trails
Then:
+ Fit the total trap density across all datasets
+ Plot the evolution of the total trap density
Then:
+ Use the fitted trap model to remove CTI from the images
+ Repeat the analysis using the corrected images and check the trap density

See hst_utilities.py to set parameters like the trail length and bin edges.

By default, runs all the first-stage functions for the chosen list of image
datasets, skipping any that have been run before and saved their output.
Use the optional flags to choose manually which functions to run and to run
the next-stage functions.


Example use
-----------

./hst_warm_pixels.py directory --plot warm-pixels density


Parameters
----------
directory
    A directory containing dataset directories. Each dataset directory is loaded
    as a dataset.

--downsample
    Pass in an integer such that only every nth dataset is included in the fit

--quadrants, -q : str (opt.)
    The image quadrants to use, e.g. "A" or "ABCD" (default). To analyse the
    quadrants separately (done regardless for functions before the stacking),
    use e.g. "A_B_C_D", or e.g. "AB_CD" for combined A & B kept separate from
    combined C & D.

--plot
    Pass in a list of plots that should be output chosen from:
        warm-pixels
        warm-pixel_distributions
        stacked-trails
        density

--prep_density, -d
    Fit the total trap density across all datasets.

--plot_density, -D
    Plot the evolution of the total trap density.

--use_corrected, -u
    Use the corrected images with CTI removed instead of the originals for
    the trails etc (keeping the same selected warm pixel locations). Must first
    remove CTI from the images in each dataset (e.g. with `-r .`).
    Note, mdate_plot_stack defaults to "0" in this mode.

"""
import argparse
import datetime as dt
from pathlib import Path

from warm_pixels import WarmPixels
from warm_pixels.data.source import FileDatasetSource
from warm_pixels.hst_utilities import output_path
from warm_pixels.plot import Plot

parser = argparse.ArgumentParser()

# Positional arguments
parser.add_argument(
    "directory",
    nargs="?",
    type=str,
    help="The path to the directory containing data.",
)

# Filter
parser.add_argument(
    "--after",
    default=None,
    type=str
)
parser.add_argument(
    "--before",
    default=None,
    type=str
)
parser.add_argument(
    "-w",
    "--downsample",
    default=None,
    type=int,
    help="Downsample to run 1/N of the datasets",
)

# Optional arguments
parser.add_argument(
    "-q",
    "--quadrants",
    default="ABCD",
    type=str,
    help="The image quadrants to use.",
)

# Other functions
parser.add_argument(
    '--plot',
    nargs='+',
    help='Specify plots from {TODO}',
    required=True
)
parser.add_argument(
    "-u",
    "--use-corrected",
    action="store_true",
    help="Use the corrected images with CTI removed instead of the originals.",
)

DAY_ZERO = dt.date(2002, 3, 1)


def parse_date(value):
    try:
        days = int(value)
        return DAY_ZERO + dt.timedelta(days=days)
    except ValueError:
        return dt.date.fromisoformat(value)


def main():
    args = parser.parse_args()

    source = FileDatasetSource(
        Path(args.directory),
        output_path=output_path,
        quadrants_string=args.quadrants
    )

    downsample = args.downsample
    if downsample is not None:
        source = source.downsample(
            step=downsample
        )

    after = args.after
    if after is not None:
        source = source.after(
            parse_date(after)
        )
    before = args.before
    if before is not None:
        source = source.before(
            parse_date(before)
        )

    use_corrected = args.use_corrected
    if use_corrected:
        source = source.corrected()

    warm_pixels = WarmPixels(
        datasets=list(source),
    )
    plot = Plot(
        warm_pixels,
        list_name=str(source),
        use_corrected=use_corrected,
        quadrants_string=args.quadrants
    )
    plot.by_name(args.plot)


if __name__ == "__main__":
    main()
