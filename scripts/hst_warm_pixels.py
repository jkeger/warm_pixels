#!/usr/bin/env python

# noinspection PyUnresolvedReferences
"""
Find, stack, and plot warm pixels in multiple datasets of HST ACS images.

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
        warm-pixel-distributions
        stacked-trails
        density

--use-corrected, -u
    Use the corrected images with CTI removed instead of the originals for
    the trails etc (keeping the same selected warm pixel locations). Must first
    remove CTI from the images in each dataset (e.g. with `-r .`).
    Note, mdate_plot_stack defaults to "0" in this mode.

--after
--before

"""
import argparse
import datetime as dt
from pathlib import Path

from warm_pixels import WarmPixels
from warm_pixels.data.image import DAY_ZERO
from warm_pixels.data.source import FileDatasetSource
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
    "--downsample",
    "-w",
    default=None,
    type=int,
    help="Downsample to run 1/N of the datasets",
)

# Optional arguments
parser.add_argument(
    "--quadrants",
    "-q",
    default="ABCD",
    type=str,
    help="The image quadrants to use.",
)

# Other functions
parser.add_argument(
    '--plot',
    nargs='+',
    help='Specify plots from:\nwarm-pixels\nwarm-pixel-distributions\nstacked-trails\ndensity',
    required=True
)
parser.add_argument(
    "-u",
    "--corrected",
    action="store_true",
    help="Use the corrected images with CTI removed instead of the originals.",
)


def parse_date(value):
    try:
        days = int(value)
        return DAY_ZERO + dt.timedelta(days=days)
    except ValueError:
        return dt.date.fromisoformat(value)


def main():
    args = parser.parse_args()

    directory = args.directory

    source = FileDatasetSource(
        Path(directory),
    )
    print(f"Found {len(source)} datasets in {directory}")

    downsample = args.downsample
    if downsample is not None:
        source = source.downsample(
            step=downsample
        )
        print(f"Down sampling to every {downsample}th dataset -> {len(source)} datasets")

    after = args.after
    if after is not None:
        date = parse_date(after)
        print(f"Only include images captured after {date}")
        source = source.after(date)
    before = args.before
    if before is not None:
        date = parse_date(before)
        print(f"Only include images captured before {date}")
        source = source.before(date)

    use_corrected = args.corrected
    if use_corrected:
        print("Correcting image before analysis")
        source = source.corrected()

    datasets = list(source)

    datasets_string = "\n".join(
        f"{dataset.name} {dataset.observation_date()}"
        for dataset in datasets
    )

    print(f"Included Datasets:\n\n{datasets_string}\n")

    warm_pixels = WarmPixels(
        datasets=datasets,
        quadrants_string=args.quadrants
    )
    plot = Plot(
        warm_pixels,
        list_name=str(source),
        use_corrected=use_corrected,
    )
    plot.by_name(args.plot)


if __name__ == "__main__":
    main()
