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

python3 hst_warm_pixels.py sample


Parameters
----------
dataset_list : str (opt.)
    The name of the list of image datasets to run. Defaults to "test". See the
    dataset_lists dictionary for the options.

--quadrants, -q : str (opt.)
    The image quadrants to use, e.g. "A" or "ABCD" (default). To analyse the
    quadrants separately (done regardless for functions before the stacking),
    use e.g. "A_B_C_D", or e.g. "AB_CD" for combined A & B kept separate from
    combined C & D.

--prep_density, -d
    Fit the total trap density across all datasets.

--plot_density, -D
    Plot the evolution of the total trap density.

--use_corrected, -u
    Use the corrected images with CTI removed instead of the originals for
    the trails etc (keeping the same selected warm pixel locations). Must first
    remove CTI from the images in each dataset (e.g. with `-r .`).
    Note, mdate_plot_stack defaults to "0" in this mode.

--downsample, -w <N> <i> : int int
    Downsample the dataset list to run 1/N of the datasets, starting with set i.
    e.g. -w 10 5 will run the datasets with indices 5, 15, 25, ... in the list.

--test_image_and_bias_files, -t
    Test loading the image and corresponding bias files in the list of datasets.
"""
import argparse
from pathlib import Path

from warm_pixels import WarmPixels
from warm_pixels.data.source import FileDatasetSource
from warm_pixels.hst_utilities import output_path

parser = argparse.ArgumentParser()

# Positional arguments
parser.add_argument(
    "directory",
    nargs="?",
    type=str,
    help="The path to the directory containing data.",
)

# Optional arguments
parser.add_argument(
    "-q",
    "--quadrants",
    default="ABCD",
    type=str,
    help="The image quadrants to use.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="If --overwrite is passed then overwrite existing files"
)

# Other functions
parser.add_argument(
    "-d",
    "--prep-density",
    action="store_true",
    help="Fit the total trap density for all datasets.",
)
parser.add_argument(
    "-D",
    "--plot-density",
    action="store_true",
    help="Plot the evolution of the total trap density.",
)
parser.add_argument(
    "-u",
    "--use-corrected",
    action="store_true",
    help="Use the corrected images with CTI removed instead of the originals.",
)
parser.add_argument(
    "-w",
    "--downsample",
    default=None,
    type=int,
    help="Downsample to run 1/N of the datasets",
)


def main():
    args = parser.parse_args()

    dataset_source = FileDatasetSource(
        Path(args.directory),
        output_path=output_path,
    )

    downsample = args.downsample
    if downsample is not None:
        dataset_source.downsample(
            step=downsample
        )

    WarmPixels(
        datasets=list(dataset_source),
        quadrants=args.quadrants,
        overwrite=args.overwrite,
        prep_density=args.prep_density,
        use_corrected=args.use_corrected,
        plot_density=args.plot_density,
        list_name=str(dataset_source),
    ).main()


if __name__ == "__main__":
    main()
