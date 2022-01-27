#!/usr/bin/env python

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
import os
from pathlib import Path

from warm_pixels import hst_functions as fu
from warm_pixels import hst_utilities as ut
from warm_pixels.hst_data import Dataset
from warm_pixels.hst_utilities import output_path
from warm_pixels.warm_pixels import find_dataset_warm_pixels

# ========
# Main
# ========
if __name__ == "__main__":
    # ========
    # Parse arguments
    # ========
    parser = ut.prep_parser()
    args = parser.parse_args()


    def need_to_make_file(filename):
        if args.overwrite:
            return True
        return not os.path.exists(filename)


    # TODO: list name was originally the input...
    list_name = "TODO"

    dataset_list = [
        Dataset(
            Path(args.directory),
            output_path=output_path
        )
    ]

    # Split quadrants into separate or combined subsets
    # e.g. "AB_CD" --> [["A", "B"], ["C", "D"]]
    quadrant_sets = [[q for q in qs] for qs in args.quadrants.split("_")]
    # All quadrants, ignoring subsets
    all_quadrants = [q for qs in quadrant_sets for q in qs]

    # Modified defaults
    if args.use_corrected:
        # Don't automatically plot stacked plots of the corrected images
        if args.mdate_plot_stack is None:
            args.mdate_plot_stack = "0"

    # Downsample the dataset list
    if args.downsample is not None:
        N = int(args.downsample[0])
        i = int(args.downsample[1])
        dataset_list = dataset_list[i::N]
        downsample_print = "[%d::%d]" % (i, N)
    else:
        downsample_print = ""

    # Test loading the image and corresponding bias files
    if args.test_image_and_bias_files:
        print("# Testing image and bias files...")
        all_okay = True

        for dataset in dataset_list:
            if not ut.test_image_and_bias_files(dataset):
                all_okay = False
        print("")

        if not all_okay:
            exit()

    # Use the corrected images with CTI removed instead
    if args.use_corrected:
        print("# Using the corrected images with CTI removed. \n")

    # ========
    # Create directories to contain output plots
    # ========
    os.makedirs(ut.output_path / "stacked_trail_plots", exist_ok=True)
    os.makedirs(ut.output_path / "plotted_distributions", exist_ok=True)

    # ========
    # Find and stack warm pixels in each dataset
    # ========
    for i_dataset, dataset in enumerate(dataset_list):
        print(
            f'Dataset "{dataset.name}" '
            f'({i_dataset + 1} of {len(dataset_list)} in {downsample_print}, '
            f'{len(dataset)} images, "{args.quadrants}")'
        )

        # TODO: Commented because arctic crashes
        # # Remove CTI
        # if need_to_make_file(
        #         dataset.images[-1].cor_path,
        # ):
        #     fu.remove_cti_dataset(dataset)

        # Find warm pixels in each image quadrant
        for quadrant in all_quadrants:
            # Find possible warm pixels in each image
            if need_to_make_file(
                    dataset.saved_lines(quadrant),
            ):
                print(
                    "  Find possible warm pixels (%s)..." % quadrant,
                    end=" ",
                    flush=True,
                )
                warm_pixels = find_dataset_warm_pixels(dataset, quadrant)

            # Consistent warm pixels
            if args.use_corrected:
                # Extract from corrected images with CTI removed
                if need_to_make_file(
                        dataset.saved_consistent_lines(quadrant, use_corrected=True),
                ):
                    print("  Extract CTI-removed warm pixels (%s)..." % quadrant)
                    fu.extract_consistent_warm_pixels_corrected(dataset, quadrant)
            else:
                # Find consistent warm pixels in the set
                if need_to_make_file(
                        dataset.saved_consistent_lines(quadrant),
                ):
                    print(
                        "  Consistent warm pixels (%s)..." % quadrant,
                        end=" ",
                        flush=True,
                    )
                    fu.find_consistent_warm_pixels(
                        dataset,
                        quadrant,
                        flux_min=ut.flux_bins[0],
                        flux_max=ut.flux_bins[-1],
                    )

        # Plot distributions of warm pixels in the set
        if need_to_make_file(
                dataset.plotted_distributions(all_quadrants),
        ):
            print("  Distributions of warm pixels...", end=" ", flush=True)
            fu.plot_warm_pixel_distributions(
                dataset,
                all_quadrants,
                save_path=dataset.plotted_distributions(all_quadrants),
            )

        # Stack warm pixels in each image quadrant or combined quadrants
        for quadrants in quadrant_sets:
            # Stack in bins
            # RJM
            # if True:
            if need_to_make_file(
                    dataset.saved_stacked_lines(quadrants, args.use_corrected),
            ):
                print(
                    "  Stack warm pixel trails (%s)..." % "".join(quadrants),
                    end=" ",
                    flush=True,
                )
                fu.stack_dataset_warm_pixels(dataset, quadrants, args.use_corrected)

            # TODO: commented because Arctic crashes
            # Plot stacked lines
            # if need_to_make_file(
            #         dataset.plotted_stacked_trails(quadrants, args.use_corrected),
            # ):
            #     print(
            #         "  Plot stacked trails (%s)..." % "".join(quadrants),
            #         end=" ",
            #         flush=True,
            #     )
            #     fu.plot_stacked_trails(
            #         dataset,
            #         quadrants,
            #         use_corrected=args.use_corrected,
            #         save_path=dataset.plotted_stacked_trails(
            #             quadrants, args.use_corrected
            #         ),
            #     )

    # ========
    # Compiled results from all datasets
    # ========
    # Fit and save the total trap densities
    if args.prep_density:
        # In each image quadrant or combined quadrants
        for quadrants in quadrant_sets:
            print(
                "Fit total trap densities (%s)..." % "".join(quadrants),
                end=" ",
                flush=True,
            )
            fu.fit_total_trap_densities(
                dataset_list, list_name, quadrants, args.use_corrected
            )

    # Plot the trap density evolution
    if args.plot_density:
        print("Plot trap density evolution...", end=" ", flush=True)
        fu.plot_trap_density_evol(
            list_name, quadrant_sets, do_sunspots=True, use_corrected=args.use_corrected
        )
