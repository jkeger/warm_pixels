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

--mdate_old_*, -* <DATE> : str (opt.)
    A DATE="year/month/day" requirement to remake files saved/modified before
    this date. Default None or "." to only check whether a file already exists.
    Alternatively, set "1" to force remaking or "0" to force not.

    --mdate_all, -a
        Sets the default for all others, can be overridden individually.

    --mdate_find, -f
        Find warm pixels.

    --mdate_consistent, -c
        Consistent warm pixels.

    --mdate_plot_consistent, -C
        Plot distributions of consistent warm pixels.

    --mdate_stack, -s
        Stacked warm pixels.

    --mdate_plot_stack, -S
        Plot stacked trails.

    --mdate_remove_cti, -r
        Remove CTI from the images in each dataset, default "0".

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

import hst_functions as fu
from hst_data import *

# ========
# Main
# ========
if __name__ == "__main__":
    # ========
    # Parse arguments
    # ========
    parser = ut.prep_parser()
    args = parser.parse_args()

    # Datasets
    list_name = args.dataset_list
    if list_name not in dataset_lists.keys():
        print("Error: Invalid dataset_list", list_name)
        print("  Choose from:", list(dataset_lists.keys()))
        raise ValueError
    dataset_list = dataset_lists[list_name]

    # Split quadrants into separate or combined subsets
    # e.g. "AB_CD" --> [["A", "B"], ["C", "D"]]
    quadrant_sets = [[q for q in qs] for qs in args.quadrants.split("_")]
    # All quadrants, ignoring subsets
    all_quadrants = [q for qs in quadrant_sets for q in qs]

    # Override unset modified-date requirements
    if args.mdate_all is not None:
        if args.mdate_find is None:
            args.mdate_find = args.mdate_all
        if args.mdate_consistent is None:
            args.mdate_consistent = args.mdate_all
        if args.mdate_plot_consistent is None:
            args.mdate_plot_consistent = args.mdate_all
        if args.mdate_stack is None:
            args.mdate_stack = args.mdate_all
        if args.mdate_plot_stack is None:
            args.mdate_plot_stack = args.mdate_all

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
    if not os.path.exists(ut.path + "/stacked_trail_plots"):
        os.mkdir(ut.path + "/stacked_trail_plots")
    if not os.path.exists(ut.path + "/plotted_distributions"):
        os.mkdir(ut.path + "/plotted_distributions")

    # ========
    # Find and stack warm pixels in each dataset
    # ========
    for i_dataset, dataset in enumerate(dataset_list):
        print(
            'Dataset "%s" (%d of %d in "%s"%s, %d images, "%s")'
            % (
                dataset.name,
                i_dataset + 1,
                len(dataset_list),
                list_name,
                downsample_print,
                dataset.n_images,
                args.quadrants,
            )
        )

        # Remove CTI
        if ut.need_to_make_file(dataset.cor_paths[-1], mdate_old=args.mdate_remove_cti):
            fu.remove_cti_dataset(dataset)

        # Find warm pixels in each image quadrant
        for quadrant in all_quadrants:
            # Find possible warm pixels in each image
            if ut.need_to_make_file(
                    dataset.saved_lines(quadrant), mdate_old=args.mdate_find
            ):
                print(
                    "  Find possible warm pixels (%s)..." % quadrant,
                    end=" ",
                    flush=True,
                )
                fu.find_dataset_warm_pixels(dataset, quadrant)

            # Consistent warm pixels
            if args.use_corrected:
                # Extract from corrected images with CTI removed
                if ut.need_to_make_file(
                        dataset.saved_consistent_lines(quadrant, use_corrected=True),
                        mdate_old=args.mdate_consistent,
                ):
                    print("  Extract CTI-removed warm pixels (%s)..." % quadrant)
                    fu.extract_consistent_warm_pixels_corrected(dataset, quadrant)
            else:
                # Find consistent warm pixels in the set
                if ut.need_to_make_file(
                        dataset.saved_consistent_lines(quadrant),
                        mdate_old=args.mdate_consistent,
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
        if ut.need_to_make_file(
                dataset.plotted_distributions(all_quadrants),
                mdate_old=args.mdate_plot_consistent,
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
            if ut.need_to_make_file(
                    dataset.saved_stacked_lines(quadrants, args.use_corrected),
                    mdate_old=args.mdate_stack,
            ):
                print(
                    "  Stack warm pixel trails (%s)..." % "".join(quadrants),
                    end=" ",
                    flush=True,
                )
                fu.stack_dataset_warm_pixels(dataset, quadrants, args.use_corrected)

            # Plot stacked lines
            # RJM
            # if True:
            if ut.need_to_make_file(
                    dataset.plotted_stacked_trails(quadrants, args.use_corrected),
                    mdate_old=args.mdate_plot_stack,
            ):
                print(
                    "  Plot stacked trails (%s)..." % "".join(quadrants),
                    end=" ",
                    flush=True,
                )
                fu.plot_stacked_trails(
                    dataset,
                    quadrants,
                    use_corrected=args.use_corrected,
                    save_path=dataset.plotted_stacked_trails(
                        quadrants, args.use_corrected
                    ),
                )

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
