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
import os
from pathlib import Path

from warm_pixels import hst_functions as fu
from warm_pixels import hst_utilities as ut
from warm_pixels.hst_data import Dataset
from warm_pixels.hst_utilities import output_path
from warm_pixels.warm_pixels import find_dataset_warm_pixels


class DatasetProcess:
    def __init__(
            self,
            dataset,
            warm_pixels
    ):
        self.dataset = dataset
        self.warm_pixels = warm_pixels

    def process_quadrant(self, quadrant):
        # Find possible warm pixels in each image
        if self.warm_pixels.need_to_make_file(
                self.dataset.saved_lines(quadrant),
        ):
            print(
                "  Find possible warm pixels (%s)..." % quadrant,
                end=" ",
                flush=True,
            )
            find_dataset_warm_pixels(self.dataset, quadrant)

        # Consistent warm pixels
        if self.warm_pixels.use_corrected:
            # Extract from corrected images with CTI removed
            if self.warm_pixels.need_to_make_file(
                    self.dataset.saved_consistent_lines(quadrant, use_corrected=True),
            ):
                print("  Extract CTI-removed warm pixels (%s)..." % quadrant)
                fu.extract_consistent_warm_pixels_corrected(self.dataset, quadrant)
        else:
            # Find consistent warm pixels in the set
            if self.warm_pixels.need_to_make_file(
                    self.dataset.saved_consistent_lines(quadrant),
            ):
                print(
                    "  Consistent warm pixels (%s)..." % quadrant,
                    end=" ",
                    flush=True,
                )
                fu.find_consistent_warm_pixels(
                    self.dataset,
                    quadrant,
                    flux_min=ut.flux_bins[0],
                    flux_max=ut.flux_bins[-1],
                )

    def run(self):
        for quadrant in self.warm_pixels.all_quadrants:
            self.process_quadrant(quadrant)

        self.plot_distributions()

        # Stack warm pixels in each image quadrant or combined quadrants
        for quadrants in self.warm_pixels.quadrant_sets:
            self.stack_warm_pixels(quadrants)

    def plot_distributions(self):
        # Plot distributions of warm pixels in the set
        if self.warm_pixels.need_to_make_file(
                self.dataset.plotted_distributions(self.warm_pixels.all_quadrants),
        ):
            print("  Distributions of warm pixels...", end=" ", flush=True)
            fu.plot_warm_pixel_distributions(
                self.dataset,
                self.warm_pixels.all_quadrants,
                save_path=self.dataset.plotted_distributions(self.warm_pixels.all_quadrants),
            )

    def stack_warm_pixels(self, quadrants):
        # Stack in bins
        if self.warm_pixels.need_to_make_file(
                self.dataset.saved_stacked_lines(quadrants, self.warm_pixels.use_corrected),
        ):
            print(
                f"  Stack warm pixel trails ({''.join(quadrants)})...",
                end=" ",
                flush=True,
            )
            fu.stack_dataset_warm_pixels(self.dataset, quadrants, self.warm_pixels.use_corrected)

        # TODO: commented because Arctic crashes
        # Plot stacked lines
        # if not self.use_corrected and self.need_to_make_file(
        #         dataset.plotted_stacked_trails(
        #             quadrants,
        #             self.use_corrected
        #         ),
        # ):
        #     print(
        #         f"  Plot stacked trails ({''.join(quadrants)})...",
        #         end=" ",
        #         flush=True,
        #     )
        #     fu.plot_stacked_trails(
        #         dataset,
        #         quadrants,
        #         use_corrected=self.use_corrected,
        #         save_path=dataset.plotted_stacked_trails(
        #             quadrants, self.use_corrected
        #         ),
        #     )


class WarmPixels:
    def __init__(
            self,
            directory,
            quadrants,
            overwrite=False,
            downsample=None,
            prep_density=False,
            use_corrected=False,
            plot_density=False,
    ):
        self.directory = directory

        self.quadrants = quadrants
        self.overwrite = overwrite
        self.prep_density = prep_density
        self.use_corrected = use_corrected
        self.plot_density = plot_density

        # TODO: list name was originally the input...
        self.list_name = "TODO"

        self.datasets = self._load_datasets(downsample)

    def _load_datasets(self, downsample=None):
        datasets = [
            Dataset(
                Path(self.directory),
                output_path=output_path
            )
        ]
        # Downsample the dataset list
        if downsample is not None:
            n = int(downsample[0])
            i = int(downsample[1])
            datasets = datasets[i::n]
            print(f"Down-sampling [{i}::{n}]")
        return datasets

    @property
    def quadrant_sets(self):
        return [[q for q in qs] for qs in args.quadrants.split("_")]

    def need_to_make_file(self, filename):
        if self.overwrite:
            return True
        return not os.path.exists(filename)

    @property
    def all_quadrants(self):
        # All quadrants, ignoring subsets
        return [q for qs in self.quadrant_sets for q in qs]

    def main(self):
        # Use the corrected images with CTI removed instead
        if self.use_corrected:
            print("# Using the corrected images with CTI removed. \n")

        # ========
        # Create directories to contain output plots
        # ========
        os.makedirs(ut.output_path / "stacked_trail_plots", exist_ok=True)
        os.makedirs(ut.output_path / "plotted_distributions", exist_ok=True)

        # ========
        # Find and stack warm pixels in each dataset
        # ========
        for i_dataset, dataset in enumerate(self.datasets):
            print(
                f'Dataset "{dataset.name}" '
                f'({i_dataset + 1} of {len(self.datasets)}, '
                f'{len(dataset)} images, "{self.quadrants}")'
            )
            DatasetProcess(dataset, self).run()

        # ========
        # Compiled results from all datasets
        # ========
        # Fit and save the total trap densities
        if self.prep_density:
            # In each image quadrant or combined quadrants
            for quadrants in self.quadrant_sets:
                print(
                    f"Fit total trap densities ({''.join(quadrants)})...",
                    end=" ",
                    flush=True,
                )
                fu.fit_total_trap_densities(
                    self.datasets, self.list_name, quadrants, self.use_corrected
                )

        # Plot the trap density evolution
        if self.plot_density:
            print("Plot trap density evolution...", end=" ", flush=True)
            fu.plot_trap_density_evol(
                self.list_name, self.quadrant_sets, do_sunspots=True, use_corrected=self.use_corrected
            )


if __name__ == "__main__":
    parser = ut.prep_parser()
    args = parser.parse_args()

    WarmPixels(
        directory=args.directory,
        quadrants=args.quadrants,
        overwrite=args.overwrite,
        downsample=args.downsample,
        prep_density=args.prep_density,
        use_corrected=args.use_corrected,
        plot_density=args.plot_density,
    ).main()
