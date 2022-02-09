import os

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
        if self.warm_pixels.use_corrected:
            fu.remove_cti_dataset(self.dataset)

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
            datasets,
            quadrants,
            overwrite=False,
            prep_density=False,
            use_corrected=False,
            plot_density=False,
    ):
        self.quadrants = quadrants
        self.overwrite = overwrite
        self.prep_density = prep_density
        self.use_corrected = use_corrected
        self.plot_density = plot_density

        # TODO: list name was originally the input...
        self.list_name = "test"

        self.datasets = datasets

    @property
    def quadrant_sets(self):
        return [[q for q in qs] for qs in self.quadrants.split("_")]

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
