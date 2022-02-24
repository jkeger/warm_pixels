import os

import warm_pixels.hst_functions.plot
from warm_pixels import hst_functions as fu
from warm_pixels import hst_utilities as ut
from warm_pixels import hst_utilities as ut
from warm_pixels.hst_data import Dataset
from warm_pixels.hst_functions import plot
from warm_pixels.hst_utilities import output_path
from warm_pixels.pixel_lines import PixelLine, PixelLineCollection
from warm_pixels.process import RawProcess, CorrectedProcess
from warm_pixels.warm_pixels import find_dataset_warm_pixels


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
        self.datasets = datasets
        self.quadrants = quadrants
        self.overwrite = overwrite
        self.prep_density = prep_density
        self.use_corrected = use_corrected
        self.plot_density = plot_density

        # TODO: list name was originally the input...
        self.list_name = "test"

    def main(self):
        # ========
        # Create directories to contain output plots
        # ========
        os.makedirs(ut.output_path / "stacked_trail_plots", exist_ok=True)
        os.makedirs(ut.output_path / "plotted_distributions", exist_ok=True)

        # ========
        # Find and stack warm pixels in each dataset
        # ========
        processes = []

        for i_dataset, dataset in enumerate(self.datasets):
            print(
                f'Dataset "{dataset.name}" '
                f'({i_dataset + 1} of {len(self.datasets)}, '
                f'{len(dataset)} images, "{self.quadrants}")'
            )
            process_ = RawProcess(
                dataset,
                self.quadrants,
                overwrite=self.overwrite,
            )
            if self.use_corrected:
                process_ = CorrectedProcess(process_)

            process_.plot()
            processes.append(process_)

        # ========
        # Compiled results from all datasets
        # ========
        # Fit and save the total trap densities
        if self.prep_density or self.plot_density:
            # In each image quadrant or combined quadrants
            for quadrants in self.quadrants.groups:
                print(
                    f"Fit total trap densities ({''.join(quadrants)})...",
                    end=" ",
                    flush=True,
                )
                fu.fit_total_trap_densities(
                    processes, self.list_name, quadrants, self.use_corrected
                )

        # Plot the trap density evolution
        if self.plot_density:
            print("Plot trap density evolution...", end=" ", flush=True)
            plot.plot_trap_density_evol(
                self.list_name, self.quadrants.groups, do_sunspots=True, use_corrected=self.use_corrected
            )
