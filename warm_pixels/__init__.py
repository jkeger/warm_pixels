import os
from pathlib import Path

import warm_pixels.hst_functions.plot
from warm_pixels import hst_functions as fu
from warm_pixels import hst_utilities as ut
from warm_pixels.data import Dataset
from warm_pixels.hst_functions import plot
from warm_pixels.hst_functions.fit import TrapDensities
from warm_pixels.hst_utilities import output_path
from warm_pixels.model.group import QuadrantGroup
from warm_pixels.model.quadrant import Quadrant, CorrectedQuadrant
from warm_pixels.pixel_lines import PixelLine, PixelLineCollection
from warm_pixels.quadrant_dataset import QuadrantDataset


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
        if use_corrected:
            datasets = [
                dataset.corrected()
                for dataset
                in datasets
            ]

        self.datasets = datasets
        self.quadrants = quadrants
        self.overwrite = overwrite
        self.prep_density = prep_density
        self.use_corrected = use_corrected
        self.plot_density = plot_density

        # TODO: list name was originally the input...
        self.list_name = "test"

    def need_to_make_file(self, filename):
        if self.overwrite:
            return True
        return not os.path.exists(filename)

    def main(self):
        # ========
        # Create directories to contain output plots
        # ========
        os.makedirs(ut.output_path / "stacked_trail_plots", exist_ok=True)
        os.makedirs(ut.output_path / "plotted_distributions", exist_ok=True)

        # ========
        # Find and stack warm pixels in each dataset
        # ========
        all_groups = []

        for i_dataset, dataset in enumerate(self.datasets):
            print(
                f'Dataset "{dataset.name}" '
                f'({i_dataset + 1} of {len(self.datasets)}, '
                f'{len(dataset)} images, "{self.quadrants}")'
            )

            quadrant_dataset_ = QuadrantDataset(
                dataset=dataset,
                quadrants_string=self.quadrants,
            )

            # List of lists of groups
            all_groups.append(quadrant_dataset_.groups)

            for quadrant in quadrant_dataset_.all_quadrants:
                for image_quadrant in quadrant.image_quadrants:
                    # Plot
                    fu.plot_warm_pixels(
                        image_quadrant.array(),
                        PixelLineCollection(
                            image_quadrant.warm_pixels(),
                        ),
                        save_path=dataset.output_path / image_quadrant.name,
                    )

            filename = dataset.plotted_distributions(self.quadrants)
            if self.need_to_make_file(filename):
                print("  Distributions of warm pixels...", end=" ", flush=True)
                fu.plot_warm_pixel_distributions(
                    quadrant_dataset_.all_quadrants,
                    save_path=filename,
                )

            for group in quadrant_dataset_.groups:
                filename = dataset.plotted_stacked_trails(
                    group,
                )
                if self.need_to_make_file(
                        filename
                ):
                    fu.plot_stacked_trails(
                        use_corrected=False,
                        save_path=filename,
                        group=group,
                    )

        all_trap_densities = []

        # ========
        # Compiled results from all datasets
        # ========
        # Fit and save the total trap densities
        if self.prep_density or self.plot_density:
            # Pivot groups into one list for each quadrant group over time
            for groups in zip(*all_groups):
                print(
                    f"Fit total trap densities ({''.join(self.quadrants)})...",
                    end=" ",
                    flush=True,
                )
                filename = Path(ut.dataset_list_saved_density_evol(
                    self.list_name,
                    groups,
                    self.use_corrected
                ))

                if not filename.exists():
                    trap_densities = fu.fit_total_trap_densities(groups)
                    trap_densities.save(filename)
                else:
                    trap_densities = TrapDensities.load(filename)
                all_trap_densities.append(
                    trap_densities
                )

        # Plot the trap density evolution
        if self.plot_density:
            print("Plot trap density evolution...", end=" ", flush=True)
            plot.plot_trap_density_evol(
                all_trap_densities=all_trap_densities,
                list_name=self.list_name,
                do_sunspots=True,
                use_corrected=self.use_corrected
            )
