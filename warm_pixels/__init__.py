import os

import warm_pixels.hst_functions.plot
from warm_pixels import hst_functions as fu
from warm_pixels import hst_utilities as ut
from warm_pixels.data import Dataset
from warm_pixels.hst_functions import plot
from warm_pixels.hst_functions.fit import TrapDensities
from warm_pixels.hst_utilities import output_path
from warm_pixels.model.cache import cache
from warm_pixels.model.group import QuadrantGroup
from warm_pixels.model.quadrant import Quadrant, CorrectedQuadrant
from warm_pixels.pixel_lines import PixelLine, PixelLineCollection
from warm_pixels.quadrant_dataset import QuadrantDataset


def plot_all_warm_pixels(quadrant_dataset_):
    for quadrant in quadrant_dataset_.all_quadrants:
        for image_quadrant in quadrant.image_quadrants:
            # Plot
            fu.plot_warm_pixels(
                image_quadrant.array(),
                PixelLineCollection(
                    image_quadrant.warm_pixels(),
                ),
                save_path=quadrant_dataset_.dataset.output_path / image_quadrant.name,
            )


def output_plots(
        warm_pixels_,
        list_name,
        use_corrected=False,
        plot_density=False,
        plot_warm_pixels=False,
):
    for quadrant_dataset_ in warm_pixels_.quadrant_datasets():
        dataset = quadrant_dataset_.dataset

        if plot_warm_pixels:
            plot_all_warm_pixels(quadrant_dataset_)

        filename = dataset.plotted_distributions(
            warm_pixels_.quadrants
        )
        fu.plot_warm_pixel_distributions(
            quadrant_dataset_.all_quadrants,
            save_path=filename,
        )

        for group in quadrant_dataset_.groups:
            filename = dataset.plotted_stacked_trails(
                group,
            )
            fu.plot_stacked_trails(
                use_corrected=False,
                save_path=filename,
                group=group,
            )

    if plot_density:
        save_path = ut.dataset_list_plotted_density_evol(
            list_name,
            [
                trap_densities.quadrants_string
                for trap_densities
                in warm_pixels_.all_trap_densities()
            ]
        )
        print("Plot trap density evolution...", end=" ", flush=True)
        plot.plot_trap_density_evol(
            all_trap_densities=warm_pixels_.all_trap_densities(),
            use_corrected=use_corrected,
            save_path=save_path
        )


class WarmPixels:
    def __init__(
            self,
            datasets,
            quadrants,
    ):
        self.datasets = datasets
        self.quadrants = quadrants

    @cache
    def quadrant_datasets(self):
        return [
            QuadrantDataset(
                dataset=dataset,
                quadrants_string=self.quadrants,
            )
            for dataset
            in self.datasets
        ]

    def all_groups(self):
        return [
            quadrant_dataset_.groups
            for quadrant_dataset_
            in self.quadrant_datasets()
        ]

    @cache
    def all_trap_densities(self):
        # Pivot groups into one list for each quadrant group over time
        return [
            fu.fit_total_trap_densities(groups)
            for groups in zip(*self.all_groups())
        ]
