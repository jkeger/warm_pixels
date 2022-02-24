import os
from abc import ABC, abstractmethod

from warm_pixels import hst_functions as fu, PixelLineCollection
from warm_pixels import hst_utilities as ut


class AbstractProcess(ABC):
    def __init__(
            self,
            dataset,
            quadrants,
            overwrite,
    ):
        self.dataset = dataset
        self.overwrite = overwrite
        self.quadrants = quadrants

    def need_to_make_file(self, filename):
        if self.overwrite:
            return True
        return not os.path.exists(filename)

    def consistent_lines_for_quadrant(
            self,
            quadrant
    ):
        filename = self.dataset.saved_consistent_lines(quadrant)
        if self.need_to_make_file(filename):
            consistent_lines = self.consistent_lines_for(quadrant)
            consistent_lines.save(filename)
            return consistent_lines

        return PixelLineCollection.load(filename)

    def run(self):
        all_consistent_lines = []
        for group in self.quadrants.groups:
            consistent_line_group = []
            for quadrant in group:
                consistent_line_group.append(
                    self.consistent_lines_for_quadrant(quadrant)
                )
            all_consistent_lines.extend(
                consistent_line_group
            )

            filename = self.dataset.saved_stacked_lines(group)
            if self.need_to_make_file(filename):
                stacked_lines = sum(
                    consistent_line_group
                ).generate_stacked_lines_from_bins(
                    n_row_bins=ut.n_row_bins,
                    flux_bins=ut.flux_bins,
                    n_background_bins=ut.n_background_bins,
                )
                stacked_lines.save(filename)

        filename = self.dataset.plotted_distributions(self.quadrants)
        if self.need_to_make_file(filename):
            print("  Distributions of warm pixels...", end=" ", flush=True)
            fu.plot_warm_pixel_distributions(
                self.quadrants,
                all_consistent_lines,
                save_path=filename,
            )

    def stack_dataset_warm_pixels(self, quadrants):
        """Stack a set of premade warm pixel trails into bins.

        find_dataset_warm_pixels() and find_consistent_warm_pixels() must first be
        run for the dataset.

        Parameters
        ----------
        quadrants : [str]
            The list of quadrants (A, B, C, D) of the images to load, combined
            together if more than one provided.
        """
        warm_pixels = sum(
            self.consistent_lines_for_quadrant(
                quadrant
            )
            for quadrant in quadrants
        )

        # Stack the lines in bins by distance from readout and total flux
        return warm_pixels.generate_stacked_lines_from_bins(
            n_row_bins=ut.n_row_bins,
            flux_bins=ut.flux_bins,
            n_background_bins=ut.n_background_bins,
        )

    @abstractmethod
    def consistent_lines_for(self, quadrant):
        pass
