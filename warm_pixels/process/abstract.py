import os
from abc import ABC, abstractmethod

from warm_pixels import PixelLineCollection
from warm_pixels import hst_functions as fu
from warm_pixels import hst_utilities as ut
from warm_pixels.pixel_lines import StackedPixelLineCollection


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
        self._cache = {}

    def need_to_make_file(self, filename):
        if self.overwrite:
            return True
        return not os.path.exists(filename)

    def consistent_lines_for_quadrant(
            self,
            quadrant
    ):
        if quadrant not in self._cache:
            self._cache[quadrant] = self._consistent_lines_for_quadrant(
                quadrant
            )
        return self._cache[quadrant]

    def _consistent_lines_for_quadrant(
            self,
            quadrant
    ):
        filename = self.dataset.saved_consistent_lines(quadrant)
        if self.need_to_make_file(filename):
            consistent_lines = self.consistent_lines_for(quadrant)
            consistent_lines.save(filename)
            return consistent_lines

        return PixelLineCollection.load(filename)

    def all_consistent_lines(self):
        return [
            self.consistent_lines_for_quadrant(quadrant)
            for quadrant in self.quadrants
        ]

    def all_stacked_lines(self):
        all_stacked_lines = []

        for group in self.quadrants.groups:
            stacked_lines = self.stacked_lines_for_group(group)
            all_stacked_lines.append(stacked_lines)
        return all_stacked_lines

    @abstractmethod
    def consistent_lines_for(self, quadrant):
        pass

    def plot(self):
        filename = self.dataset.plotted_distributions(self.quadrants)
        if self.need_to_make_file(filename):
            print("  Distributions of warm pixels...", end=" ", flush=True)
            fu.plot_warm_pixel_distributions(
                self.quadrants,
                self.all_consistent_lines(),
                save_path=filename,
            )

    def stacked_lines_for_group(self, group):
        consistent_line_group = []
        for quadrant in group:
            consistent_line_group.append(
                self.consistent_lines_for_quadrant(quadrant)
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
        else:
            stacked_lines = StackedPixelLineCollection.load(
                filename
            )
        return stacked_lines
