import os
from abc import ABC

from warm_pixels import hst_functions as fu
from warm_pixels.process.quadrant import Group, Quadrant


class AbstractProcess(ABC):
    def __init__(
            self,
            dataset,
            quadrants,
            overwrite,
    ):
        self.dataset = dataset
        self.overwrite = overwrite

        self.groups = [
            Group([
                Quadrant(
                    quadrant=quadrant,
                    dataset=self.dataset
                )
                for quadrant in group
            ])
            for group in quadrants.groups
        ]
        self._cache = {}

    @property
    def quadrants(self):
        return [
            quadrant
            for group in self.groups
            for quadrant in group.quadrants
        ]

    def need_to_make_file(self, filename):
        if self.overwrite:
            return True
        return not os.path.exists(filename)

    def all_consistent_lines(self):
        return [
            quadrant.consistent_lines()
            for quadrant in self.quadrants
        ]

    def all_stacked_lines(self):
        return [
            group.stacked_lines()
            for group in self.groups
        ]

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
        return Group(
            self.dataset,
            [
                Quadrant(
                    quadrant=quadrant,
                    dataset=self.dataset
                )
                for quadrant in group
            ]
        ).stacked_lines()
