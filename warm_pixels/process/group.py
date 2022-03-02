from typing import List

from warm_pixels import hst_utilities as ut
from .cache import cache
from .quadrant import Quadrant


class Group:
    def __init__(self, dataset, quadrants: List[Quadrant]):
        self.dataset = dataset
        self.quadrants = quadrants

    def __iter__(self):
        return iter(self.quadrants)

    @cache
    def consistent_lines(self):
        return [
            quadrant.consistent_lines()
            for quadrant in self.quadrants
        ]

    @cache
    def stacked_lines(self):
        return sum(
            self.consistent_lines()
        ).generate_stacked_lines_from_bins(
            n_row_bins=ut.n_row_bins,
            flux_bins=ut.flux_bins,
            n_background_bins=ut.n_background_bins,
        )
