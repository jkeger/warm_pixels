from typing import List, Tuple

from warm_pixels import hst_functions as fu
from warm_pixels import hst_utilities as ut
from warm_pixels import plot
from warm_pixels.data import Dataset, Image
from warm_pixels.hst_functions.fit import TrapDensities
from warm_pixels.hst_utilities import output_path
from warm_pixels.model.cache import cache
from warm_pixels.model.group import QuadrantGroup
from warm_pixels.model.quadrant import Quadrant, CorrectedQuadrant
from warm_pixels.pixel_lines import PixelLine, PixelLineCollection


class WarmPixels:
    def __init__(
            self,
            datasets: List[Dataset],
    ):
        self.datasets = datasets

    def all_groups(self) -> List[List[QuadrantGroup]]:
        """
        A list of lists of quadrant groups with each sub-list containing
        groups corresponding to a dataset.
        """
        return [
            dataset.groups
            for dataset
            in self.datasets
        ]

    def all_groups_by_time(self) -> List[Tuple[QuadrantGroup]]:
        """
        A list of tuples of quadrant groups with each sub-list containing
        groups corresponding to some quadrants over time.
        """
        return list(zip(*self.all_groups()))

    @cache
    def all_trap_densities(self) -> List[TrapDensities]:
        """
        Fit the variation in trap density over time for each group of quadrants.
        """
        return [
            fu.fit_total_trap_densities(groups)
            for groups in self.all_groups_by_time()
        ]
