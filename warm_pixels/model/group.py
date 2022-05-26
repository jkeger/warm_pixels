from typing import List

from warm_pixels import hst_utilities as ut
from warm_pixels.pixel_lines import StackedPixelLineCollection
from .cache import persist
from .quadrant import DatasetQuadrant


def directory_func(quadrant_group):
    return f"{quadrant_group.dataset}/{quadrant_group}"


class QuadrantGroup:
    def __init__(
            self,
            quadrants: List[DatasetQuadrant]
    ):
        """
        Stacked and consistent lines are computed for a group of CCD quadrants
        and a collection of images.

        Parameters
        ----------
        quadrants
            Objects comprising a dataset and a single quadrant that can be
            used to find warm pixels and lines
        """
        self.quadrants = quadrants

    def __iter__(self):
        return iter(self.quadrants)

    def __str__(self):
        return "".join(
            quadrant.quadrant
            for quadrant
            in self.quadrants
        )

    @property
    def dataset(self):
        return self.quadrants[0].dataset

    @persist(directory_func)
    def stacked_lines(self) -> StackedPixelLineCollection:
        """
        A combined collection of stacked lines for every quadrant
        computed by averaging within bins
        """
        combined = sum(
            quadrant.consistent_lines()
            for quadrant in self.quadrants
        )
        return combined.generate_stacked_lines_from_bins(
            flux_bins=ut.flux_bins,
        )
