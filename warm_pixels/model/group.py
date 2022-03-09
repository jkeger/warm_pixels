from typing import List

from warm_pixels import hst_utilities as ut
from .cache import cache
from .quadrant import DatasetQuadrant
from warm_pixels.dataset import Dataset
from warm_pixels.pixel_lines import PixelLineCollection, StackedPixelLineCollection


class QuadrantGroup:
    def __init__(
            self,
            dataset: Dataset,
            quadrants: List[DatasetQuadrant]
    ):
        """
        Stacked and consistent lines are computed for a group of CCD quadrants
        and a collection of images.

        Parameters
        ----------
        dataset
            A collection of images all captured on the same date
        quadrants
            Objects comprising a dataset and a single quadrant that can be
            used to find warm pixels and lines
        """
        self.dataset = dataset
        self.quadrants = quadrants

    def __iter__(self):
        return iter(self.quadrants)

    @cache
    def consistent_lines(self) -> List[PixelLineCollection]:
        """
        Consistently warm lines for each quadrant
        """
        return [
            quadrant.consistent_lines()
            for quadrant in self.quadrants
        ]

    @cache
    def stacked_lines(self) -> StackedPixelLineCollection:
        """
        A combined collection of stacked lines for every quadrant
        computed by averaging within bins
        """
        return sum(
            self.consistent_lines()
        ).generate_stacked_lines_from_bins(
            n_row_bins=ut.n_row_bins,
            flux_bins=ut.flux_bins,
            n_background_bins=ut.n_background_bins,
        )
