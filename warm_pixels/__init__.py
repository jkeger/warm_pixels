from warm_pixels import hst_functions as fu
from warm_pixels import hst_utilities as ut
from warm_pixels.data import Dataset
from warm_pixels.hst_functions.fit import TrapDensities
from warm_pixels.hst_utilities import output_path
from warm_pixels.model.cache import cache
from warm_pixels.model.group import QuadrantGroup
from warm_pixels.model.quadrant import Quadrant, CorrectedQuadrant
from warm_pixels.pixel_lines import PixelLine, PixelLineCollection


class WarmPixels:
    def __init__(
            self,
            datasets,
    ):
        self.datasets = datasets

    @property
    def quadrants(self):
        return map(str, self.datasets[0].all_quadrants)

    def all_groups(self):
        return [
            quadrant_dataset_.groups
            for quadrant_dataset_
            in self.datasets
        ]

    @cache
    def all_trap_densities(self):
        # Pivot groups into one list for each quadrant group over time
        return [
            fu.fit_total_trap_densities(groups)
            for groups in zip(*self.all_groups())
        ]
