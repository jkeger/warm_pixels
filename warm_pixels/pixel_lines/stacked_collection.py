import numpy as np

from .collection import AbstractPixelLineCollection
from .pixel_line import AbstractPixelLine


class StackedPixelLine(AbstractPixelLine):
    def __init__(
            self,
            length,
            location=None,
            date=None,
            background=None,
            flux=None,
    ):
        self._length = length
        super().__init__(
            location=location,
            date=date,
            background=background,
            flux=flux,
        )
        self.stacked_lines = []

    def append(self, pixel_line):
        self.stacked_lines.append(
            pixel_line
        )

    @property
    def n_stacked(self):
        return len(self.stacked_lines)

    @property
    def data(self):
        if len(self.stacked_lines) == 0:
            return np.zeros(self.length)
        return np.mean(self.stacked_data, axis=0)

    @property
    def stacked_data(self):
        return np.stack([line.data for line in self.stacked_lines])

    @property
    def noise(self):
        if self.n_stacked == 0:
            return np.zeros(self.length)
        return np.std(self.stacked_data, axis=0) / np.sqrt(self.n_stacked)

    @property
    def length(self):
        return self._length


class StackedPixelLineCollection(AbstractPixelLineCollection):
    def __init__(
            self,
            lines,
            row_bins,
            flux_bins,
            date_bins,
            background_bins,
    ):
        self._lines = lines
        self.row_bins = row_bins
        self.flux_bins = flux_bins
        self.date_bins = date_bins
        self.background_bins = background_bins

    @property
    def n_row_bins(self):
        return self.row_bins.number

    @property
    def n_flux_bins(self):
        return self.flux_bins.number

    @property
    def n_background_bins(self):
        return self.background_bins.number

    @property
    def lines(self):
        return self._lines
