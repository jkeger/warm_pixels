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
    def mean_row(self):
        return sum(line.location[0] for line in self.stacked_lines) / self.n_stacked

    @property
    def rms_row(self):
        return np.sqrt(sum(line.location[0] ** 2 for line in self.stacked_lines) / self.n_stacked)

    @property
    def mean_background(self):
        return sum(line.background for line in self.stacked_lines) / self.n_stacked

    @property
    def rms_background(self):
        return np.sqrt(sum(line.background ** 2 for line in self.stacked_lines) / self.n_stacked)

    @property
    def mean_flux(self):
        return sum(line.flux for line in self.stacked_lines) / self.n_stacked

    @property
    def rms_flux(self):
        return np.sqrt(sum(line.flux ** 2 for line in self.stacked_lines) / self.n_stacked)

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
            length,
            row_bins,
            flux_bins,
            date_bins,
            background_bins,
    ):
        self._lines = dict()
        self.length = length
        self.row_bins = row_bins
        self.flux_bins = flux_bins
        self.date_bins = date_bins
        self.background_bins = background_bins

    def stacked_line_for_indices(self, row_index, flux_index, date_index, background_index):
        key = (
            row_index,
            flux_index,
            date_index,
            background_index,
        )
        if key not in self._lines:
            self._lines[key] = StackedPixelLine(
                length=self.length,
                location=[self.row_bins[row_index], 0],
                date=self.date_bins[date_index],
                background=self.background_bins[background_index],
                flux=self.flux_bins[flux_index],
            )
        return self._lines[key]

    def stacked_line_for(self, row, flux, date, background):
        row_index = self.row_bins.index(row)
        flux_index = self.flux_bins.index(flux)
        date_index = self.date_bins.index(date)
        background_index = self.background_bins.index(background)
        return self.stacked_line_for_indices(
            row_index,
            flux_index,
            date_index,
            background_index
        )

    def add_line(self, line):
        self.stacked_line_for(
            line.location[0],
            line.flux,
            line.date,
            line.background
        ).append(line)

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
        return self._lines.values()
