import numpy as np

from .collection import AbstractPixelLineCollection
from .pixel_line import AbstractPixelLine, _dump
from ..bins import Bins


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
    def dict(self) -> dict:
        """
        A dictionary representation of the pixel line. This can
        be used to create a Dataset1D in autocti.
        """
        return _dump({
            **super().dict,
            "mean_row": self.mean_row,
            "rms_row": self.rms_row,
            "mean_background": self.mean_background,
            "rms_background": self.rms_background,
            "mean_flux": self.mean_flux,
            "rms_flux": self.rms_flux,
        })

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
            length: int,
            row_bins: Bins,
            flux_bins: Bins,
            date_bins: Bins,
            background_bins: Bins,
    ):
        """
        A collection where pixel lines are grouped into bins by their attributes.

        For example, two pixel lines in nearby rows on the CCD, with similar flux,
        similar background and captured on similar dates may be added to the same
        group in 4D.

        Parameters
        ----------
        length
            The length of the data array containing each pixel line
        row_bins
            Bins for rows in the CCD
        flux_bins
            Bins for different flux values of each warm pixel
        date_bins
            Bins for the data on which warm pixels or pixel lines
            were observed
        background_bins
            Bins for the background when a pixel line was observed
        """
        self._lines = dict()
        self.length = length
        self.row_bins = row_bins
        self.flux_bins = flux_bins
        self.date_bins = date_bins
        self.background_bins = background_bins

    def stacked_line_for_indices(
            self,
            row_index: int,
            flux_index: int,
            date_index: int,
            background_index: int,
    ) -> "StackedPixelLine":
        """
        Retrieve or create a StackedLine for a given hyperbin in 4D parameter
        space.

        Parameters
        ----------
        row_index
            The index of the bin in the row direction
        flux_index
            The index of the bin in the flux direction
        date_index
            The index of the bin in the date direction
        background_index
            The index of the bin in the background direction

        Returns
        -------
        An object comprising zero or more pixel lines with similar attributes
        """
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

    def stacked_line_for(
            self,
            row: int,
            flux: float,
            date: float,
            background: float,
    ) -> StackedPixelLine:
        """
        Retrieve or create a StackedLine by computing which bin a given combination
        of attribute values corresponds to.

        Parameters
        ----------
        row
            The value used to determine the index of the bin in the row direction
        flux
            The value used to determine the index of the bin in the flux direction
        date
            The value used to determine the index of the bin in the date direction
        background
            The value used to determine the index of the bin in the background direction

        Returns
        -------
        An object comprising zero or more pixel lines with similar attributes
        """
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

    def add_line(self, line: AbstractPixelLine):
        """
        Add a pixel line to the relevant group in the collection.

        The bin is created or retrieved based on the which bins the
        attributes of the line (row, flux, date and background)
        correspond to

        Parameters
        ----------
        line
            A line of pixels representing the trail from a warm
            pixel
        """
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
