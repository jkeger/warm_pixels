from abc import ABC, abstractmethod

import numpy as np

from warm_pixels import hst_utilities as ut
from .pixel_line import PixelLine
from ..bins import Bins


class AbstractPixelLineCollection(ABC):
    def __getitem__(self, item):
        return self.lines[item]

    def consistent(
            self,
            flux_min=ut.flux_bins[0],
            flux_max=ut.flux_bins[-1],
    ):
        """Find the consistent warm pixels in a dataset.

        find_dataset_warm_pixels() must first be run for the dataset.

        Parameters
        ----------
        flux_min, flux_max : float (opt.)
            If provided, then before checking for consistent pixels, discard any
            with fluxes outside of these limits.
        """
        # Ignore warm pixels below the minimum flux
        sel_flux = np.where(
            (flux_min < self.fluxes) & (self.fluxes < flux_max)
        )[0]
        within_fluxes = PixelLineCollection(
            np.array(self.lines)[sel_flux]
        )

        # Find the warm pixels present in at least e.g. 2/3 of the images
        consistent_lines = within_fluxes.find_consistent_lines(
            fraction_present=ut.fraction_present
        )

        return PixelLineCollection(
            within_fluxes.lines[consistent_lines]
        )

    def __radd__(self, other):
        if other == 0:
            return self
        raise ValueError()

    def __add__(self, other):
        collection = PixelLineCollection()
        collection.extend(self.lines)
        if isinstance(
                other, PixelLineCollection
        ):
            collection.extend(other.lines)
        if isinstance(
                other, PixelLine
        ):
            collection.append(other)
        return collection

    def __eq__(self, other):
        return self.lines == other.lines

    def __len__(self):
        return len(self.lines)

    def __iter__(self):
        return iter(self.lines)

    @property
    def data(self) -> np.ndarray:
        """
        The pixel counts of each line, in units of electrons.
        """
        return np.array([line.data for line in self.lines])

    @property
    def origins(self):
        """
        The identifiers for the origins (e.g. image name) of each line.
        """
        return np.array([line.origin for line in self.lines])

    @property
    def locations(self):
        """
        The row and column indices of the first pixel in the line in the
            image, for each line.
        """
        return np.array([line.location for line in self.lines])

    @property
    def dates(self):
        """
        The Julian date of each line.
        """
        return np.array([line.date for line in self.lines])

    @property
    def backgrounds(self):
        """
        The background charge count of each line, in units of electrons.
        """
        return np.array([line.background for line in self.lines])

    @property
    def fluxes(self):
        """
        The maximum charge in each line, in units of electrons.
        """
        return np.array([line.flux for line in self.lines])

    @property
    def lengths(self):
        """
        The number of pixels in the data array of each line.
        """
        return np.array([line.length for line in self.lines])

    @property
    def n_lines(self):
        """
        The number of lines in the collection.
        """
        return len(self.lines)

    @property
    @abstractmethod
    def lines(self):
        pass

    def remove_symmetry(self, n_pixels_used_for_background=5):
        """Convert each line from a format that has a warm pixel in the middle (and background)
        to just the trail (without background)."""
        self = np.array([line.remove_symmetry(n_pixels_used_for_background) for line in self.lines])
        return

    def find_consistent_lines(self, fraction_present=2 / 3):
        """Identify lines that are consistently present across several images.

        This helps to identify warm pixels by discarding noise peaks.

        Parameters
        ----------
        self : PixelLineCollection
            Must contain lines from multiple images as identified by their
            PixelLine.origin with potentially matching lines with the same
            PixelLine.location in their images.

        fraction_present : float
            The minimum fraction of images in which the pixel must be present.

        Returns
        -------
        consistent_lines : [int]
            The indices of consistently present pixel lines in the attribute
            arrays.
        """
        if len(self._lines) == 0:
            return []

        # Number of separate images
        n_images = len(np.unique(self.origins))

        # Map the 2D locations to a 1D array of single numbers
        max_column = np.amax(self.locations[:, 1]) + 1
        locations_1D = self.locations[:, 0] * max_column + self.locations[:, 1]

        # The possible locations of warm pixels and the number at that location
        unique_locations, counts = np.unique(locations_1D, axis=0, return_counts=True)

        # The unique locations with sufficient numbers of matching pixels
        consistent_locations = unique_locations[counts / n_images >= fraction_present]

        # Find whether each line is at one of the valid locations
        consistent_lines = np.argwhere(np.isin(locations_1D, consistent_locations))

        return consistent_lines.flatten()

    @staticmethod
    def stacked_bin_index(
            i_row=0,
            i_flux=0,
            n_flux_bins=1,
            i_date=0,
            n_date_bins=1,
            i_background=0,
            n_background_bins=1,
    ):
        """
        Return the index for the 1D ordering of stacked lines in bins, given the
        index and number of each bin.

        See generate_stacked_lines_from_bins().
        """
        return int(
            i_row * n_flux_bins * n_date_bins * n_background_bins
            + i_flux * n_date_bins * n_background_bins
            + i_date * n_background_bins
            + i_background
        )

    def row_bins(self, n_bins=ut.n_row_bins):
        return Bins.from_values(
            self.locations[:, 0],
            n_bins=n_bins,
        )

    def flux_bins(self, n_bins=ut.n_flux_bins):
        return Bins.from_values(
            self.fluxes[self.fluxes > 0],
            n_bins=n_bins,
            scale="log",
        )

    def date_bins(self, n_bins=1):
        return Bins.from_values(
            self.dates,
            n_bins=n_bins,
        )

    def background_bins(self, n_bins=ut.n_background_bins):
        return Bins.from_values(
            self.backgrounds,
            n_bins=n_bins,
        )

    def generate_stacked_lines_from_bins(
            self,
            row_bins=None,
            flux_bins=None,
            date_bins=None,
            background_bins=None,
    ):
        """Create a collection of stacked lines by averaging within bins.

        The following metadata variables must be set for all lines: data,
        location, date, background, and flux. Set n_*_bins=1 (default) to ignore
        any of these variables for the actual binning, but their values must
        still be set, i.e. not None.

        Lines should all be the same length.

        The full bin edge values may be provided, or instead the number and
        limits of the bins. Bin minima and maxima default to the extremes of the
        lines' values with, by default, logarithmic spacing for the flux bins
        and linear for the others.

        Lines with values outside of the bin minima or maxima are discarded.

        Parameters
        ----------
        row_bins : [float]
            The edge values of the bins for the rows, i.e. distance from the
            readout register (minus one). If provided, this overrides the other
            bin inputs to allow for uneven bin spacings, for example. If this is
            None (default), the other inputs are used to create it.

        flux_bins
            As above, for the bins by flux.

        date_bins
            As above, for the bins by Julian date.

        background_bins
            As above, for the bins by background.

        Returns
        -------
        stacked_lines : StackedPixelLineCollection
            A new collection of the stacked pixel lines, including errors.
            Metadata parameters contain the lower edge bin value.

            The following extra parameters are added to each PixelLine object,
            containing the mean and rms of the parameters in each bin: mean_row,
            rms_row, mean_background, rms_background, mean_flux, rms_flux.
        """
        from .stacked_collection import StackedPixelLineCollection

        row_bins = row_bins or self.row_bins()
        flux_bins = flux_bins or self.flux_bins()
        date_bins = date_bins or self.date_bins()
        background_bins = background_bins or self.background_bins()

        collection = StackedPixelLineCollection(
            length=self.lengths[0],
            row_bins=row_bins,
            flux_bins=flux_bins,
            date_bins=date_bins,
            background_bins=background_bins,
        )

        # Line length
        length = self.lengths[0]
        assert all(self.lengths == length)

        # Add the line data to each stack
        #
        # RJM: Does this loop over each line too many times? Should only need to look at each line once
        #
        for line in self.lines:
            # Discard lines with values outside of the bins

            try:
                collection.add_line(line)
            except IndexError:
                continue

        return collection


class PixelLineCollection(AbstractPixelLineCollection):
    def __init__(self, lines=None):
        """A collection of 1D lines of pixels with metadata.

        Enables convenient analysis e.g. binning and stacking of CTI trails.

        Parameters
        ----------
        lines : [PixelLine]
            A list of the PixelLine objects.
        """
        if lines is None:
            lines = []
        self._lines = lines

    @property
    def lines(self):
        return self._lines

    def extend(self, new_lines):
        if isinstance(
                new_lines,
                PixelLineCollection
        ):
            new_lines = new_lines.lines
        self._lines.extend(new_lines)

    def append(self, new_line):
        self._lines.append(new_line)
