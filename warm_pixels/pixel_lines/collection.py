import pickle

import numpy as np

from warm_pixels import hst_utilities as ut
from .pixel_line import PixelLine
from ..bins import Bins


class PixelLineCollection:
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
            self.lines[sel_flux]
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
        collection.append(self.lines)
        if isinstance(
                other, PixelLineCollection
        ):
            other = other.lines
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
        return np.array([line.data for line in self._lines])

    @property
    def origins(self):
        """
        The identifiers for the origins (e.g. image name) of each line.
        """
        return np.array([line.origin for line in self._lines])

    @property
    def locations(self):
        """
        The row and column indices of the first pixel in the line in the
            image, for each line.
        """
        return np.array([line.location for line in self._lines])

    @property
    def dates(self):
        """
        The Julian date of each line.
        """
        return np.array([line.date for line in self._lines])

    @property
    def backgrounds(self):
        """
        The background charge count of each line, in units of electrons.
        """
        return np.array([line.background for line in self._lines])

    @property
    def fluxes(self):
        """
        The maximum charge in each line, in units of electrons.
        """
        return np.array([line.flux for line in self._lines])

    @property
    def lengths(self):
        """
        The number of pixels in the data array of each line.
        """
        return np.array([line.length for line in self._lines])

    @property
    def n_lines(self):
        """
        The number of lines in the collection.
        """
        return len(self._lines)

    @property
    def lines(self):
        return np.array(self._lines)

    def append(self, new_lines):
        if isinstance(
                new_lines,
                PixelLineCollection
        ):
            new_lines = new_lines.lines
        if isinstance(
                new_lines,
                (list, np.ndarray)
        ):
            self._lines.extend(new_lines)
        else:
            self._lines.append(new_lines)

    def save(self, filename):
        """ Save the lines data. """
        # Check the file extension
        filename = str(filename)
        if filename[-7:] != ".pickle":
            filename += ".pickle"

        # Save the lines
        with open(filename, "wb") as f:
            pickle.dump(self.lines, f)

    @classmethod
    def load(cls, filename):
        """ Load and append lines that were previously saved. """
        # Check the file extension
        filename = str(filename)
        if filename[-7:] != ".pickle":
            filename += ".pickle"

        # Load the lines
        with open(filename, "rb") as f:
            return PixelLineCollection(pickle.load(f))

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

        n_row_bins : int
            The number of row bins, if row_bins is not provided.

        row_min : float
            The minimum value for the row bins, if row_bins is not provided.

        row_max : float
            The maximum value for the row bins, if row_bins is not provided.

        row_scale : str
            The spacing (linear or logarithmic) for the row bins, if row_bins is
            not provided.

        flux_bins, n_flux_bins, flux_min, flux_max, flux_scale
            As above, for the bins by flux.

        date_bins, n_date_bins, date_min, date_max, date_scale
            As above, for the bins by Julian date.

        background_bins, n_background_bins, background_min, background_max,
        background_scale : [float], int, float, float, str
            As above, for the bins by background.
        n_background_bins
        background_min
        background_max
        background_bins

        Returns
        -------
        stacked_lines : StackedPixelLineCollection
            A new collection of the stacked pixel lines, including errors.
            Metadata parameters contain the lower edge bin value.

            The following extra parameters are added to each PixelLine object,
            containing the mean and rms of the parameters in each bin: mean_row,
            rms_row, mean_background, rms_background, mean_flux, rms_flux.
        """
        row_bins = row_bins or self.row_bins()
        flux_bins = flux_bins or self.flux_bins()
        date_bins = date_bins or self.date_bins()
        background_bins = background_bins or self.background_bins()

        # Line length
        length = self.lengths[0]
        assert all(self.lengths == length)

        # Find the bin indices for each parameter for each line
        if row_bins.is_single:
            row_indices = np.zeros(self.n_lines)
        else:
            row_indices = row_bins.indices(self.locations[:, 0])
        if flux_bins.is_single:
            flux_indices = np.zeros(self.n_lines)
        else:
            flux_indices = flux_bins.indices(self.fluxes)
        if date_bins.is_single:
            date_indices = np.zeros(self.n_lines)
        else:
            date_indices = date_bins.indices(self.dates)
        if background_bins.is_single:
            background_indices = np.zeros(self.n_lines)
        else:
            background_indices = background_bins.indices(self.backgrounds)

        # Initialise the array of empty lines in each bin, as a long 1D array
        stacked_lines = [
            PixelLine(
                data=[np.zeros(length)],
                location=[row, 0],
                date=date,
                background=background,
                flux=flux,
                n_stacked=0,
            )
            for row in row_bins[:-1]
            for date in date_bins[:-1]
            for background in background_bins[:-1]
            for flux in flux_bins[:-1]
        ]

        # Initialise sums of other parameters and other parameters squared
        n_bins = row_bins.number * flux_bins.number * date_bins.number * background_bins.number
        sum_rows = np.zeros(n_bins)
        sum_backgrounds = np.zeros(n_bins)
        sum_fluxes = np.zeros(n_bins)
        sum_sq_rows = np.zeros(n_bins)
        sum_sq_backgrounds = np.zeros(n_bins)
        sum_sq_fluxes = np.zeros(n_bins)

        # Add the line data to each stack
        #
        # RJM: Does this loop over each line too many times? Should only need to look at each line once
        #
        for i_row, i_flux, i_date, i_background, line in zip(
                row_indices, flux_indices, date_indices, background_indices, self.lines
        ):
            # Discard lines with values outside of the bins
            # print(i_row, i_flux, i_date, i_background)
            if -1 in [i_row, i_flux, i_date, i_background]:
                continue

            # Get the index in the 1D array for this bin
            index = self.stacked_bin_index(
                i_row=i_row,
                i_flux=i_flux,
                n_flux_bins=flux_bins.number,
                i_date=i_date,
                n_date_bins=date_bins.number,
                i_background=i_background,
                n_background_bins=background_bins.number,
            )

            #
            # RJM: This is REALLY slow! It loops over all warm pixels, and appends them to bins, rather than just adding them.
            #      But it does avoid floating point errors when adding LOTS of small numbers.
            #
            # Append the line data
            if stacked_lines[index].n_stacked == 0:
                stacked_lines[index].data = [line.data]
            else:
                stacked_lines[index].data.append(line.data)
            stacked_lines[index].n_stacked += 1

            # Append the other parameters
            sum_rows[index] += line.location[0]
            sum_backgrounds[index] += line.background
            sum_fluxes[index] += line.flux
            sum_sq_rows[index] += line.location[0] ** 2
            sum_sq_backgrounds[index] += line.background ** 2
            sum_sq_fluxes[index] += line.flux ** 2

        for sl in stacked_lines:
            sl.data = np.stack(sl.data)

        # Take the means and standard errors
        #
        # RJM: adding variables like this, which are not defined in the general class descriptor, seems a bit dodgy
        #
        for index, line in enumerate(stacked_lines):
            if line.n_stacked > 0:
                line.noise = np.std(line.data, axis=0) / np.sqrt(line.n_stacked)

                line.mean_row = sum_rows[index] / line.n_stacked
                line.rms_row = np.sqrt(sum_sq_rows[index] / line.n_stacked)
                line.mean_background = sum_backgrounds[index] / line.n_stacked
                line.rms_background = np.sqrt(
                    sum_sq_backgrounds[index] / line.n_stacked
                )
                line.mean_flux = sum_fluxes[index] / line.n_stacked
                line.rms_flux = np.sqrt(sum_sq_fluxes[index] / line.n_stacked)
            else:
                line.noise = np.zeros(length)

                line.mean_row = None
                line.rms_row = None
                line.mean_background = None
                line.rms_background = None
                line.mean_flux = None
                line.rms_flux = None

            line.data = np.mean(line.data, axis=0)

        from .stacked_collection import StackedPixelLineCollection

        return StackedPixelLineCollection(
            lines=stacked_lines,
            row_bins=row_bins,
            flux_bins=flux_bins,
            date_bins=date_bins,
            background_bins=background_bins,
        )
