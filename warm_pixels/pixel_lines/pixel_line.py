from abc import ABC, abstractmethod
from typing import Tuple, Optional

import numpy as np


def _dump(value):
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return float(value)
        return list(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.number):
        return float(value)
    if isinstance(value, list):
        return list(map(_dump, value))
    if isinstance(value, dict):
        return {
            key: _dump(value)
            for key, value
            in value.items()
        }
    return value


class AbstractPixelLine(ABC):
    def __init__(
            self,
            location: Optional[Tuple[int, int]] = None,
            date: Optional[float] = None,
            background: Optional[float] = None,
            flux: Optional[float] = None,
    ):
        """
        Represents a trail

        Parameters
        ----------
        location
            The location of the warm pixel in (row, column)
        date
            The date on which the image was captured
        background
            Background sky in the original image
        flux
            The estimated flux of the warm pixel before trailing
        """
        self.location = location
        self.date = date
        self.background = background
        self._flux = flux

    @property
    def dict(self) -> dict:
        """
        A dictionary representation of the pixel line. This can
        be used to create a Dataset1D in autocti.
        """
        d = {
            "location": self.location,
            "date": self.date,
            "background": self.background,
            "flux": self.flux,
            "data": self.data,
            "noise": self.noise,
        }
        return _dump(d)

    @property
    def flux(self):
        # Default flux from data
        if self._flux is None and self.data is not None:
            self._flux = np.amax(self.data)
        return self._flux

    @property
    @abstractmethod
    def data(self):
        pass

    @property
    @abstractmethod
    def noise(self):
        pass

    @property
    @abstractmethod
    def length(self):
        pass

    @property
    def trail_length(self):
        """Number of pixels in only the trailed section of the data array
        (which is assumed to be N preceding pixels, 1 warm pixel, N trailed pixels)"""
        assert (self.length % 2) == 1
        return (self.length - 1) // 2

    @property
    def model_background(self, n_pixels_used_for_background=12):
        """Re-estimate the background, locally"""
        if self.data is None:
            return None
        n_pixels_used_for_background = min(n_pixels_used_for_background, self.trail_length)
        return np.mean(self.data[: n_pixels_used_for_background])

    @property
    def model_flux(self):
        """Extract just the number of electrons in the warm pixel, locally
        add_trail = True will push any trailed electrons back into the warm pixel"""
        if self.data is None:
            return None
        return self.data[-self.trail_length - 1]

    @property
    def model_trail(self):
        """Convert a line with a warm pixel in the middle to just the trail (without background),
        suitable for modelling as sums of exponentials.
        """
        # Subtract preceding pixels, as a way of removing spurious sources (and the constant background)
        return self.data[-self.trail_length:] - np.flip(self.data[: self.trail_length])

    @property
    def model_trail_noise(self):
        """Convert a line with a warm pixel in the middle to just the trail (without background),
        suitable for modelling as sums of exponentials.
        """
        # Add noise from preceding and trailed pixels in quadrature
        return np.sqrt(self.noise[-self.trail_length:] ** 2 + np.flip(self.noise[:self.trail_length]) ** 2)

    @property
    def model_full_trail_length(self):
        """Determine the length of array needed to model the trail as enough of a full column to
        conveniently pass to arCTIc, and not then need to use windows.
        """
        # RJM: do we also need to shift the location by length pixels?
        return np.int(np.floor(self.mean_row) + 1 + self.trail_length)

    @property
    def model_full_trail(self):
        """Convert a line with a warm pixel in the middle to the entire relevant part of a column
        (with background), suitable for passing to arCTIc.
        """

        # Constant background level
        full_trail = np.full(self.model_full_trail_length, self.model_background)
        # Add warm pixel itself
        full_trail[-self.trail_length - 1] = self.model_flux
        # Add trail
        full_trail[-self.trail_length:] += self.model_trail

        return full_trail

    @property
    def model_full_trail_untrailed(self):
        """Convert a line with a warm pixel in the middle to the entire relevant part of a column
        (with background), suitable for passing to arCTIc.
        Ideally wouldn't ever use this, but would iterate to find this during fitting. This is
        because the pushing back of trailed electrons into the warm pixel is noisy and truncated
        (any trailed electrons past the original line.data have been lost, creating a biased
        underestimate).
        """

        # Constant background level
        full_trail_untrailed = np.full(self.model_full_trail_length, self.model_background)
        # Add warm pixel itself
        full_trail_untrailed[-self.trail_length - 1] = self.model_flux + sum(self.model_trail)

        return full_trail_untrailed
    
    @property
    def model_full_trail_untrailed_abs(self):
        """Convert a line with a warm pixel in the middle to the entire relevant part of a column
        (with background), suitable for passing to arCTIc.
        Ideally wouldn't ever use this, but would iterate to find this during fitting. This is
        because the pushing back of trailed electrons into the warm pixel is noisy and truncated
        (any trailed electrons past the original line.data have been lost, creating a biased
        underestimate).
        """

        # Constant background level
        full_trail_untrailed = np.full(self.model_full_trail_length, self.model_background)
        # Add warm pixel itself
        full_trail_untrailed[-self.trail_length - 1] = self.model_flux + abs(sum(self.model_trail))

        return full_trail_untrailed

    @property
    def model_full_trail_noise(self):
        """Noise model for the entire relevant part of a column, suitable for passing to arCTIc.
        """

        # Constant background level
        # full_trail_noise = np.full(self.model_full_trail_length, np.sqrt(self.model_background))
        full_trail_noise = np.full(self.model_full_trail_length, np.inf)  # downweight in fit
        # Add warm pixel itself
        # full_trail_noise[-self.trail_length - 1] = self.noise[self.trail_length]
        # Add trail
        full_trail_noise[-self.trail_length:] = self.model_trail_noise

        return full_trail_noise


class PixelLine(AbstractPixelLine):
    def __init__(
            self,
            data=None,
            noise=None,
            origin=None,
            location=None,
            date=None,
            background=None,
            flux=None,
    ):
        """A 1D line of pixels (e.g. a single CTI trail) with metadata.

        Or could be an averaged stack of many lines, in which case the metadata
        parameters may be e.g. the average value or the minimum value of a bin.

        Parameters
        ----------
        data : [float]
            The pixel counts, in units of electrons.

        noise : [float]
            The noise errors on the pixel counts, in units of electrons.

        origin : str
            An identifier for the origin (e.g. image name) of the data.

        location : [int, int]
            The row and column indices of the first pixel in the line in the
            image. The row index is the distance in pixels to the readout
            register minus one.

        date : float
            The Julian date.

        background : float
            The background charge count, in units of electrons. It is assumed
            that the background has not been subtracted from the data.

        flux : float
            The maximum charge in the line, or e.g. for a CTI trail the original
            flux before trailing, in units of electrons.

        """
        self._data = data
        super().__init__(location, date, background, flux)
        self._noise = noise
        self.origin = origin
        self.format = None

    @classmethod
    def from_dict(cls, pixel_line_dict):
        return cls(
            location=pixel_line_dict["location"],
            date=pixel_line_dict["date"],
            background=pixel_line_dict["background"],
            flux=pixel_line_dict["flux"],
            data=pixel_line_dict["data"],
            noise=np.array(pixel_line_dict["noise"]),
        )

    @property
    def length(self):
        """Number of pixels in the data array"""
        if self.data is not None:
            return len(self.data)
        return None

    @property
    def noise(self):
        return self._noise

    @property
    def data(self):
        return self._data
