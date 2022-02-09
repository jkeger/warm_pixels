"""Defines where the HST data is located on disc, and a class to read it in/contain it"""
import os
from glob import glob
from pathlib import Path

import autoarray as aa
from autoarray.instruments.acs import ImageACS

from warm_pixels import hst_utilities as ut


class Image:
    def __init__(
            self,
            path: Path,
            output_path: Path
    ):
        self.path = path
        self._output_path = output_path

    @property
    def name(self):
        return self.path.name.split("_")[0]

    @property
    def output_path(self):
        output_path = self._output_path / self.name
        os.makedirs(
            output_path,
            exist_ok=True
        )
        return output_path

    @property
    def cor_path(self):
        return self.path.parent / f"{self.name}_raw_cor.fits"

    def image(self):
        return aa.acs.ImageACS.from_fits(
            file_path=str(self.path),
            quadrant_letter="A"
        )

    def load_quadrant(self, quadrant):
        return ImageACS.from_fits(
            file_path=str(self.path),
            quadrant_letter=quadrant,
            bias_subtract_via_bias_file=True,
            bias_subtract_via_prescan=True,
        ).native

    def date(self):
        return 2400000.5 + self.image().header.modified_julian_date


class Dataset:
    def __init__(
            self,
            path: Path,
            output_path: Path
    ):
        """Simple class to store a list of image file paths and mild metadata.

        Parameters
        ----------
        path
            The path to a directory containing fits files

        Attributes
        ----------
        path : str
            File path to the dataset directory.
        """
        self.path = path
        self._output_path = output_path
        self.images = [
            Image(
                Path(image_path),
                output_path=self.output_path,
            )
            for image_path in glob(
                f"{self.path}/*_raw.fits"
            )
        ]

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        return iter(self.images)

    @property
    def name(self):
        return self.path.name

    @property
    def output_path(self):
        output_path = self._output_path / self.name
        os.makedirs(
            output_path,
            exist_ok=True,
        )
        return output_path

    @property
    def date(self):
        """Return the Julian date of the set, taken from the first image."""
        return self.images[0].date()

    # ========
    # File paths for saved data, including the quadrant(s)
    # ========
    def saved_lines(self, quadrant):
        """Return the file name including the path for saving derived data."""
        return self.output_path / f"saved_lines_{quadrant}.pickle"

    def saved_consistent_lines(self, quadrant, use_corrected=False):
        """Return the file name including the path for saving derived data."""
        suffix = "_cor" if use_corrected else ""
        return self.output_path / f"saved_consistent_lines_{quadrant}{suffix}.pickle"

    def saved_stacked_lines(self, quadrants, use_corrected=False):
        """Return the file name including the path for saving derived data."""
        suffix = "_cor" if use_corrected else ""
        quadrant_string = "".join(quadrants)
        return self.output_path / f"saved_stacked_lines_{quadrant_string}{suffix}.pickle"

    def saved_stacked_info(self, quadrants, use_corrected=False):
        """Return the file name including the path for saving derived data."""
        suffix = "_cor" if use_corrected else ""
        quadrant_string = "".join(quadrants)
        return self.output_path / f"saved_stacked_info_{quadrant_string}{suffix}.npz"

    def plotted_stacked_trails(self, quadrants, use_corrected=False):
        """Return the file name including the path for saving derived data."""
        suffix = "_cor" if use_corrected else ""
        return ut.output_path / f"stacked_trail_plots/{self.name}_plotted_stacked_trails_{''.join(quadrants)}{suffix}.png"

    def plotted_distributions(self, quadrants):
        """Return the file name including the path for saving derived data."""
        return ut.output_path / f"plotted_distributions/{self.name}_plotted_distributions_{''.join(quadrants)}.png"
