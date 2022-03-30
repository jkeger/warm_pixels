"""Defines where the HST data is located on disc, and a class to read it in/contain it"""
import datetime as dt
from pathlib import Path

import autoarray as aa
from autoarray.instruments.acs import ImageACS


# https://hst-crds.stsci.edu/unchecked_get/references/hst/58518491j_bia.fits


class Image:
    def __init__(
            self,
            path: Path,
    ):
        self.path = path

    @property
    def name(self):
        return self.path.name.split("_")[0]

    @property
    def bia_path(self):
        return self.path.parent / self.path.name.replace("_raw", "_bia")

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

    def quadrants(self):
        for quadrant in ["A", "B", "C", "D"]:
            yield self.load_quadrant(quadrant)

    def date(self):
        return 2400000.5 + self.image().header.modified_julian_date

    def observation_date(self) -> dt.date:
        """
        The date of observation
        """
        return dt.date.fromisoformat(
            self.image().header.date_of_observation
        )

    def corrected(self):
        return CorrectedImage(self)


class CorrectedImage(Image):
    def __init__(self, image):
        super().__init__(
            path=(
                    image.path.parent
                    / f"{image.name}_raw_cor.fits"
            ),
        )
        self.image = image
