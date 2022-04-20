"""Defines where the HST data is located on disc, and a class to read it in/contain it"""
import datetime as dt
from pathlib import Path

import autoarray as aa
import requests
from autoarray.structures.arrays.two_d.array_2d_util import header_obj_from

from warm_pixels.model.cache import cache, CachedObject

HST_DATA_URL = "https://hst-crds.stsci.edu/unchecked_get/references/hst"

DAY_ZERO = dt.date(2002, 3, 1)


class Image(CachedObject):
    def __init__(
            self,
            path: Path,
    ):
        self.path = path
        self._quadrants = {}

    @property
    def name(self):
        return self.path.name.split("_")[0]

    @property
    @cache
    def bia_path(self):
        return self.path.parent / header_obj_from(
            str(self.path), 0
        )["BIASFILE"].replace("jref$", "")

    def _check_bia_exists(self):
        if not self.bia_path.exists():
            response = requests.get(
                f"{HST_DATA_URL}/{self.bia_path.name}"
            )
            response.raise_for_status()
            with open(self.bia_path, "w+b") as f:
                f.write(response.content)

    def image(self):
        self._check_bia_exists()
        return aa.acs.ImageACS.from_fits(
            file_path=str(self.path),
            quadrant_letter="A"
        )

    def quadrant(self, item):
        if item not in self._quadrants:
            from warm_pixels.model.quadrant import ImageQuadrant
            self._quadrants[item] = ImageQuadrant(
                item, self,
            )
        return self._quadrants[item]

    def quadrants(self):
        for quadrant in ["A", "B", "C", "D"]:
            yield self.quadrant(quadrant)

    def date(self):
        return 2400000.5 + self.image().header.modified_julian_date

    def days_since_launch(self):
        return (self.observation_date() - DAY_ZERO).days

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
