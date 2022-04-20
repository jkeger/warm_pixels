"""Defines where the HST data is located on disc, and a class to read it in/contain it"""
import os
from glob import glob
from pathlib import Path

import autoarray as aa
from autoarray.instruments.acs import ImageACS

import arcticpy as cti
from warm_pixels.hst_functions.cti_model import cti_model_hst
from warm_pixels.model.cache import cache
from .image import Image


class Dataset:
    def __init__(
            self,
            path: Path,
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
        self._images = None
        self._quadrants = {}

    def groups(self, quadrants_string):
        return [
            self.group(group)
            for group in tuple(map(
                tuple,
                quadrants_string.split("_")
            ))
        ]

    def group(self, quadrants_string):
        from warm_pixels.model.group import QuadrantGroup
        return QuadrantGroup(list(
            map(self.quadrant, quadrants_string)
        ))

    def quadrant(self, quadrant):
        if quadrant not in self._quadrants:
            from warm_pixels.model.quadrant import DatasetQuadrant
            self._quadrants[quadrant] = DatasetQuadrant(
                quadrant=quadrant,
                dataset=self
            )
        return self._quadrants[quadrant]

    @property
    @cache
    def images(self):
        return [
            Image(
                Path(image_path),
            )
            for image_path in glob(
                f"{self.path}/*_raw.fits"
            )
        ]

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        return iter(self.images)

    def __getitem__(self, item):
        return self.images[item]

    @property
    def name(self):
        return self.path.name

    def __str__(self):
        return self.name

    def observation_date(self):
        return self.images[0].observation_date()

    @property
    def date(self):
        """Return the Julian date of the set, taken from the first image."""
        return self.images[0].date()

    def corrected(self):
        """
        Remove CTI trails using arctic from all images in the dataset.
        """
        corrected_dataset = CorrectedDataset(self)
        os.makedirs(
            corrected_dataset.path,
            exist_ok=True,
        )

        # Remove CTI from each image
        for i, image in enumerate(self):
            image_name = image.path.name
            image_path = corrected_dataset.path / image_name

            if image_path.exists():
                print(f"{image_path} already exists")
                continue

            print(
                f"  Correcting {image_name} ({i + 1} of {self})... ",
                end="",
                flush=True,
            )

            # CTI model
            date = image.date()
            roe, ccd, traps = cti_model_hst(date)

            corrected_quadrants = []

            for quadrant in image.quadrants():
                corrected_quadrant = ImageACS(
                    cti.remove_cti(
                        image=quadrant,
                        n_iterations=5,
                        parallel_roe=roe,
                        parallel_ccd=ccd,
                        parallel_traps=traps,
                        parallel_express=5
                    ),
                    mask=quadrant.mask,
                    header=quadrant.header,
                )
                corrected_quadrants.append(corrected_quadrant)

            # Save the corrected image
            aa.acs.output_quadrants_to_fits(
                image_path,
                *corrected_quadrants,
                overwrite=True,
            )

            print(f"Saved {image.corrected().path.stem}")

        return corrected_dataset


class CorrectedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        corrected_path = dataset.path / "corrected"
        super().__init__(
            path=corrected_path,
        )

    @property
    def output_path(self):
        return self.path

    @property
    def name(self):
        return f"{self.path.parent.name}_corrected"
