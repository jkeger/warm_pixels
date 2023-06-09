"""Defines where the HST data is located on disc, and a class to read it in/contain it"""
import logging
import os
from glob import glob
from pathlib import Path
from typing import Tuple, List
import arcticpy as arctic
import autoarray as aa
from autoarray.instruments.acs import ImageACS

import arcticpy as cti
#from warm_pixels.hst_functions.cti_model import cti_model_hst
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
        self.logger = logging.getLogger(self.name)

    def groups(self, quadrants_string: str) -> list:
        """
        Create or retrieve a list of groups. Each group corresponds
        to one or more quadrants defined in the quadrant string. For
        example, AB_CD would give two groups: AB and CD.

        Parameters
        ----------
        quadrants_string
            A string defining which quadrants are processed and how
            they are grouped.

        Returns
        -------
        A list of groups
        """
        self.logger.info("Retrieving groups")
        return [
            self.group(group)
            for group in tuple(map(
                tuple,
                quadrants_string.split("_")
            ))
        ]

    def group(
            self,
            quadrants: Tuple[str]
    ):
        """
        Create a group for a collection of quadrants

        Parameters
        ----------
        quadrants
            A tuple with letters defining which quadrants should be included
            int he group

        Returns
        -------
        A group of quadrants
        """
        from warm_pixels.model.group import QuadrantGroup
        return QuadrantGroup(list(
            map(self.quadrant, quadrants)
        ))

    def quadrant(self, quadrant: str):
        """
        Retrieve or create an object representing a particular quadrant
        for a particular dataset.

        Parameters
        ----------
        quadrant
            The name of a quadrant

        Returns
        -------
        An object comprising a quadrant and this dataset
        """
        self.logger.debug(
            f"Recovering dataset quadrant for quadrant {quadrant}..."
        )
        if quadrant not in self._quadrants:
            self.logger.debug(
                f"Not found. Creating..."
            )
            from warm_pixels.model.quadrant import DatasetQuadrant
            self._quadrants[quadrant] = DatasetQuadrant(
                quadrant=quadrant,
                dataset=self
            )
        return self._quadrants[quadrant]

    def days_since_launch(self) -> int:
        """
        The number of days since the start of the mission that images from this
        dataset were captured.

        Assumes all images were captured on the same date and simply uses the
        date of the first image.
        """
        return self.images[0].days_since_launch()

    @property
    @cache
    def images(self) -> List[Image]:
        """
        Images found in this dataset. Images for a dataset are stored in the same
        directory and are assumed to have been take on the same or a similar date
        meaning that the CTI properties will be similar.
        """
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

    @cache
    def observation_date(self):
        return self.images[0].observation_date()

    @property
    @cache
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
            # These are the parameters for the model with N
            rho_q = 1.842773381
            a = 0.166732406
            b = 0.831794543
            c = 0.00147305
            tau_a = 0.51062611
            tau_b = 5.163915449
            tau_c = 44.8384793
            density_a=a*rho_q
            density_b=b*rho_q
            density_c=c*rho_q
            
            date = image.date()
            roe = arctic.ROE()
            ccd = arctic.CCD(full_well_depth=84700, well_fill_power=0.432785976)
            traps = [
                 arctic.TrapInstantCapture(density=density_a, release_timescale=tau_a),
                 arctic.TrapInstantCapture(density=density_b, release_timescale=tau_b),
                 arctic.TrapInstantCapture(density=density_c, release_timescale=tau_c),
             ]
            corrected_quadrants = []

            for quadrant in image.quadrants():
                quadrant = quadrant.array()
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
        corrected_path = dataset.path / "Paolo_corrected"
        super().__init__(
            path=corrected_path,
        )

    @property
    def output_path(self):
        return self.path

    @property
    def name(self):
        return f"{self.path.parent.name}_Paolo_corrected"
