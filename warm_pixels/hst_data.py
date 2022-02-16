"""Defines where the HST data is located on disc, and a class to read it in/contain it"""
import os
from glob import glob
from pathlib import Path

import autoarray as aa
from autoarray.instruments.acs import ImageACS

import arcticpy as cti
from warm_pixels import hst_utilities as ut
from warm_pixels.hst_functions.cti_model import cti_model_hst


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

    def corrected(self):
        return CorrectedImage(self)


class CorrectedImage(Image):
    def __init__(self, image):
        super().__init__(
            path=(
                    image.path.parent
                    / f"{image.name}_raw_cor.fits"
            ),
            output_path=image.path,
        )
        self.image = image


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
        self._images = None

    @property
    def images(self):
        return [
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

    def saved_consistent_lines(self, quadrant):
        """Return the file name including the path for saving derived data."""
        return self.output_path / f"saved_consistent_lines_{quadrant}.pickle"

    def saved_stacked_lines(self, quadrants):
        """Return the file name including the path for saving derived data."""
        quadrant_string = "".join(quadrants)
        return self.output_path / f"saved_stacked_lines_{quadrant_string}.pickle"

    def saved_stacked_info(self, quadrants):
        """Return the file name including the path for saving derived data."""
        quadrant_string = "".join(quadrants)
        return self.output_path / f"saved_stacked_info_{quadrant_string}.npz"

    def plotted_stacked_trails(self, quadrants):
        """Return the file name including the path for saving derived data."""
        return ut.output_path / f"stacked_trail_plots/{self.name}_plotted_stacked_trails_{''.join(quadrants)}.png"

    def plotted_distributions(self, quadrants):
        """Return the file name including the path for saving derived data."""
        return ut.output_path / f"plotted_distributions/{self.name}_plotted_distributions_{''.join(quadrants)}.png"

    def corrected(self):
        """Remove CTI trails using arctic from all images in the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset object with a list of image file paths and metadata.

        Saves
        -----
        dataset.cor_paths
            The corrected images with CTI removed in the same location as the
            originals.
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
                corrected_quadrant = cti.remove_cti(
                    image=quadrant,
                    n_iterations=5,
                    parallel_roe=roe,
                    parallel_ccd=ccd,
                    parallel_traps=traps,
                    parallel_express=5
                )
                corrected_quadrant.header = quadrant.header
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
        corrected_path = dataset.output_path.parent / f"{dataset.output_path.name}_corrected"
        super().__init__(
            path=corrected_path,
            output_path=dataset.output_path,
        )

    @property
    def output_path(self):
        return self.path
