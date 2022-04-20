import numpy as np
from autoarray.instruments.acs import ImageACS

from warm_pixels import hst_utilities as ut
from warm_pixels.data import Dataset, Image, CorrectedDataset
from warm_pixels.pixel_lines import PixelLine, PixelLineCollection
from warm_pixels.warm_pixels import find_warm_pixels
from .cache import cache
from .cache import persist


class ImageQuadrant:
    def __init__(
            self,
            quadrant: str,
            image: Image,
    ):
        self.quadrant = quadrant
        self.image = image

    @cache
    def array(self):
        self.image._check_bia_exists()
        return ImageACS.from_fits(
            file_path=str(self.image.path),
            quadrant_letter=self.quadrant,
            bias_subtract_via_bias_file=True,
            bias_subtract_via_prescan=True,
        ).native

    @property
    def name(self):
        return f"{self.image.name}_{self.quadrant}"

    def __str__(self):
        return self.name

    def warm_pixels(self):
        return find_warm_pixels(
            image=self.array(),
            trail_length=ut.trail_length,
            n_parallel_overscan=20,
            n_serial_prescan=24,
            origin=self.name,
            date=self.image.date(),
        )


def directory_func(
        quadrant
):
    return f"{quadrant.dataset}/{quadrant}"


class DatasetQuadrant:
    def __new__(
            cls,
            quadrant: str,
            dataset: Dataset,
    ):
        if isinstance(
                dataset,
                CorrectedDataset
        ):
            return object.__new__(CorrectedDatasetQuadrant)
        return object.__new__(DatasetQuadrant)

    def __init__(
            self,
            quadrant: str,
            dataset: Dataset
    ):
        """
        Computes warm pixels and lines across a given CCD quadrant for a
        dataset.

        Parameters
        ----------
        quadrant
            The name of a quadrant (A, B, C or D)
        dataset
            A dataset with images from a given date
        """
        self.quadrant = quadrant
        self.dataset = dataset
        self.image_quadrants = [
            ImageQuadrant(
                quadrant=quadrant,
                image=image,
            )
            for image in self.dataset
        ]

    def __str__(self):
        return self.quadrant

    @cache
    def warm_pixels(self) -> PixelLineCollection:
        """
        A collection of warm pixels found in images in the dataset.

        This is where a single pixel is much brighter than surrounding pixels indicating
        that it is not due to a true light source in an image.
        """
        warm_pixels = PixelLineCollection()

        # Find the warm pixels in each image
        for image_quadrant in self.image_quadrants:
            # Add them to the collection
            warm_pixels.append(image_quadrant.warm_pixels())

        return warm_pixels

    @persist(directory_func)
    def consistent_lines(self) -> PixelLineCollection:
        """
        Warm pixels that are consistent across multiple images and therefore highly unlikely
        to have been caused by true light sources.
        """
        return self.warm_pixels().consistent()


class CorrectedDatasetQuadrant(DatasetQuadrant):
    @persist(directory_func)
    def consistent_lines(self):
        """Extract the corresponding warm pixels from the corrected images with CTI
        removed, in the same locations as the orignal consistent warm pixels.
        """
        warm_pixels = super().consistent_lines()

        # Corrected images
        warm_pixels_cor = PixelLineCollection()
        for i, image in enumerate(self.dataset):
            image_name = image.name
            print(
                f"\r    {image_name}_cor_{self.quadrant} ({i + 1} of {len(self.dataset)}) ",
                end="",
                flush=True,
            )

            # Load the image
            array = image.quadrant(self.quadrant).array()

            # Select consistent warm pixels found from this image
            image_name_q = f"{image_name}_{self.quadrant}"
            sel = np.where(warm_pixels.origins == image_name_q)[0]
            for j in sel:
                line = warm_pixels.lines[j]
                row, column = line.location

                # Copy the original metadata but take the data from the corrected image
                warm_pixels_cor.append(
                    PixelLine(
                        data=array[
                             row - ut.trail_length: row + ut.trail_length + 1, column
                             ],
                        origin=line.origin,
                        location=line.location,
                        date=line.date,
                        background=line.background,
                    )
                )

        print(f"Extracted {warm_pixels_cor.n_lines} lines")

        return warm_pixels_cor
