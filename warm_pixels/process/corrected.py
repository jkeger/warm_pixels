import numpy as np

from warm_pixels import hst_utilities as ut
from warm_pixels.pixel_lines import PixelLine, PixelLineCollection
from .abstract import AbstractProcess


class CorrectedProcess(AbstractProcess):
    def __init__(
            self,
            raw_process
    ):
        super().__init__(
            dataset=raw_process.dataset.corrected(),
            overwrite=raw_process.overwrite,
            quadrants=raw_process.quadrants,
        )
        self.raw_process = raw_process

    def consistent_lines_for(self, quadrant):
        """Extract the corresponding warm pixels from the corrected images with CTI
        removed, in the same locations as the orignal consistent warm pixels.

        Parameters
        ----------
        quadrant : str (opt.)
            The quadrant (A, B, C, D) of the image to load.

        Saves
        -----
        warm_pixels_cor : PixelLineCollection
            The set of consistent warm pixel trails, saved to
            dataset.saved_consistent_lines(use_corrected=True).
        """
        warm_pixels = self.raw_process.consistent_lines_for(quadrant)

        # Corrected images
        warm_pixels_cor = PixelLineCollection()
        for i, image in enumerate(self.dataset):
            image_name = image.name
            print(
                f"\r    {image_name}_cor_{quadrant} ({i + 1} of {len(self.dataset)}) ",
                end="",
                flush=True,
            )

            # Load the image
            array = image.load_quadrant(quadrant)

            # Select consistent warm pixels found from this image
            image_name_q = f"{image_name}_{quadrant}"
            sel = np.where(warm_pixels.origins == image_name_q)[0]
            for i in sel:
                line = warm_pixels.lines[i]
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

        print("Extracted %d lines" % warm_pixels_cor.n_lines)

        # Save
        return warm_pixels_cor
