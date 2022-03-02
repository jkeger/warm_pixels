from .abstract import AbstractProcess
from .quadrant import CorrectedQuadrant


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
        """
        return CorrectedQuadrant(
            quadrant=quadrant,
            dataset=self.dataset
        ).consistent_lines()
