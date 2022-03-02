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
