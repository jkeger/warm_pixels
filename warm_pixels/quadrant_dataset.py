from typing import List

from warm_pixels.process.group import Group
from warm_pixels.process.quadrant import Quadrant, CorrectedQuadrant
from warm_pixels.quadrant_groups import QuadrantsString
from warm_pixels.hst_data import Dataset


class QuadrantDataset:
    def __init__(
            self,
            dataset: Dataset,
            quadrants_string: QuadrantsString,
            use_corrected: bool,
    ):
        """
        Comprises a dataset with an object indicating which quadrants should be processed
        and how they should be processed.

        Parameters
        ----------
        dataset
            A dataset containing images associated with a single date
        quadrants_string
            An object indicating which quadrants should be processed and how they should
            be grouped
        use_corrected
            If True then images are corrected to test the efficacy of CTI removal
        """
        QuadrantClass = Quadrant
        if use_corrected:
            QuadrantClass = CorrectedQuadrant
            dataset = dataset.corrected()

        self.groups = [
            Group(
                dataset,
                [
                    QuadrantClass(
                        quadrant=quadrant,
                        dataset=dataset
                    )
                    for quadrant in group
                ])
            for group in quadrants_string.groups
        ]

    @property
    def all_quadrants(self) -> List[Quadrant]:
        """
        A list of all quadrants
        """
        return [
            quadrant
            for group in self.groups
            for quadrant in group.quadrants
        ]
