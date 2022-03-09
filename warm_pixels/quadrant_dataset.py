from typing import List

from warm_pixels.dataset import Dataset
from warm_pixels.model.group import QuadrantGroup
from warm_pixels.model.quadrant import DatasetQuadrant, CorrectedQuadrant


class QuadrantDataset:
    def __init__(
            self,
            dataset: Dataset,
            quadrants_string: str,
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
        QuadrantClass = DatasetQuadrant
        if use_corrected:
            QuadrantClass = CorrectedQuadrant
            dataset = dataset.corrected()

        self.groups = [
            QuadrantGroup(
                dataset,
                [
                    QuadrantClass(
                        quadrant=quadrant,
                        dataset=dataset
                    )
                    for quadrant in group
                ])
            for group in tuple(map(
                tuple,
                quadrants_string.split("_")
            ))
        ]

    @property
    def all_quadrants(self) -> List[DatasetQuadrant]:
        """
        A list of all quadrants
        """
        return [
            quadrant
            for group in self.groups
            for quadrant in group.quadrants
        ]
