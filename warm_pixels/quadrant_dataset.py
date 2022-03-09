from typing import List

from warm_pixels.data import Dataset
from warm_pixels.model.group import QuadrantGroup
from warm_pixels.model.quadrant import Quadrant


class QuadrantDataset:
    def __init__(
            self,
            dataset: Dataset,
            quadrants_string: str,
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
        """
        self.groups = [
            QuadrantGroup(
                dataset,
                [
                    Quadrant(
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
    def all_quadrants(self) -> List[Quadrant]:
        """
        A list of all quadrants
        """
        return [
            quadrant
            for group in self.groups
            for quadrant in group.quadrants
        ]
