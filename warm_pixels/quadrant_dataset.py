from warm_pixels.process.group import Group
from warm_pixels.process.quadrant import Quadrant, CorrectedQuadrant
from warm_pixels.quadrant_groups import QuadrantsString


class QuadrantDataset:
    def __init__(
            self,
            dataset,
            quadrants_string: QuadrantsString,
            use_corrected
    ):
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
    def all_quadrants(self):
        return [
            quadrant
            for group in self.groups
            for quadrant in group.quadrants
        ]
