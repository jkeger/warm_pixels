from typing import Tuple


class QuadrantsString:
    def __init__(self, string: str):
        """
        Parse a string indicating which quadrants should be grouped together

        e.g. AB_CD has four quadrants A, B, C and D and two groups AB and CD

        Parameters
        ----------
        string
            A string listing quadrants
        """
        self.string = string

    @property
    def groups(self) -> Tuple[Tuple[str, ...], ...]:
        """
        Tuples of the name of quadrants in each group
        """
        return tuple(map(
            tuple,
            self.string.split("_")
        ))

    def __iter__(self):
        for group in self.groups:
            for quadrant in group:
                yield quadrant

    def __len__(self):
        i = 0
        for i, _ in enumerate(self):
            pass
        return i + 1
