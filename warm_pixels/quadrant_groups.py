class Quadrants:
    def __init__(self, string):
        self.string = string

    @property
    def groups(self):
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
