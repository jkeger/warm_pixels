import os
import pickle
from functools import wraps
from pathlib import Path

from warm_pixels import hst_utilities as hu


def cache(func):
    """
    Cache the results of a method that takes no arguments
    """

    @wraps(func)
    def wrapper(*args):
        self = args[0]
        key = f"__{func.__name__}"
        if key not in self.__dict__:
            self.__dict__[key] = func(self)
        return self.__dict__[key]

    return wrapper


class Persist:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, func):
        name = f"{func.__name__}.pickle"

        def wrapper(instance):
            directory = self.path / str(instance)
            path = directory / name
            os.makedirs(directory, exist_ok=True)

            if path.exists():
                with open(path, "b+r") as f:
                    return pickle.load(f)

            result = func(instance)
            with open(path, "b+w") as f:
                pickle.dump(result, f)

            return result

        return wrapper


persist = Persist(hu.output_path / "cache")
